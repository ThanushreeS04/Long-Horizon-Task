import tempfile
import time
import unittest
from pathlib import Path

from agents.adapter import (AgentAdapter, Ledger, ModelRouter, RoleLLM, format_ledger,
                            merge_ledgers)
from agents.llm import MockLLM
from agents.llm_agents import make_plan
from agents.mea import MEAAgent


class CountingLLM(MockLLM):
    def __init__(self, label="m"):
        super().__init__()
        self.label, self.calls, self.max_tokens_seen = label, 0, []

    def complete(self, prompt, kind="step", max_tokens=500, meta=None):
        self.calls += 1
        self.max_tokens_seen.append(max_tokens)
        return super().complete(prompt, kind=kind, max_tokens=max_tokens, meta=meta)


class SlowLLM(CountingLLM):
    def complete(self, *a, **kw):
        time.sleep(0.02)
        return super().complete(*a, **kw)


META = {"index": 1, "code": "K1-aaaa", "desc": "do something"}


class RouterTests(unittest.TestCase):
    def test_roles_use_their_own_model_and_others_use_default(self):
        d, e = CountingLLM("d"), CountingLLM("e")
        r = ModelRouter(d, executor=e)
        self.assertIs(r.for_role("executor"), e)
        self.assertIs(r.for_role("manager"), d)
        self.assertIs(r.for_role("auditor"), d)

    def test_single_router_uses_one_model_everywhere(self):
        m = CountingLLM()
        r = ModelRouter.single(m)
        self.assertTrue(all(r.for_role(x) is m for x in ("manager", "executor", "auditor")))


class AdapterTests(unittest.TestCase):
    def test_call_is_recorded_under_the_right_role(self):
        m = CountingLLM()
        a = AgentAdapter(ModelRouter.single(m))
        res = a.run_episode("auditor", "prompt text", kind="step", meta=META)
        row = a.ledger.rows["auditor"]
        self.assertEqual((row["calls"], a.ledger.rows["executor"]["calls"]), (1, 0))
        self.assertEqual(row["input_tokens"], res["input_tokens"])
        self.assertEqual(res["role"], "auditor")

    def test_budget_max_tokens_reaches_the_model(self):
        m = CountingLLM()
        a = AgentAdapter(ModelRouter.single(m))
        a.run_episode("executor", "p", {"max_tokens": 123}, meta=META)
        self.assertEqual(m.max_tokens_seen, [123])

    def test_time_overrun_is_flagged_not_hidden(self):
        a = AgentAdapter(ModelRouter.single(SlowLLM()))
        res = a.run_episode("executor", "p", {"seconds": 0.001}, meta=META)
        self.assertTrue(res["over_budget"])
        self.assertEqual(a.ledger.rows["executor"]["over_budget"], 1)

    def test_within_budget_is_not_flagged(self):
        a = AgentAdapter(ModelRouter.single(CountingLLM()))
        self.assertFalse(a.run_episode("executor", "p", meta=META)["over_budget"])

    def test_role_view_returns_the_same_tokens_as_the_model(self):
        plain, routed = CountingLLM(), CountingLLM()
        direct = plain.complete("hello", kind="step", max_tokens=400, meta=META)
        via = RoleLLM(AgentAdapter(ModelRouter.single(routed)), "executor").complete(
            "hello", kind="step", max_tokens=400, meta=META)
        self.assertEqual((direct["text"], direct["input_tokens"], direct["output_tokens"]),
                         (via["text"], via["input_tokens"], via["output_tokens"]))


class LedgerTests(unittest.TestCase):
    def test_rule_based_time_has_seconds_but_no_calls_or_tokens(self):
        led = Ledger()
        with led.time_block("manager"):
            time.sleep(0.005)
        s = led.summary()["manager"]
        self.assertEqual((s["calls"], s["tokens"]), (0, 0))
        self.assertGreater(s["seconds"], 0)

    def test_token_shares_add_up_to_one(self):
        led = Ledger()
        led.add_call("executor", 300, 100, 1.0)
        led.add_call("auditor", 80, 20, 0.5)
        shares = [r["token_share"] for r in led.summary().values()]
        self.assertAlmostEqual(sum(shares), 1.0, places=3)
        self.assertEqual(led.summary()["executor"]["token_share"], 0.8)

    def test_merge_adds_sessions(self):
        a, b = Ledger(), Ledger()
        a.add_call("executor", 100, 50, 1.0)
        b.add_call("executor", 200, 70, 2.0)
        m = merge_ledgers([a.summary(), b.summary()])
        self.assertEqual((m["executor"]["calls"], m["executor"]["input_tokens"]), (2, 300))
        self.assertEqual(merge_ledgers([None, None]), None)

    def test_format_lists_every_role(self):
        led = Ledger()
        led.add_call("executor", 10, 5, 0.1)
        text = format_ledger(led.summary())
        for role in ("manager", "executor", "auditor"):
            self.assertIn(role, text)


class MEAIntegrationTests(unittest.TestCase):
    def _run(self, router, n=5):
        steps, _ = make_plan(MockLLM(), "Compare options, then recommend", n)
        task = {"task_id": "t", "goal": "Compare options, then recommend",
                "success_criteria": [], "steps": steps}
        path = Path(tempfile.mkdtemp()) / "m.json"
        return MEAAgent(MockLLM(), path, router=router).run(task)

    def test_executor_model_gets_every_call(self):
        default, exe = CountingLLM("default"), CountingLLM("exec")
        rep = self._run(ModelRouter(default, executor=exe))
        self.assertEqual(default.calls, 0)
        self.assertEqual(exe.calls, rep["llm_calls"])
        self.assertTrue(rep["completed"])

    def test_report_ledger_matches_run_totals(self):
        rep = self._run(None)
        led = rep["ledger"]
        self.assertEqual(led["executor"]["calls"], rep["llm_calls"])
        self.assertEqual(led["executor"]["input_tokens"], rep["input_tokens"])
        self.assertEqual(rep["calls_by_role"]["executor"], rep["llm_calls"])
        self.assertEqual((led["manager"]["calls"], led["auditor"]["calls"]), (0, 0))
        self.assertGreater(led["manager"]["seconds"], 0)    # rule-based work is timed

    def test_default_router_keeps_old_behaviour(self):
        steps, _ = make_plan(MockLLM(), "Compare options, then recommend", 5)
        task = {"task_id": "t", "goal": "g", "success_criteria": [], "steps": steps}
        a = MEAAgent(MockLLM(), Path(tempfile.mkdtemp()) / "a.json").run(task)
        b = MEAAgent(MockLLM(), Path(tempfile.mkdtemp()) / "b.json", router=None).run(task)
        self.assertEqual((a["input_tokens"], a["llm_calls"]), (b["input_tokens"], b["llm_calls"]))


if __name__ == "__main__":
    unittest.main()