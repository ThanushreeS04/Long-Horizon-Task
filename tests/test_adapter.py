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

    def test_single_router_uses_one_model_everywhere(self):
        m = CountingLLM()
        r = ModelRouter.single(m)
        self.assertTrue(all(r.for_role(x) is m for x in ("manager", "executor", "auditor")))


class AdapterTests(unittest.TestCase):

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


class LedgerTests(unittest.TestCase):

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

    def test_default_router_keeps_old_behaviour(self):
        steps, _ = make_plan(MockLLM(), "Compare options, then recommend", 5)
        task = {"task_id": "t", "goal": "g", "success_criteria": [], "steps": steps}
        a = MEAAgent(MockLLM(), Path(tempfile.mkdtemp()) / "a.json").run(task)
        b = MEAAgent(MockLLM(), Path(tempfile.mkdtemp()) / "b.json", router=None).run(task)
        self.assertEqual((a["input_tokens"], a["llm_calls"]), (b["input_tokens"], b["llm_calls"]))


if __name__ == "__main__":
    unittest.main()
