import json
import tempfile
import unittest
from pathlib import Path

from agents.llm import MockLLM
from agents.llm_agents import make_plan
from agents.mea import MEAAgent


class ScriptLLM(MockLLM):
    """MockLLM with switches for the replan and audit answers; records every prompt."""

    def __init__(self, replan_text=None, audit_text=None, bad_first_step=False):
        super().__init__()
        self.replan_text, self.audit_text = replan_text, audit_text
        self.bad_first_step, self.step_calls = bad_first_step, 0
        self.prompts = []

    def complete(self, prompt, kind="step", max_tokens=500, meta=None):
        self.prompts.append((kind, prompt))
        if kind == "replan" and self.replan_text is not None:
            return {"text": self.replan_text, "input_tokens": 10, "output_tokens": 5}
        if kind == "audit" and self.audit_text is not None:
            return {"text": self.audit_text, "input_tokens": 10, "output_tokens": 5}
        if kind == "step":
            self.step_calls += 1
            if self.bad_first_step and self.step_calls == 1:
                return {"text": "I think this is fine.", "input_tokens": 5, "output_tokens": 5}
        return super().complete(prompt, kind=kind, max_tokens=max_tokens, meta=meta)


def make_task(n=6):
    steps, _ = make_plan(MockLLM(), "Compare options, then recommend", n)
    return {"task_id": "t", "goal": "Compare options, then recommend",
            "success_criteria": [], "steps": steps}


def run(llm, n=6, path=None, **kw):
    path = path or Path(tempfile.mkdtemp()) / "m.json"
    return MEAAgent(llm, path, replan=False, **kw).run(make_task(n)), path


class LLMReplanTests(unittest.TestCase):
    def test_off_by_default_makes_no_manager_calls(self):
        rep, _ = run(ScriptLLM())
        self.assertEqual(rep["llm_replan_calls"], 0)
        self.assertEqual(rep["calls_by_role"]["manager"], 0)

    def test_always_mode_is_capped_at_three_calls(self):
        rep, _ = run(ScriptLLM(), n=8, llm_replan="always")
        self.assertEqual(rep["llm_replan_calls"], 3)
        self.assertEqual(rep["calls_by_role"]["manager"], 3)
        self.assertTrue(rep["completed"])

    def test_on_reject_does_nothing_when_nothing_is_rejected(self):
        rep, _ = run(ScriptLLM(), llm_replan="on_reject")
        self.assertEqual(rep["llm_replan_calls"], 0)

    def test_on_reject_runs_after_a_rejection_and_sees_the_reason(self):
        llm = ScriptLLM(bad_first_step=True)
        rep, _ = run(llm, llm_replan="on_reject")
        self.assertEqual(rep["llm_replan_calls"], 1)
        prompt = next(p for k, p in llm.prompts if k == "replan")
        self.assertIn("Latest audit: status=incomplete", prompt)
        self.assertIn("FACT-1", prompt)

    def test_applied_rewrite_is_saved_and_keeps_the_original(self):
        rep, path = run(ScriptLLM(), llm_replan="always")
        plan = json.loads(path.read_text())["t"]["plan"]
        changed = [s for s in plan if "original_description" in s]
        self.assertTrue(changed)
        self.assertIn("(revised after audit)", changed[0]["description"])

    def test_only_descriptions_change_never_codes_or_the_final_step(self):
        before = {s["id"]: (s["code"], s["final"]) for s in make_task()["steps"]}
        rep, path = run(ScriptLLM(), llm_replan="always")
        plan = json.loads(path.read_text())["t"]["plan"]
        self.assertEqual({s["id"]: (s["code"], s["final"]) for s in plan}, before)
        final = next(s for s in plan if s["final"])
        self.assertNotIn("original_description", final)

    def test_wrong_step_count_leaves_the_plan_untouched(self):
        rep, path = run(ScriptLLM(replan_text='["only one"]'), llm_replan="always")
        self.assertEqual(rep["llm_replans"], 0)
        self.assertIn("expected", rep["llm_replan_log"][0]["why"])
        plan = json.loads(path.read_text())["t"]["plan"]
        self.assertFalse(any("original_description" in s for s in plan))
        self.assertTrue(rep["completed"])

    def test_descriptions_with_codes_are_rejected(self):
        text = json.dumps(["do it K9-abcd"] * 5)
        rep, _ = run(ScriptLLM(replan_text=text), llm_replan="always")
        self.assertEqual(rep["llm_replans"], 0)
        self.assertIn("code", rep["llm_replan_log"][0]["why"])

    def test_cap_survives_a_restart(self):
        path = Path(tempfile.mkdtemp()) / "m.json"
        task = make_task(8)
        first = MEAAgent(ScriptLLM(), path, replan=False, llm_replan="always").run(task, stop_after=2)
        second = MEAAgent(ScriptLLM(), path, replan=False, llm_replan="always").run(task)
        self.assertEqual(first["llm_replan_calls"] + second["llm_replan_calls"], 3)

    def test_totals_include_manager_tokens(self):
        rep, _ = run(ScriptLLM(), llm_replan="always")
        led = rep["ledger"]
        self.assertEqual(rep["llm_calls"], led["executor"]["calls"] + led["manager"]["calls"])
        self.assertGreater(led["manager"]["input_tokens"], 0)
        self.assertEqual(rep["input_tokens"], sum(v["input_tokens"] for v in led.values()))

    def test_invalid_mode_is_refused(self):
        with self.assertRaises(ValueError):
            MEAAgent(MockLLM(), Path(tempfile.mkdtemp()) / "m.json", llm_replan="sometimes")


class LLMAuditTests(unittest.TestCase):
    def test_off_by_default(self):
        rep, path = run(ScriptLLM())
        self.assertEqual(rep["llm_audit_checked"], 0)
        hist = json.loads(path.read_text())["t"]["step_history"]
        self.assertTrue(all("llm_check" not in h["evidence"] for h in hist))

    def test_every_verified_step_is_checked_and_stored_as_evidence(self):
        rep, path = run(ScriptLLM(), llm_audit=True)
        self.assertEqual(rep["llm_audit_checked"], 6)
        self.assertEqual(rep["calls_by_role"]["auditor"], 6)
        hist = json.loads(path.read_text())["t"]["step_history"]
        self.assertTrue(all(h["evidence"]["llm_check"]["agrees"] for h in hist))

    def test_disagreement_is_counted_but_never_blocks(self):
        rep, _ = run(ScriptLLM(audit_text="NO. Looks empty."), llm_audit=True)
        self.assertEqual(rep["llm_audit_disagreements"], 6)
        self.assertTrue(rep["completed"])
        self.assertTrue(all(ok for _, ok in rep["executed"]))      # no step was rejected

    def test_unclear_answer_is_neither_agree_nor_disagree(self):
        rep, _ = run(ScriptLLM(audit_text="Hard to say."), llm_audit=True)
        self.assertEqual(rep["llm_audit_disagreements"], 0)
        self.assertTrue(all(c["agrees"] is None for c in rep["llm_audit_log"]))

    def test_audit_calls_are_in_the_totals(self):
        rep, _ = run(ScriptLLM(), llm_audit=True)
        self.assertEqual(rep["llm_calls"], 12)          # 6 executor + 6 auditor


if __name__ == "__main__":
    unittest.main()