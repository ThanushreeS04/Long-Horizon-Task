import tempfile
import unittest
from pathlib import Path

from agents.llm import MockLLM
from agents.llm_agents import make_plan
from agents.mea import MEAAgent
from agents.workflow import add_dependencies

GOAL = "Compare options, then write a recommendation"


def make_task(n, deps=True):
    steps, _ = make_plan(MockLLM(), GOAL, n)
    return {"task_id": "t", "goal": GOAL, "success_criteria": [],
            "steps": add_dependencies(steps) if deps else steps}


def tmp():
    return Path(tempfile.mkdtemp()) / "m.json"


class FailsFor(MockLLM):
    """Always fails the listed step indexes (kind 'step' and 'final')."""
    def __init__(self, indexes):
        self.indexes, self.prompts = set(indexes), []

    def complete(self, prompt, kind="step", max_tokens=500, meta=None):
        self.prompts.append(prompt)
        if meta and meta.get("index") in self.indexes:
            return {"text": "I could not do it.", "input_tokens": 1, "output_tokens": 1}
        return super().complete(prompt, kind, max_tokens, meta)


class ReplannerTests(unittest.TestCase):
    def test_failing_step_is_replaced_and_task_completes(self):
        rep = MEAAgent(FailsFor({2}), tmp()).run(make_task(6))
        self.assertTrue(rep["completed"])
        self.assertEqual(rep["replans"], 1)
        self.assertEqual(rep["replan_log"][0]["failed"], "step-2")

    def test_new_step_carries_the_rejection_reason(self):
        llm = FailsFor({2})
        MEAAgent(llm, tmp()).run(make_task(6))
        alt_prompts = [p for p in llm.prompts if "Alternative approach for step-2" in p]
        self.assertTrue(alt_prompts)
        self.assertIn("Earlier attempts were rejected", alt_prompts[0])
        self.assertIn("FACT-7", alt_prompts[0])        # reason now names the NEW label
        self.assertIn("K7-", alt_prompts[0])
        self.assertNotIn("K2-", alt_prompts[0])        # old step's code must not leak in

    def test_linear_plan_replan_updates_required_codes(self):
        rep = MEAAgent(FailsFor({2}), tmp()).run(make_task(6, deps=False))
        self.assertTrue(rep["completed"])

    def test_failing_final_step_is_replaced_by_a_final_step(self):
        rep = MEAAgent(FailsFor({6}), tmp()).run(make_task(6))
        self.assertTrue(rep["completed"])
        self.assertEqual(rep["replan_log"][0]["failed"], "step-6")

    def test_replan_budget_then_blocked(self):
        # step-1 and every alternative (7, 8, 9) fail: 3 replans, then blocked
        rep = MEAAgent(FailsFor({1, 7, 8, 9}), tmp()).run(make_task(6))
        self.assertEqual(rep["replans"], 3)
        self.assertFalse(rep["completed"])
        self.assertEqual(rep["stopped_because"], "blocked")

    def test_replans_survive_a_crash(self):
        path, task = tmp(), make_task(6)
        first = MEAAgent(FailsFor({2}), path).run(task, stop_after=3)
        second = MEAAgent(FailsFor({2}), path).run(task)
        self.assertTrue(second["completed"])
        saved = MEAAgent(MockLLM(), path).memory.get_task("t")
        self.assertEqual(len(saved["replans"]), first["replans"] + second["replans"])
        self.assertEqual(len(saved["replans"]), 1)

    def test_no_replan_when_everything_succeeds(self):
        rep = MEAAgent(MockLLM(), tmp()).run(make_task(8))
        self.assertEqual(rep["replans"], 0)


if __name__ == "__main__":
    unittest.main()