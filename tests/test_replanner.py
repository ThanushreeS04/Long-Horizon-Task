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


if __name__ == "__main__":
    unittest.main()
