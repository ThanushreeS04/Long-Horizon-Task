import tempfile
import unittest
from pathlib import Path

from agents.llm import MockLLM
from agents.llm_agents import make_plan
from agents.manager import Manager
from agents.mea import Auditor, Executor, MEAAgent
from agents.workflow import add_dependencies

GOAL = "Compare options, then write a recommendation"


def make_task(n):
    steps, _ = make_plan(MockLLM(), GOAL, n)
    return {"task_id": "t", "goal": GOAL, "success_criteria": [],
            "steps": add_dependencies(steps)}


def tmp():
    return Path(tempfile.mkdtemp()) / "m.json"


class FlakyLLM(MockLLM):
    """Fails the first `times` calls for the given step index."""
    def __init__(self, index, times):
        self.index, self.times, self.prompts = index, times, []

    def complete(self, prompt, kind="step", max_tokens=500, meta=None):
        self.prompts.append(prompt)
        if meta and meta.get("index") == self.index and kind == "step" and self.times > 0:
            self.times -= 1
            return {"text": "I could not do it.", "input_tokens": 1, "output_tokens": 1}
        return super().complete(prompt, kind, max_tokens, meta)


class LoopTests(unittest.TestCase):

    def test_crash_and_resume_repeats_nothing(self):
        path, task = tmp(), make_task(8)
        first = MEAAgent(MockLLM(), path).run(task, stop_after=4)
        self.assertFalse(first["completed"])
        self.assertEqual(first["stopped_because"], "interrupted")
        second = MEAAgent(MockLLM(), path).run(task)
        self.assertTrue(second["completed"])
        ran = [s for s, _ in first["executed"] + second["executed"]]
        self.assertEqual(len(ran), len(set(ran)))

    def test_stops_blocked_when_step_keeps_failing(self):
        rep = MEAAgent(FlakyLLM(index=1, times=99), tmp(), replan=False).run(make_task(6))
        self.assertFalse(rep["completed"])
        self.assertEqual(rep["stopped_because"], "blocked")
        self.assertIn("step-1", rep["blocked"])

    def test_round_budget_stops_the_loop(self):
        rep = MEAAgent(MockLLM(), tmp(), n_max=3).run(make_task(8))
        self.assertEqual(rep["stopped_because"], "round_budget")
        self.assertEqual(rep["rounds"], 3)

    def test_ask_stops_and_saves_the_question(self):
        task = make_task(4)
        task["steps"][1]["needs_input"] = True
        task["steps"][1]["question"] = "Which region?"
        path = tmp()
        rep = MEAAgent(MockLLM(), path).run(task)
        self.assertEqual(rep["stopped_because"], "ask")
        self.assertEqual(MEAAgent(MockLLM(), path).memory.get_task("t")["question"], "Which region?")


class AuditorTests(unittest.TestCase):
    def setUp(self):
        self.agent = MEAAgent(MockLLM(), tmp())
        self.state = self.agent.memory.create_or_load(make_task(6))
        self.state.update(blocked_step_ids=[], replaced_step_ids=[], answers={})
        self.contract = Manager().decide(self.state)["contract"]
        self.step = self.state["plan"][0]

    def result(self, text):
        return {"success": True, "message": "m", "output": text}

    def good_text(self):
        return f"FACT-1: Step 1 established {self.step['code']}.\n" + "work " * 30

    def test_mutating_the_state_is_a_violation(self):
        class Bad(Auditor):
            def __init__(s):
                super().__init__()
                s.rules = type("R", (), {"verify": lambda self_, st, r: (
                    state_ref.update(status="hacked") or
                    {"verified": True, "evidence": {}, "reason": "ok"})})()
        state_ref = self.state
        a = Bad().audit(self.contract, self.step, self.result(self.good_text()), self.state)
        self.assertEqual(a["integrity"], "violation")
        self.assertFalse(a["verified"])


if __name__ == "__main__":
    unittest.main()
