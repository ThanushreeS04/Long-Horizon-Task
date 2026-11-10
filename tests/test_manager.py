import copy
import tempfile
import unittest
from pathlib import Path

from agents.llm import MockLLM
from agents.llm_agents import make_plan
from agents.manager import ASK, BLOCKED, DONE, EXECUTE, Manager
from agents.memory_manager import MemoryManager
from agents.workflow import add_dependencies

GOAL = "Compare options, then write a recommendation"


def fresh_state(n=6):
    steps, _ = make_plan(MockLLM(), GOAL, n)
    task = {"task_id": "t", "goal": GOAL, "success_criteria": [],
            "steps": add_dependencies(steps)}
    path = Path(tempfile.mkdtemp()) / "m.json"
    mem = MemoryManager(path)
    return mem, mem.create_or_load(task)


def verify(mem, state, sid):
    step = next(s for s in state["plan"] if s["id"] == sid)
    result = {"success": True, "message": f"FACT for {sid}", "output": "x"}
    return mem.record_attempt(state, step, result, {"verified": True, "evidence": {}, "reason": "ok"})


def reject(mem, state, sid, reason="bad"):
    step = next(s for s in state["plan"] if s["id"] == sid)
    result = {"success": True, "message": "m", "output": "x"}
    return mem.record_attempt(state, step, result, {"verified": False, "evidence": {}, "reason": reason})


class ManagerTests(unittest.TestCase):

    def test_evidence_survives_history_compaction(self):
        mem, state = fresh_state(12)
        for i in range(1, 12):
            step = next(x for x in state["plan"] if x["id"] == f"step-{i}")
            res = {"success": True, "message": f"FACT-{i}: done {step['code']}", "output": "x"}
            state = mem.record_attempt(state, step, res,
                                       {"verified": True, "evidence": {}, "reason": "ok"})
        self.assertLess(len(state["step_history"]), 11)      # history really was compacted
        c = Manager().decide(state)["contract"]
        self.assertTrue(c["final"])
        self.assertEqual(len(c["evidence"]), 11)

if __name__ == "__main__":
    unittest.main()
