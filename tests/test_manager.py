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
    def test_first_contract_has_acceptance_and_no_evidence(self):
        mem, state = fresh_state()
        d = Manager().decide(state)
        self.assertEqual(d["decision"], EXECUTE)
        c = d["contract"]
        self.assertEqual(c["step_id"], "step-1")
        self.assertEqual(len(c["acceptance"]), 3)
        self.assertEqual(c["evidence"], {})
        self.assertEqual(c["attempt"], 1)

    def test_dependency_evidence_is_included(self):
        mem, state = fresh_state(8)
        for s in ("step-1", "step-2", "step-3"):
            state = verify(mem, state, s)
        c = Manager().decide(state)["contract"]
        self.assertEqual(c["step_id"], "step-4")
        self.assertEqual(list(c["evidence"]), ["step-1"])

    def test_rejection_reason_and_attempt_count(self):
        mem, state = fresh_state()
        state = reject(mem, state, "step-1", "missing code")
        c = Manager().decide(state)["contract"]
        self.assertEqual(c["attempt"], 2)
        self.assertEqual(c["last_rejection"], "missing code")

    def test_done_when_nothing_pending(self):
        mem, state = fresh_state(4)
        for s in ("step-1", "step-2", "step-3", "step-4"):
            state = verify(mem, state, s)
        self.assertEqual(Manager().decide(state)["decision"], DONE)

    def test_final_contract_lists_all_codes(self):
        mem, state = fresh_state(4)
        for s in ("step-1", "step-2", "step-3"):
            state = verify(mem, state, s)
        c = Manager().decide(state)["contract"]
        self.assertTrue(c["final"])
        self.assertIn("all 3 step codes", c["acceptance"][0])

    def test_exhausted_step_is_skipped_then_blocked(self):
        mem, state = fresh_state(4)
        for _ in range(3):
            state = reject(mem, state, "step-1")
        d = Manager().decide(state)
        self.assertEqual(d["exhausted"], ["step-1"])
        self.assertEqual(d["contract"]["step_id"], "step-2")  # independent chain still runs
        for s in ("step-2", "step-3"):
            state = verify(mem, state, s)
        d = Manager().decide(state)
        self.assertEqual(d["decision"], BLOCKED)
        self.assertIn("step-1", d["reason"])

    def test_ask_when_input_missing_then_execute_after_answer(self):
        mem, state = fresh_state(4)
        state["plan"][0]["needs_input"] = True
        state["plan"][0]["question"] = "Which region?"
        d = Manager().decide(state)
        self.assertEqual((d["decision"], d["reason"]), (ASK, "Which region?"))
        state["answers"] = {"step-1": "EU"}
        self.assertEqual(Manager().decide(state)["decision"], EXECUTE)

    def test_decide_does_not_change_state(self):
        mem, state = fresh_state()
        before = copy.deepcopy(state)
        Manager().decide(state)
        self.assertEqual(before, state)

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

    def test_linear_plan_final_step_still_gets_all_evidence(self):
        steps, _ = make_plan(MockLLM(), GOAL, 5)           # no depends_on anywhere
        task = {"task_id": "t", "goal": GOAL, "success_criteria": [], "steps": steps}
        mem = MemoryManager(Path(tempfile.mkdtemp()) / "m.json")
        state = mem.create_or_load(task)
        for i in range(1, 5):
            step = state["plan"][i - 1]
            res = {"success": True, "message": f"FACT-{i}: ok {step['code']}", "output": "x"}
            state = mem.record_attempt(state, step, res,
                                       {"verified": True, "evidence": {}, "reason": "ok"})
        c = Manager().decide(state)["contract"]
        self.assertTrue(c["final"])
        self.assertEqual(len(c["evidence"]), 4)
        self.assertEqual(len(c["depends_on"]), 4)

if __name__ == "__main__":
    unittest.main()