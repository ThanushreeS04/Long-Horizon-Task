import tempfile
import unittest
from pathlib import Path

from agents.durable import DurableMemoryManager
from agents.llm import MockLLM
from agents.llm_agents import make_plan
from agents.workflow import add_dependencies

GOAL = "Compare options, then write a recommendation"


class DropCheckpointTests(unittest.TestCase):
    def test_dropping_a_step_also_makes_a_replan_checkpoint(self):
        steps, _ = make_plan(MockLLM(), GOAL, 6)
        task = {"task_id": "t", "goal": GOAL, "success_criteria": [], "steps": add_dependencies(steps)}
        mem = DurableMemoryManager(Path(tempfile.mkdtemp()) / "m.json")
        state = mem.create_or_load(task)
        mem.save_task(state)
        sid = state["pending_step_ids"][2]
        state["pending_step_ids"].remove(sid)                  # what StateReplanner does to a dropped step
        state.setdefault("replaced_step_ids", []).append(sid)
        next(s for s in state["plan"] if s["id"] == sid)["dropped_by_replan"] = True
        mem.save_task(state)
        reasons = [reason for _, reason, _, _ in mem.checkpoints("t").list()]
        self.assertIn("replan", reasons)
        mem.rollback("t", 1)
        self.assertIn(sid, mem.get_task("t")["pending_step_ids"])


if __name__ == "__main__":
    unittest.main()