"""M4 - a model replan changes the plan, and mem.rollback() undoes it from a checkpoint."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.adapter import AgentAdapter, ModelRouter
from agents.cot_executor import CoTExecutor
from agents.durable import make_durable
from agents.hl_planner import HighLevelPlanner
from agents.llm_auditor import LLMAuditor
from agents.llm_manager import LLMManager
from agents.mea import MEAAgent
from agents.observation_gc import ObservationGC
from agents.state_replanner import StateReplanner
from test_agentic import Scripted

GOAL = "Plan a launch, budget it, and announce it"
NEW_STEP = "Check legal requirements"


def build(llm, path, steps=None):
    adapter = AgentAdapter(ModelRouter.single(llm))
    steps = steps or HighLevelPlanner(adapter).plan(GOAL, 5)["steps"]
    agent = MEAAgent(llm, path, adapter=adapter, manager=LLMManager(adapter), auditor=LLMAuditor(adapter),
                     executor=CoTExecutor(adapter.for_role("executor"), gc=ObservationGC()),
                     state_replanner=StateReplanner(adapter))
    task = {"task_id": "t", "goal": GOAL, "success_criteria": [], "steps": steps}
    return make_durable(agent), task


def add_once(p, m, n):                       # first call: add a step
    d = list(m["descriptions"])
    return json.dumps(d + [NEW_STEP] if n == 1 else d)


def drop_once(p, m, n):                      # first call: drop the last non-final pending step
    d = list(m["descriptions"])
    return json.dumps(d[:-1] if n == 1 else d)


def keep(p, m, n):
    return json.dumps(m["descriptions"])


class ReplanRollback(unittest.TestCase):
    def _first_replan(self, fn, tmp):
        path = Path(tmp) / "m.json"
        agent, task = build(Scripted(state_replan=fn), path)
        agent.run(task, stop_after=1)                       # one verified step, then the replan, then stop
        return path, task, agent.memory

    def _before_after(self, mem):
        ck = mem.checkpoints("t").list()
        seqs = [s for s, reason, _, ok in ck if ok]
        replan_seq = next(s for s, reason, _, ok in ck if reason == "replan")
        before = max(s for s in seqs if s < replan_seq)
        return before, replan_seq

    def test_added_step_creates_a_replan_checkpoint_and_rolls_back(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path, task, mem = self._first_replan(add_once, tmp)
            state = mem.get_task("t")
            self.assertTrue(any(e["applied"] for e in state["llm_replans"]))
            self.assertTrue(any(s.get("added_by_replan") for s in state["plan"]))
            before, _ = self._before_after(mem)

            restored = mem.rollback("t", before)
            self.assertFalse(any(s.get("added_by_replan") for s in restored["plan"]))
            self.assertEqual(restored["llm_replans"], [])
            self.assertNotIn(NEW_STEP, [s["description"] for s in mem.get_task("t")["plan"]])
            self.assertIn("rollback", [e["event"] for e in mem.journal.read()])

    def test_run_after_rollback_finishes_without_the_undone_step(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path, task, mem = self._first_replan(add_once, tmp)
            state = mem.get_task("t")
            added = next(s for s in state["plan"] if s.get("added_by_replan"))
            before, _ = self._before_after(mem)
            mem.rollback("t", before)
            agent2, task2 = build(Scripted(state_replan=keep), path, task["steps"])
            rep = agent2.run(task2)                           # resumes from the rolled-back state
            self.assertTrue(rep["completed"])
            self.assertNotIn(added["code"], rep["final_output"])
            self.assertFalse(any(s.get("added_by_replan") for s in agent2.memory.get_task("t")["plan"]))

    def test_dropped_step_is_also_checkpointed_and_restorable(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path, task, mem = self._first_replan(drop_once, tmp)
            state = mem.get_task("t")
            self.assertEqual(len(state["replaced_step_ids"]), 1)
            before, _ = self._before_after(mem)               # fails if a drop makes no 'replan' checkpoint
            restored = mem.rollback("t", before)
            self.assertEqual(restored.get("replaced_step_ids", []), [])
            self.assertEqual(len(restored["pending_step_ids"]), len(state["pending_step_ids"]) + 1)

    def test_rejected_replan_leaves_plan_and_makes_no_checkpoint(self):
        bad = lambda p, m, n: json.dumps(m["descriptions"] + ["a", "b", "c", "d"])
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path, task, mem = self._first_replan(bad, tmp)
            state = mem.get_task("t")
            self.assertFalse(any(e["applied"] for e in state["llm_replans"]))
            self.assertNotIn("replan", [r for _, r, _, _ in mem.checkpoints("t").list()])


if __name__ == "__main__":
    unittest.main()