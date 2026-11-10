import copy
import tempfile
import unittest
from pathlib import Path

from agents.llm import MockLLM
from agents.llm_agents import OutputAuditor, make_plan, make_code
from agents.workflow import (WorkflowBaselineAgent, WorkflowMemoryAgent,
                             add_dependencies, make_alternative, ready_step)
from recovery_bench import FaultyLLM

GOAL = "Compare options, then write a recommendation"


def make_task(n, deps=True):
    steps, _ = make_plan(MockLLM(), GOAL, n)
    if deps:
        steps = add_dependencies(steps)
    return {"task_id": "t", "goal": GOAL, "success_criteria": [], "steps": steps}


class DependencyTests(unittest.TestCase):

    def test_alternative_repoints_dependents(self):
        plan = make_task(8)["steps"]
        failed = next(s for s in plan if s["id"] == "step-2")
        alt = make_alternative(plan, failed)
        step5 = next(s for s in plan if s["id"] == "step-5")
        self.assertEqual(step5["depends_on"], [alt["id"]])
        self.assertEqual(alt["depends_on"], failed["depends_on"])
        self.assertEqual(plan[plan.index(failed) + 1]["id"], alt["id"])


class FaultInjectionTests(unittest.TestCase):
    def test_auditor_rejects_injected_bad_output(self):
        task = make_task(4, deps=False)
        llm = FaultyLLM(MockLLM(), {1: 1})
        step = task["steps"][0]
        out = llm.complete("p", kind="step",
                           meta={"index": 1, "code": step["code"], "desc": ""})
        audit = OutputAuditor().verify(step, {"output": out["text"]})
        self.assertFalse(audit["verified"])
        out2 = llm.complete("p", kind="step",
                            meta={"index": 1, "code": step["code"], "desc": ""})
        self.assertTrue(OutputAuditor().verify(step, {"output": out2["text"]})["verified"])


class ReplanTests(unittest.TestCase):
    def test_replan_on_completes_with_alternative_step(self):
        task = make_task(8)
        llm = FaultyLLM(MockLLM(), {2: 99})
        with tempfile.TemporaryDirectory() as d:
            agent = WorkflowMemoryAgent(llm, Path(d) / "m.json", replan=True)
            r = agent.run(copy.deepcopy(task))
        self.assertTrue(r["completed"])
        self.assertEqual(r["replans"], 1)
        self.assertIn(make_code(9), r["final_output"])
        self.assertNotIn(make_code(2), r["final_output"])

    def test_baseline_replans_too(self):
        task = make_task(8)
        r = WorkflowBaselineAgent(FaultyLLM(MockLLM(), {2: 99}), replan=True).run(copy.deepcopy(task))
        self.assertTrue(r["completed"])


class PersistenceTests(unittest.TestCase):
    def test_memory_agent_remembers_failures_after_restart(self):
        task = make_task(5, deps=False)
        llm = FaultyLLM(MockLLM(), {1: 99})
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.json"
            WorkflowMemoryAgent(llm, path, replan=False).run(copy.deepcopy(task))
            used_before = llm.used[1]
            WorkflowMemoryAgent(llm, path, replan=False).run(copy.deepcopy(task))
        self.assertEqual(used_before, 3)
        self.assertEqual(llm.used[1], 3)

    def test_baseline_forgets_failures_after_restart(self):
        task = make_task(5, deps=False)
        llm = FaultyLLM(MockLLM(), {1: 99})
        WorkflowBaselineAgent(llm, replan=False).run(copy.deepcopy(task))
        WorkflowBaselineAgent(llm, replan=False).run(copy.deepcopy(task))
        self.assertEqual(llm.used[1], 6)

    def test_resume_does_not_repeat_verified_steps(self):
        task = make_task(8)
        llm = MockLLM()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.json"
            first = WorkflowMemoryAgent(llm, path).run(copy.deepcopy(task), stop_after=3)
            second = WorkflowMemoryAgent(llm, path).run(copy.deepcopy(task))
        done_first = {s for s, ok in first["executed"] if ok}
        done_second = {s for s, ok in second["executed"] if ok}
        self.assertEqual(done_first & done_second, set())
        self.assertTrue(second["completed"])


if __name__ == "__main__":
    unittest.main()
