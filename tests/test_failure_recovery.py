
import tempfile
import unittest
from pathlib import Path

from agents.executor import BenchmarkEnvironment
from agents.memory_agent import MemoryEnabledAgent
from agents.planner import sample_task


class FailingOnceEnvironment(BenchmarkEnvironment):
    """Simulate a temporary failure on the first execution attempt."""

    def __init__(self):
        super().__init__()
        self.failed_once = False

    def execute(self, step):
        if not self.failed_once:
            self.failed_once = True
            self.actions.append({"step_id": step["id"]})
            return {
                "success": False,
                "message": "Simulated temporary failure",
                "fact": None,
                "evidence": {"observed": False},
                "timestamp": "simulated",
            }

        return super().execute(step)


class FailureRecoveryTests(unittest.TestCase):
    def test_failed_step_remains_pending_and_can_be_retried(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            environment = FailingOnceEnvironment()
            task = sample_task()

            first_agent = MemoryEnabledAgent(
                environment,
                memory_path,
            )

            first_result = first_agent.run_task(task)

            self.assertFalse(first_result["completed"])

            saved_state = first_agent.memory.get_task(
                task["task_id"]
            )

            self.assertIn(
                task["steps"][0]["id"],
                saved_state["pending_step_ids"],
            )

            # Restart the agent using the same persistent memory.
            restarted_agent = MemoryEnabledAgent(
                environment,
                memory_path,
            )

            resumed_result = restarted_agent.resume_task(
                task["task_id"]
            )

            self.assertTrue(resumed_result["completed"])
            self.assertEqual(
                resumed_result["pending_steps"],
                [],
            )

            actions_for_first_step = [
                action
                for action in environment.actions
                if action["step_id"] == task["steps"][0]["id"]
            ]

            self.assertEqual(len(actions_for_first_step), 2)

    def test_auditor_rejects_missing_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            environment = BenchmarkEnvironment()
            task = sample_task()
            step = task["steps"][0]

            # Even a success message must not be enough without
            # the expected environment fact and evidence.
            result = {
                "success": True,
                "message": "Step succeeded",
                "fact": step["expected"],
                "evidence": {"observed": False},
            }

            from agents.auditor import Auditor

            audit = Auditor().verify(
                step,
                result,
                environment,
            )

            self.assertFalse(audit["verified"])

            agent = MemoryEnabledAgent(
                environment,
                memory_path,
            )
            state = agent.memory.create_or_load(task)

            state = agent.memory.record_attempt(
                state,
                step,
                result,
                audit,
            )

            self.assertIn(
                step["id"],
                state["pending_step_ids"],
            )
            self.assertNotEqual(state["status"], "completed")


if __name__ == "__main__":
    unittest.main()
