
import json
import tempfile
import unittest
from pathlib import Path

from agents.executor import DemoEnvironment
from agents.memory_agent import MemoryEnabledAgent
from agents.memory_manager import MemoryManager
from agents.planner import sample_task


class MemoryTests(unittest.TestCase):

    def test_resume_from_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.json"
            task = sample_task()
            env = DemoEnvironment()

            first = MemoryEnabledAgent(env, path).run_task(
                task,
                max_steps=2,
            )
            self.assertFalse(first["completed"])

            resumed = MemoryEnabledAgent(
                env,
                path,
            ).resume_task(task["task_id"])

            self.assertTrue(resumed["completed"])

            saved = json.loads(
                path.read_text(encoding="utf-8")
            )[task["task_id"]]

            self.assertEqual(saved["status"], "completed")
            self.assertEqual(saved["pending_step_ids"], [])

    def test_invalid_json_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.json"
            path.write_text("{broken", encoding="utf-8")

            manager = MemoryManager(path)

            with self.assertRaises(ValueError):
                manager.get_task("missing")


class TestBoundedMemoryCompaction(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        self.manager = MemoryManager(
            Path(self.temp_dir.name) / "memory.json"
        )

    def make_step(self, step_id, status="verified"):
        return {
            "step_id": step_id,
            "description": f"Complete {step_id}",
            "status": status,
            "result": f"Result for {step_id}",
        }

    def test_compact_context_limits_recent_items(self):
        state = {
            "goal": "Complete a long task",
            "success_criteria": [],
            "status": "in_progress",
            "summary": "x" * 2000,
            "pending_step_ids": ["step-10"],
            "step_history": [
                self.make_step(f"step-{i}")
                for i in range(1, 11)
            ],
            "errors": [
                {"reason": f"error-{i}"}
                for i in range(1, 6)
            ],
            "findings": [
                f"finding-{i}"
                for i in range(1, 9)
            ],
        }

        context = self.manager.build_compact_context(state)

        self.assertLessEqual(len(context["summary"]), 1200)
        self.assertEqual(len(context["recent_steps"]), 3)
        self.assertEqual(len(context["recent_errors"]), 3)
        self.assertEqual(len(context["findings"]), 5)
        self.assertEqual(
            context["pending_step_ids"],
            ["step-10"],
        )



if __name__ == "__main__":
    unittest.main()
