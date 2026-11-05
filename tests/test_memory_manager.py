
import json
import tempfile
import unittest
from pathlib import Path

from agents.executor import DemoEnvironment
from agents.memory_agent import MemoryEnabledAgent
from agents.memory_manager import MemoryManager
from agents.planner import sample_task


class MemoryTests(unittest.TestCase):
    def test_create_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.json"
            manager = MemoryManager(path)
            task = sample_task()

            state = manager.create_or_load(task)

            self.assertEqual(state["task_id"], task["task_id"])
            self.assertEqual(
                len(manager.get_task(task["task_id"])["pending_step_ids"]),
                4,
            )

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

    def test_summary_stays_within_character_limit(self):
        state = {
            "goal": "Complete a long task",
            "status": "in_progress",
            "summary": "x" * 2000,
            "pending_step_ids": ["step-10"],
            "step_history": [
                self.make_step(f"step-{i}")
                for i in range(1, 11)
            ],
            "errors": [],
            "findings": [],
        }

        self.manager._compact_if_needed(
            state,
            keep_recent=3,
            max_summary_chars=1200,
        )

        self.assertLessEqual(len(state["summary"]), 1200)

    def test_compaction_preserves_recent_history_and_pending_steps(self):
        state = {
            "goal": "Complete a long task",
            "success_criteria": ["All steps verified"],
            "status": "in_progress",
            "summary": "",
            "pending_step_ids": ["step-5", "step-6"],
            "step_history": [
                self.make_step("step-1"),
                self.make_step("step-2"),
                self.make_step("step-3"),
                self.make_step("step-4"),
                self.make_step("step-5", "attempted"),
                self.make_step("step-6", "failed"),
            ],
            "errors": [],
            "findings": [],
        }

        self.manager._compact_if_needed(
            state,
            keep_recent=3,
        )

        remaining_ids = [
            step["step_id"]
            for step in state["step_history"]
        ]

        self.assertEqual(
            remaining_ids,
            ["step-4", "step-5", "step-6"],
        )
        self.assertEqual(
            state["pending_step_ids"],
            ["step-5", "step-6"],
        )
        self.assertIn("step-1", state["summary"])
        self.assertIn("step-3", state["summary"])

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

    
    def test_summary_preserves_newer_information(self):
        state = {
            "goal": "Complete a long task",
            "success_criteria": [],
            "status": "in_progress",
            "summary": "OLD_INFORMATION " * 100,
            "pending_step_ids": ["step-6"],
            "step_history": [
                {
                    "step_id": f"step-{index}",
                    "description": f"Complete step {index}",
                    "status": (
                        "attempted" if index == 6 else "verified"
                    ),
                    "result": f"NEW_INFORMATION_{index}",
                }
                for index in range(1, 7)
            ],
            "errors": [],
            "findings": [],
        }

        self.manager._compact_if_needed(
            state,
            keep_recent=2,
            max_summary_chars=120,
        )

        self.assertLessEqual(len(state["summary"]), 120)

        remaining_ids = [
            step["step_id"]
            for step in state["step_history"]
        ]

        self.assertIn("step-6", remaining_ids)
        self.assertIn("step-5", remaining_ids)
        self.assertIn("step-6", state["pending_step_ids"])

        self.assertIn("step-6", remaining_ids)
        self.assertNotIn("step-6", state["summary"])



if __name__ == "__main__":
    unittest.main()
