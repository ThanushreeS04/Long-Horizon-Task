import json
import unittest

from agents.memory_manager import MemoryManager


def make_state(n_facts, pending=("step-99",), errors=()):
    return {
        "goal": "Prepare a long report",
        "status": "in_progress",
        "pending_step_ids": list(pending),
        "step_history": [{"step_id": f"step-{i}", "status": "verified"} for i in range(1, 6)],
        "errors": [{"step_id": s, "reason": r} for s, r in errors],
        "findings": [f"FACT-{i}: Step {i} established K{i}-abcd." for i in range(1, n_facts + 1)],
        "summary": "",
    }


class BoundedContextTests(unittest.TestCase):
    def setUp(self):
        self.mm = MemoryManager("data/_unused_test_memory.json")

    def test_size_stays_under_budget_for_any_length(self):
        for n in (5, 50, 500):
            ctx = self.mm.build_bounded_context(make_state(n), budget_chars=1500)
            self.assertLessEqual(len(json.dumps(ctx)), 1500)

    def test_dropped_facts_are_reported_not_hidden(self):
        ctx = self.mm.build_bounded_context(make_state(300), budget_chars=1000)
        self.assertGreater(ctx["older_facts_dropped"], 0)
        self.assertEqual(ctx["older_facts_dropped"] + len(ctx["key_facts"]), 300)


if __name__ == "__main__":
    unittest.main()
