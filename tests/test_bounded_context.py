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

    def test_essential_state_is_preserved(self):
        state = make_state(300, pending=("step-8", "step-9", "step-10", "step-11"),
                           errors=[("step-7", "Missing evidence")])
        ctx = self.mm.build_bounded_context(state, budget_chars=1200)
        self.assertEqual(ctx["goal"], "Prepare a long report")
        self.assertEqual(ctx["pending_count"], 4)
        self.assertEqual(ctx["next_pending"], ["step-8", "step-9", "step-10"])
        self.assertEqual(ctx["recent_errors"][-1]["reason"], "Missing evidence")

    def test_newest_facts_kept_in_full(self):
        ctx = self.mm.build_bounded_context(make_state(300), budget_chars=1500)
        self.assertIn("FACT-300: Step 300 established K300-abcd.", ctx["key_facts"])

    def test_dropped_facts_are_reported_not_hidden(self):
        ctx = self.mm.build_bounded_context(make_state(300), budget_chars=1000)
        self.assertGreater(ctx["older_facts_dropped"], 0)
        self.assertEqual(ctx["older_facts_dropped"] + len(ctx["key_facts"]), 300)

    def test_nothing_dropped_when_budget_is_large(self):
        ctx = self.mm.build_bounded_context(make_state(10), budget_chars=5000)
        self.assertEqual(ctx["older_facts_dropped"], 0)
        self.assertEqual(len(ctx["key_facts"]), 10)

    def test_old_facts_shortened_keep_start_and_end(self):
        long_fact = "FACT-1: " + "x" * 200 + " K1-abcd"
        state = make_state(10)
        state["findings"][0] = long_fact
        ctx = self.mm.build_bounded_context(state, budget_chars=5000)
        short = ctx["key_facts"][0]
        self.assertLessEqual(len(short), 48)
        self.assertTrue(short.startswith("FACT-1:"))
        self.assertTrue(short.endswith("K1-abcd"))

    def test_state_itself_is_not_modified(self):
        state = make_state(50)
        before = json.dumps(state, sort_keys=True)
        self.mm.build_bounded_context(state, budget_chars=1000)
        self.assertEqual(before, json.dumps(state, sort_keys=True))


if __name__ == "__main__":
    unittest.main()