import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from agents.llm import MockLLM
from bench import one_trial

AGENTS = ["baseline", "baseline-window", "memory+criteria", "mea"]


class BenchTrialTests(unittest.TestCase):
    def setUp(self):
        with contextlib.redirect_stdout(io.StringIO()):      # bench prints progress lines
            self.rows = one_trial(MockLLM(), "Compare options, then recommend", 8, 4, 4000,
                                  Path(tempfile.mkdtemp()), "t", AGENTS, window=3)

    def test_every_requested_agent_is_run(self):
        self.assertEqual(list(self.rows), AGENTS)

    def test_full_baseline_and_memory_agents_complete(self):
        for name in ("baseline", "memory+criteria", "mea"):
            self.assertEqual(self.rows[name]["completed"], 1, name)

    def test_window_baseline_forgets_early_facts_and_fails(self):
        row = self.rows["baseline-window"]
        self.assertEqual(row["completed"], 0)
        self.assertLess(row["recall_pct"], 100)
        self.assertIn("failed_report", row)

    def test_passing_trials_store_no_failed_report(self):
        self.assertNotIn("failed_report", self.rows["mea"])

    def test_crash_costs_only_the_baselines_repeats(self):
        self.assertEqual(self.rows["mea"]["repeated_steps"], 0)
        self.assertEqual(self.rows["memory+criteria"]["repeated_steps"], 0)
        self.assertGreater(self.rows["baseline"]["repeated_steps"], 0)


if __name__ == "__main__":
    unittest.main()