import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import memory_baselines as mb
from agents.llm import MockLLM


class Baselines(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.task, agent, self.rep = mb.run_agent(MockLLM(), 20, self.tmp.name)
        self.state = agent.memory.get_task(self.task["task_id"])

    def test_the_run_used_for_replay_completed(self):
        self.assertTrue(self.rep["completed"])

    def test_dependency_keeps_every_needed_fact_window_does_not(self):
        r = mb.evaluate(self.task, self.state, budget=2000, window=5)
        self.assertEqual(r["dependency"]["min_recall"], 1.0)
        self.assertEqual(r["full_log"]["min_recall"], 1.0)
        self.assertLess(r["window_k"]["final_recall"], 0.5)        # the final report needs every fact

    def test_small_budget_makes_bounded_context_lose_facts_but_stay_in_budget(self):
        r = mb.evaluate(self.task, self.state, budget=700, window=5)
        self.assertLess(r["bounded"]["final_recall"], 1.0)
        self.assertEqual(r["bounded"]["steps_over_budget"], 0)

    def test_io_cost_reports_growth(self):
        io = mb.io_cost(self.task, self.state, 2000, self.tmp.name)
        self.assertGreater(io["write_ms_avg"], 0)
        self.assertGreater(io["peak_ram_kib"], 0)
        self.assertGreater(io["state_chars_final"], 1000)


if __name__ == "__main__":
    unittest.main()