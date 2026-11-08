"""Tests for the how-to guide file, fact splitting, the window baseline and the solve.py command."""
import io
import contextlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import solve
from agents.guide import build_guide, slugify, write_guide
from agents.llm import MockLLM
from agents.llm_agents import LLMBaselineAgent, LLMMemoryAgent, make_code, make_plan, split_fact

STEP = {"id": "step-3", "code": make_code(3)}
CODE = make_code(3)


class SplitFactTests(unittest.TestCase):
    def test_same_line_code_is_removed(self):
        fact, body = split_fact(f"FACT-3: Python is installed ({CODE}).\nUse the installer.", STEP)
        self.assertEqual(fact, "Python is installed")
        self.assertEqual(body, "Use the installer.")

    def test_code_on_second_line(self):
        fact, body = split_fact(f"FACT-3:\nThe key identifier is {CODE}.\nBody text.", STEP)
        self.assertNotIn(CODE, fact + body)
        self.assertEqual(body, "Body text.")

    def test_markdown_bold_label_is_handled(self):
        fact, body = split_fact(f"**FACT-3: Install Python 3 ({CODE})**\n\nThen open a terminal.", STEP)
        self.assertEqual(fact, "Install Python 3")
        self.assertEqual(body, "Then open a terminal.")

    def test_trailing_with_code_phrase_is_dropped(self):
        fact, _ = split_fact(f"FACT-3: Use virtual environments with code {CODE}\nBody.", STEP)
        self.assertEqual(fact, "Use virtual environments")

    def test_missing_code_still_returns_fact_and_body(self):
        fact, body = split_fact("FACT-3: no code at all\nSecond line.", STEP)
        self.assertEqual((fact, body), ("no code at all", "Second line."))


class GuideTests(unittest.TestCase):
    def setUp(self):
        steps, _ = make_plan(MockLLM(), "Learn Python, practise daily", 4)
        self.steps = steps
        self.outputs = {s["id"]: f"FACT-{i}: Do part {i} ({make_code(i)}).\nExplain part {i} clearly."
                        for i, s in enumerate(steps[:-1], start=1)}

    def guide(self, **kw):
        args = dict(goal="Learn Python", steps=self.steps, outputs=self.outputs,
                    agent_label="memory agent", model="tiny", completed=True, stats_line="Run summary.")
        args.update(kw)
        return build_guide(**args)

    def test_has_one_section_per_work_step_and_no_codes(self):
        text = self.guide()
        self.assertIn("# How to: Learn Python", text)
        for n in (1, 2, 3):
            self.assertIn(f"## Step {n}:", text)
        self.assertNotIn("## Step 4", text)                # the final report step is not a how-to step
        self.assertNotIn("K1-", text)
        self.assertIn("Explain part 2 clearly.", text)
        self.assertIn("**Key point:** Do part 1", text)
        self.assertIn("## Summary of key points", text)

    def test_missing_step_is_marked(self):
        del self.outputs["step-2"]
        text = self.guide(completed=False)
        self.assertIn("*This step was not completed.*", text)
        self.assertIn("did not finish", text)

    def test_header_has_no_doubled_word(self):
        self.assertIn("Written by the memory agent (tiny)", self.guide())

    def test_slugify(self):
        self.assertEqual(slugify("How to learn Python!!"), "how-to-learn-python")
        self.assertEqual(slugify("???"), "task")
        self.assertLessEqual(len(slugify("a " * 100)), 50)

    def test_write_guide_creates_folders(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_guide(Path(d) / "a" / "b" / "g.txt", "hello")
            self.assertEqual(path.read_text(encoding="utf-8"), "hello")


class WindowBaselineTests(unittest.TestCase):
    def task(self, n):
        steps, _ = make_plan(MockLLM(), "a, b, c", n)
        return {"task_id": "t", "goal": "g", "success_criteria": [], "steps": steps}

    def test_window_keeps_prompt_small_but_forgets_early_facts(self):
        task = self.task(12)
        full = LLMBaselineAgent(MockLLM()).run(task)
        windowed = LLMBaselineAgent(MockLLM(), window=3).run(task)
        self.assertTrue(full["final_verified"])
        self.assertFalse(windowed["final_verified"])      # cannot list facts it no longer sees
        self.assertLess(max(windowed["prompt_series"]), max(full["prompt_series"]))

    def test_memory_agent_beats_the_window_on_recall(self):
        task = self.task(12)
        with tempfile.TemporaryDirectory() as d:
            mem = LLMMemoryAgent(MockLLM(), Path(d) / "m.json").run(task)
        self.assertTrue(mem["final_verified"])

    def test_window_zero_or_none_means_full_history(self):
        task = self.task(6)
        a = LLMBaselineAgent(MockLLM(), window=0).run(task)["prompt_series"]
        b = LLMBaselineAgent(MockLLM()).run(task)["prompt_series"]
        self.assertEqual(a, b)


class SolveCommandTests(unittest.TestCase):
    def run_cli(self, *argv):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "guide.md"
            buf = io.StringIO()
            with mock.patch.object(solve, "DATA", Path(d) / "data"), \
                    mock.patch.object(sys, "argv", ["solve.py", *argv, "--out", str(out)]), \
                    contextlib.redirect_stdout(buf):
                solve.main()
            guide = out.read_text(encoding="utf-8") if out.exists() else None
            data_files = sorted(p.name for p in (Path(d) / "data").glob("*"))
        return buf.getvalue(), guide, data_files

    def test_both_agents_run_prints_stats_and_writes_guide(self):
        text, guide, files = self.run_cli("Plan a launch", "--steps", "6")
        self.assertIn("STATISTICS", text)
        self.assertIn("Memory agent", text)
        self.assertIn("Steps redone after crash", text)
        self.assertIn("## Step 1:", guide)
        self.assertEqual(len(files), 1)                    # only the results JSON, no memory files left

    def test_only_memory_has_one_column_and_no_crash_by_default(self):
        text, guide, _ = self.run_cli("Plan a launch", "--steps", "5", "--only", "memory")
        self.assertIn("Crash after: never", text)
        self.assertNotIn("BASELINE AGENT", text)
        self.assertIn("completed", guide)

    def test_window_flag_is_announced_in_the_banner(self):
        text, _, _ = self.run_cli("Plan a launch", "--steps", "8", "--baseline-window", "2")
        self.assertIn("keeps only its last 2 outputs", text)


if __name__ == "__main__":
    unittest.main()