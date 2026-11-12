import builtins
import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agentic_run
from agents.cleanup import RunFiles
from agents.llm import MockLLM
from agents.natural import NaturalRules, natural_fact


class Chat(MockLLM):
    """A model that just answers in plain prose: no FACT labels, no codes."""

    def __init__(self, bad_first=False):
        self.prompts, self.step_calls, self.bad_first = [], 0, bad_first

    def complete(self, prompt, kind="step", max_tokens=500, meta=None):
        meta = meta or {}
        if kind == "step":
            self.prompts.append(prompt)
            self.step_calls += 1
            if self.bad_first and self.step_calls == 1:
                text = "[insert the text here]"
            else:
                i = meta["index"]
                text = (f"Here is the finished work for step {i}. The key finding is zebra{i}, which later "
                        "steps should build on. " + "It is specific and concrete. " * 5)
        elif kind == "final":
            self.prompts.append(prompt)
            found = sorted(set(re.findall(r"zebra\d+", prompt)))
            text = "The overall answer builds on " + ", ".join(found) + ". " + "It combines every earlier result. " * 5
        else:
            return super().complete(prompt, kind, max_tokens, meta)
        return {"text": text, "input_tokens": len(prompt) // 4, "output_tokens": len(text) // 4}


def run_cli(d, argv, llm):
    runs, out = Path(d) / "runs", io.StringIO()
    with mock.patch.object(agentic_run, "RUNS", runs), mock.patch.object(agentic_run, "ROOT", Path(d)), \
            mock.patch.object(agentic_run, "RunFiles", lambda: RunFiles(runs)), \
            mock.patch.object(agentic_run, "make_llm", lambda name, model=None: llm), \
            contextlib.redirect_stdout(out):
        code = agentic_run.main(argv)
    return code, out.getvalue(), runs


class NaturalRulesTests(unittest.TestCase):
    step, final = {"id": "step-3"}, {"id": "step-9", "final": True}

    def test_verdicts(self):
        r = NaturalRules()
        good = "A concrete, specific piece of finished work. " * 4
        self.assertTrue(r.verify(self.step, {"output": good})["verified"])
        self.assertFalse(r.verify(self.step, {"output": "too short"})["verified"])
        self.assertFalse(r.verify(self.step, {"output": "[insert the budget here] " + good})["verified"])
        self.assertFalse(r.verify(self.step, {"output": "I'm sorry, but I can't. " + good})["verified"])
        self.assertFalse(r.verify(self.final, {"output": "x" * 120})["verified"])      # final needs 150

    def test_fact_label_is_added_by_the_harness_and_cut_at_a_word(self):
        fact = natural_fact("First paragraph " + "word " * 80 + "\n\nSecond paragraph.", self.step)
        self.assertTrue(fact.startswith("FACT-3: First paragraph"))
        self.assertTrue(fact.endswith("..."))
        self.assertNotIn("Second paragraph", fact)
        self.assertLessEqual(len(fact), 240)


class NaturalPipeline(unittest.TestCase):
    def test_model_is_never_asked_for_labels_or_codes(self):
        llm = Chat()
        with tempfile.TemporaryDirectory() as d:
            code, _, _ = run_cli(d, ["Explain zebras", "--steps", "5"], llm)
        self.assertEqual(code, 0)
        for p in llm.prompts:
            self.assertNotIn("FACT-", p)
            self.assertNotIn("exact code", p)
            self.assertIsNone(re.search(r"K\d+-[0-9a-f]{4}", p))

    def test_each_step_and_the_final_see_the_full_earlier_work(self):
        llm = Chat()
        with tempfile.TemporaryDirectory() as d:
            code, out, _ = run_cli(d, ["Explain zebras", "--steps", "5", "--quiet"], llm)
        self.assertEqual(code, 0)
        final_prompt = llm.prompts[-1]
        for i in (1, 2, 3, 4):
            self.assertIn(f"zebra{i}", final_prompt)             # full text, not a 200-char stub
        self.assertIn("zebra1", out)                              # and it reached the answer

    def test_quiet_prints_only_the_answer(self):
        with tempfile.TemporaryDirectory() as d:
            code, out, _ = run_cli(d, ["Explain zebras", "--steps", "4", "--quiet"], Chat())
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("The overall answer builds on"))
        for noise in ("RESULT", "Cleaned up", "Workspace", "Role"):
            self.assertNotIn(noise, out)

    def test_placeholder_reply_is_rejected_then_retried(self):
        with tempfile.TemporaryDirectory() as d:
            code, _, runs = run_cli(d, ["Explain zebras", "--steps", "3"], Chat(bad_first=True))
            report = json.loads(next(runs.glob("*_report.json")).read_text())
        self.assertEqual(code, 0)
        self.assertEqual(report["rounds"], 4)                     # 3 steps + 1 redo

    def test_default_never_waits_for_a_person(self):
        def no_input(*a, **k):
            raise AssertionError("the agent asked a human for input")
        with tempfile.TemporaryDirectory() as d, mock.patch.object(builtins, "input", no_input):
            code, _, runs = run_cli(d, ["Explain zebras", "--steps", "3", "--ask-at", "2"], Chat())
            state = json.loads(next(runs.glob("*_state.json")).read_text())
        self.assertEqual(code, 0)
        self.assertIn("step-2", next(iter(state.values()))["answers"])   # it decided alone

    def test_legacy_flag_restores_the_label_and_code_checks(self):
        with tempfile.TemporaryDirectory() as d:
            code, _, _ = run_cli(d, ["Explain zebras", "--steps", "3", "--legacy-codes"], Chat())
        self.assertNotEqual(code, 0)                              # plain prose has no FACT line or code


    def test_small_preset_uses_about_one_model_call_per_step(self):
        counts = {}
        for label, extra in (("full", []), ("small", ["--small"])):
            llm = Chat()
            calls = []
            real = llm.complete
            llm.complete = lambda p, kind="step", max_tokens=500, meta=None, _r=real, _c=calls: (
                _c.append(kind), _r(p, kind, max_tokens, meta))[1]
            with tempfile.TemporaryDirectory() as d:
                code, _, _ = run_cli(d, ["Explain zebras", "--steps", "5", *extra], llm)
            self.assertEqual(code, 0)
            counts[label] = calls
        self.assertLess(len(counts["small"]), len(counts["full"]) / 2)
        self.assertEqual(set(counts["small"]), {"plan", "step", "final"})   # no manager/audit/replan calls
        self.assertEqual(len(counts["small"]), 1 + 5)                       # planner + 4 steps + final


if __name__ == "__main__":
    unittest.main()
