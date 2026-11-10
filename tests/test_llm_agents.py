import tempfile
import unittest
from pathlib import Path

from agents.llm import MockLLM
from agents.llm_agents import (MAX_ATTEMPTS, LLMBaselineAgent, LLMMemoryAgent, OutputAuditor,
                               make_code, make_plan, parse_steps)
from solve import merge, recall

GOAL = "Compare options, then write a recommendation"


def good(i):
    return f"FACT-{i}: Step {i} established {make_code(i)}.\n" + "Did the work in detail. " * 6


def make_task(n, llm=None):
    steps, _ = make_plan(llm or MockLLM(), GOAL, n)
    return {"task_id": "t", "goal": GOAL, "success_criteria": [], "steps": steps}


class ScriptedLLM(MockLLM):
    """MockLLM whose answer can be overridden per call; records every prompt."""

    def __init__(self, bad_for=None, bad_calls=None, crash_after_calls=None):
        super().__init__()
        self.prompts, self.calls = [], 0
        self.bad_for = bad_for or set()          # step numbers that are always answered badly
        self.bad_calls = bad_calls or set()      # call numbers (1-based) answered badly
        self.crash_after_calls = crash_after_calls

    def complete(self, prompt, kind="step", max_tokens=500, meta=None):
        self.calls += 1
        self.prompts.append(prompt)
        if self.crash_after_calls is not None and self.calls > self.crash_after_calls:
            raise RuntimeError("simulated crash")
        meta = meta or {}
        if kind == "step" and (meta.get("index") in self.bad_for or self.calls in self.bad_calls):
            return {"text": "I think this is fine.", "input_tokens": len(prompt) // 4,
                    "output_tokens": 5}
        return super().complete(prompt, kind=kind, max_tokens=max_tokens, meta=meta)


def step(i, final=False, n=4):
    return {"id": f"step-{i}", "code": make_code(i), "final": final,
            "required_codes": [make_code(j) for j in range(1, n)] if final else []}


class OutputAuditorTests(unittest.TestCase):
    def verify(self, s, text):
        return OutputAuditor().verify(s, {"output": text})

    def test_code_outside_first_300_chars_is_rejected(self):
        text = "FACT-1: " + "x" * 320 + f" {make_code(1)}"
        self.assertFalse(self.verify(step(1), text)["verified"])


class ParseStepsTests(unittest.TestCase):

    def test_bullet_list(self):
        self.assertEqual(parse_steps("- one\n* two", 5), ["one", "two"])


class MakePlanTests(unittest.TestCase):

    def test_short_model_answer_is_padded(self):
        class Short(MockLLM):
            def complete(self, prompt, kind="step", **kw):
                if kind == "plan":
                    return {"text": '["only one"]', "input_tokens": 1, "output_tokens": 1}
                return super().complete(prompt, kind=kind, **kw)
        steps, _ = make_plan(Short(), GOAL, 5)
        self.assertEqual(len(steps), 5)
        self.assertTrue(steps[1]["description"].startswith("Continue working on"))


class BaselineAgentTests(unittest.TestCase):

    def test_crash_then_restart_redoes_everything(self):
        task = make_task(6)
        agent = LLMBaselineAgent(MockLLM())
        r = merge([agent.run(task, stop_after=3), LLMBaselineAgent(MockLLM()).run(task)])
        self.assertTrue(r["final_verified"])
        self.assertEqual(r["repeated_steps"], 3)


class MemoryAgentTests(unittest.TestCase):
    def path(self):
        return Path(tempfile.mkdtemp()) / "m.json"

    def test_resume_repeats_nothing(self):
        p, task = self.path(), make_task(6)
        r = merge([LLMMemoryAgent(MockLLM(), p).run(task, stop_after=3),
                   LLMMemoryAgent(MockLLM(), p).run(task)])
        self.assertTrue(r["final_verified"])
        self.assertEqual((r["repeated_steps"], r["llm_calls"]), (0, 6))

    def test_failure_count_survives_a_restart(self):
        p, task = self.path(), make_task(4)
        crashing = ScriptedLLM(bad_for={1}, crash_after_calls=2)
        with self.assertRaises(RuntimeError):
            LLMMemoryAgent(crashing, p).run(task)           # 2 rejections saved, then crash
        again = ScriptedLLM(bad_for={1})
        LLMMemoryAgent(again, p).run(task)
        self.assertEqual(again.calls, MAX_ATTEMPTS - 2)      # only 1 attempt left

    def test_prompt_stays_bounded_while_baseline_grows(self):
        task = make_task(30)
        mem = LLMMemoryAgent(MockLLM(), self.path(), ctx_budget=4000).run(task)
        base = LLMBaselineAgent(MockLLM()).run(task)
        self.assertLess(max(mem["prompt_series"]), max(base["prompt_series"]) / 4)


class MergeAndRecallTests(unittest.TestCase):

    def test_recall_credits_replaced_steps(self):
        self.assertEqual(recall(make_code(7), 1, {"step-1": "step-7"}), (1, 1))

    def test_recall_of_missing_report_is_zero(self):
        self.assertEqual(recall(None, 5), (0, 5))


if __name__ == "__main__":
    unittest.main()
