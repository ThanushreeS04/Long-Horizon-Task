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

    def test_accepts_label_code_and_enough_work(self):
        self.assertTrue(self.verify(step(1), good(1))["verified"])

    def test_rejects_missing_label_and_names_it(self):
        text = f"Step 1 established {make_code(1)}. " + "Work. " * 20
        a = self.verify(step(1), text)
        self.assertFalse(a["verified"])
        self.assertIn("FACT-1", a["reason"])

    def test_rejects_missing_code_and_names_it(self):
        a = self.verify(step(2), "FACT-2: done.\n" + "Work. " * 20)
        self.assertFalse(a["verified"])
        self.assertIn(make_code(2), a["reason"])

    def test_rejects_short_output(self):
        a = self.verify(step(1), f"FACT-1: {make_code(1)}")
        self.assertFalse(a["verified"])
        self.assertIn("80 characters", a["reason"])

    def test_code_outside_first_300_chars_is_rejected(self):
        text = "FACT-1: " + "x" * 320 + f" {make_code(1)}"
        self.assertFalse(self.verify(step(1), text)["verified"])

    def test_label_of_a_longer_number_does_not_match(self):
        text = f"FACT-12: {make_code(1)} " + "Work. " * 20
        self.assertFalse(self.verify(step(1), text)["verified"])

    def test_final_accepts_when_all_codes_present(self):
        codes = " ".join(make_code(j) for j in (1, 2, 3))
        self.assertTrue(self.verify(step(4, True), f"FINAL REPORT {codes} listed above")["verified"])

    def test_final_rejects_missing_codes_and_counts_them(self):
        a = self.verify(step(4, True), "FINAL REPORT " + make_code(1) + " " + "x" * 40)
        self.assertFalse(a["verified"])
        self.assertIn("2 required", a["reason"])

    def test_final_rejects_when_too_short(self):
        codes = " ".join(make_code(j) for j in (1, 2, 3))
        self.assertFalse(self.verify(step(4, True), codes)["verified"])


class ParseStepsTests(unittest.TestCase):
    def test_json_array(self):
        self.assertEqual(parse_steps('["a", "b"]', 5), ["a", "b"])

    def test_fenced_json(self):
        self.assertEqual(parse_steps('```json\n["a", "b"]\n```', 5), ["a", "b"])

    def test_numbered_list(self):
        self.assertEqual(parse_steps("1. first\n2) second\n3. third", 5),
                         ["first", "second", "third"])

    def test_bullet_list(self):
        self.assertEqual(parse_steps("- one\n* two", 5), ["one", "two"])

    def test_truncates_to_count(self):
        self.assertEqual(parse_steps('["a","b","c","d"]', 2), ["a", "b"])


class MakePlanTests(unittest.TestCase):
    def test_plan_has_n_steps_with_a_single_final(self):
        steps, _ = make_plan(MockLLM(), GOAL, 6)
        self.assertEqual(len(steps), 6)
        self.assertEqual([s["final"] for s in steps], [False] * 5 + [True])

    def test_final_step_requires_every_earlier_code(self):
        steps, _ = make_plan(MockLLM(), GOAL, 5)
        self.assertEqual(steps[-1]["required_codes"], [s["code"] for s in steps[:-1]])

    def test_short_model_answer_is_padded(self):
        class Short(MockLLM):
            def complete(self, prompt, kind="step", **kw):
                if kind == "plan":
                    return {"text": '["only one"]', "input_tokens": 1, "output_tokens": 1}
                return super().complete(prompt, kind=kind, **kw)
        steps, _ = make_plan(Short(), GOAL, 5)
        self.assertEqual(len(steps), 5)
        self.assertTrue(steps[1]["description"].startswith("Continue working on"))

    def test_codes_are_unique(self):
        steps, _ = make_plan(MockLLM(), GOAL, 30)
        self.assertEqual(len({s["code"] for s in steps}), 30)


class BaselineAgentTests(unittest.TestCase):
    def test_completes_without_a_crash(self):
        rep = LLMBaselineAgent(MockLLM()).run(make_task(5))
        self.assertTrue(rep["final_verified"])

    def test_crash_then_restart_redoes_everything(self):
        task = make_task(6)
        agent = LLMBaselineAgent(MockLLM())
        r = merge([agent.run(task, stop_after=3), LLMBaselineAgent(MockLLM()).run(task)])
        self.assertTrue(r["final_verified"])
        self.assertEqual(r["repeated_steps"], 3)

    def test_retry_prompt_carries_the_rejection_reason(self):
        llm = ScriptedLLM(bad_calls={1})
        LLMBaselineAgent(llm).run(make_task(3))
        self.assertIn("previous attempt was rejected", llm.prompts[1])
        self.assertIn("FACT-1", llm.prompts[1])

    def test_gives_up_after_max_attempts(self):
        llm = ScriptedLLM(bad_for={1})
        rep = LLMBaselineAgent(llm).run(make_task(4))
        self.assertEqual(llm.calls, MAX_ATTEMPTS)
        self.assertFalse(rep["final_verified"])

    def test_window_forgets_old_outputs(self):
        llm = ScriptedLLM()
        task = make_task(8)
        LLMBaselineAgent(llm, window=2).run(task)
        self.assertNotIn(make_code(1), llm.prompts[6])      # step 7 prompt
        self.assertIn(make_code(6), llm.prompts[6])


class MemoryAgentTests(unittest.TestCase):
    def path(self):
        return Path(tempfile.mkdtemp()) / "m.json"

    def test_completes_and_saves_state(self):
        p = self.path()
        rep = LLMMemoryAgent(MockLLM(), p).run(make_task(5))
        self.assertTrue(rep["final_verified"])
        self.assertTrue(p.exists())

    def test_resume_repeats_nothing(self):
        p, task = self.path(), make_task(6)
        r = merge([LLMMemoryAgent(MockLLM(), p).run(task, stop_after=3),
                   LLMMemoryAgent(MockLLM(), p).run(task)])
        self.assertTrue(r["final_verified"])
        self.assertEqual((r["repeated_steps"], r["llm_calls"]), (0, 6))

    def test_two_crashes_still_finish(self):
        p, task = self.path(), make_task(8)
        sessions = [LLMMemoryAgent(MockLLM(), p).run(task, stop_after=2),
                    LLMMemoryAgent(MockLLM(), p).run(task, stop_after=2),
                    LLMMemoryAgent(MockLLM(), p).run(task)]
        r = merge(sessions)
        self.assertTrue(r["final_verified"])
        self.assertEqual(r["repeated_steps"], 0)

    def test_retry_prompt_contains_the_rejection_reason(self):
        llm = ScriptedLLM(bad_calls={1})
        LLMMemoryAgent(llm, self.path()).run(make_task(3))
        self.assertIn("Missing", llm.prompts[1])

    def test_gives_up_after_max_attempts(self):
        llm = ScriptedLLM(bad_for={1})
        rep = LLMMemoryAgent(llm, self.path()).run(make_task(4))
        self.assertEqual(llm.calls, MAX_ATTEMPTS)
        self.assertFalse(rep["final_verified"])

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

    def test_too_small_budget_fails_instead_of_inventing_facts(self):
        task = make_task(30)
        rep = LLMMemoryAgent(MockLLM(), self.path(), ctx_budget=300).run(task)
        self.assertFalse(rep["final_verified"])
        found, total = recall(rep["final_output"], 29)
        self.assertLess(found, total)


class MergeAndRecallTests(unittest.TestCase):
    def test_merge_sums_calls_and_tokens(self):
        a = {"llm_calls": 2, "input_tokens": 10, "output_tokens": 4, "prompt_series": [5, 5],
             "executed": [("s1", True)], "final_output": None, "final_verified": False,
             "elapsed_seconds": 1.0, "memory_seconds": 0.1}
        b = dict(a, final_output="x", final_verified=True, executed=[("s1", True), ("s2", True)])
        m = merge([a, b])
        self.assertEqual((m["llm_calls"], m["input_tokens"]), (4, 20))
        self.assertEqual((m["repeated_steps"], m["verified_steps"]), (1, 2))
        self.assertTrue(m["final_verified"])

    def test_recall_counts_codes_in_report(self):
        report = f"{make_code(1)} {make_code(3)}"
        self.assertEqual(recall(report, 3), (2, 3))

    def test_recall_credits_replaced_steps(self):
        self.assertEqual(recall(make_code(7), 1, {"step-1": "step-7"}), (1, 1))

    def test_recall_of_missing_report_is_zero(self):
        self.assertEqual(recall(None, 5), (0, 5))


if __name__ == "__main__":
    unittest.main()