import unittest

from agents.llm import MockLLM
from agents.llm_agents import LLMBaselineAgent, make_plan


class RecordingLLM(MockLLM):
    def __init__(self):
        super().__init__()
        self.prompts = []

    def complete(self, prompt, **kw):
        self.prompts.append(prompt)
        return super().complete(prompt, **kw)


def run(criteria, n=4):
    llm = RecordingLLM()
    steps, _ = make_plan(llm, "Compare options, then recommend", n)
    llm.prompts.clear()          # drop the planning prompt
    task = {"task_id": "t", "goal": "g", "success_criteria": [], "steps": steps}
    rep = LLMBaselineAgent(llm, criteria=criteria).run(task)
    return llm.prompts, steps, rep


class BaselineCriteriaTests(unittest.TestCase):
    def test_default_prompt_has_no_criteria(self):
        prompts, _, _ = run(False)
        self.assertTrue(all("Acceptance criteria" not in p for p in prompts))

    def test_criteria_prompt_names_the_exact_code(self):
        prompts, steps, _ = run(True)
        self.assertIn("Acceptance criteria", prompts[0])
        self.assertIn(steps[0]["code"], prompts[0])

    def test_criteria_keep_the_full_log_in_the_prompt(self):
        prompts, steps, _ = run(True)
        self.assertIn(steps[0]["code"], prompts[-1])      # the final prompt still holds step 1's output

    def test_criteria_do_not_change_what_gets_done(self):
        _, _, off = run(False)
        _, _, on = run(True)
        self.assertTrue(off["final_verified"] and on["final_verified"])
        self.assertEqual(off["llm_calls"], on["llm_calls"])


if __name__ == "__main__":
    unittest.main()