import tempfile
import unittest
from pathlib import Path

from agents.llm import MockLLM
from agents.llm_agents import LLMMemoryAgent, make_plan


class RecordingLLM(MockLLM):
    """MockLLM that keeps every prompt it receives."""

    def __init__(self):
        super().__init__()
        self.prompts = []

    def complete(self, prompt, **kw):
        self.prompts.append(prompt)
        return super().complete(prompt, **kw)


def run_memory(criteria, n=4):
    llm = RecordingLLM()
    steps, _ = make_plan(llm, "Compare options, then recommend", n)
    llm.prompts.clear()          # drop the planning prompt; keep only step prompts
    task = {"task_id": "t", "goal": "Compare options, then recommend",
            "success_criteria": [], "steps": steps}
    path = Path(tempfile.mkdtemp()) / "m.json"
    rep = LLMMemoryAgent(llm, path, criteria=criteria).run(task)
    return llm.prompts, steps, rep


class PromptParityTests(unittest.TestCase):
    def test_default_prompt_has_no_criteria(self):
        prompts, _, _ = run_memory(False)
        self.assertTrue(all("Acceptance criteria" not in p for p in prompts))

    def test_criteria_prompt_names_the_exact_code(self):
        prompts, steps, _ = run_memory(True)
        self.assertIn("Acceptance criteria", prompts[0])
        self.assertIn(steps[0]["code"], prompts[0])

    def test_final_step_criteria_list_every_code(self):
        prompts, steps, _ = run_memory(True)
        for step in steps[:-1]:
            self.assertIn(step["code"], prompts[-1])

    def test_criteria_do_not_change_what_gets_done(self):
        _, _, off = run_memory(False)
        _, _, on = run_memory(True)
        self.assertTrue(off["final_verified"] and on["final_verified"])
        self.assertEqual(off["llm_calls"], on["llm_calls"])


if __name__ == "__main__":
    unittest.main()