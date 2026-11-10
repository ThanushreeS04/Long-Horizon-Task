import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.llm import MockLLM
from agents.llm_agents import LLMMemoryAgent
from agents.timeouts import LLMTimeout
from tests.test_llm_agents import make_task


class TimeoutOnceLLM:
    """Raises LLMTimeout on the first step call only."""
    name = "timeout-once"

    def __init__(self):
        self.calls, self.inner = 0, MockLLM()

    def complete(self, prompt, kind="step", max_tokens=500, meta=None):
        self.calls += 1
        if self.calls == 1:
            raise LLMTimeout("simulated slow model")
        return self.inner.complete(prompt, kind=kind, max_tokens=max_tokens, meta=meta)


class TimeoutAgentTests(unittest.TestCase):
    def test_timed_out_step_is_recorded_then_retried_and_task_completes(self):
        p = Path(tempfile.mkdtemp()) / "m.json"
        task = make_task(4)
        rep = LLMMemoryAgent(TimeoutOnceLLM(), p).run(task)
        self.assertTrue(rep["final_verified"])
        # a fresh agent (as after a restart) loads the saved state
        state = LLMMemoryAgent(MockLLM(), p).memory.create_or_load(task)
        self.assertEqual(len(state["errors"]), 1)     # the timeout was saved as one failure
        self.assertEqual(state["pending_step_ids"], [])


if __name__ == "__main__":
    unittest.main()