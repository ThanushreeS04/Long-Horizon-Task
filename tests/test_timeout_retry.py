import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.llm import MockLLM
from agents.llm_agents import _Stats, _execute
from agents.timeouts import LLMTimeout

STEP = {"id": "STEP-1", "code": "ALPHA-1", "description": "do x"}


class FlakyLLM:
    """Times out on the first call, then behaves like the mock."""
    name = "flaky"

    def __init__(self):
        self.calls = 0
        self.inner = MockLLM()

    def complete(self, prompt, kind="step", max_tokens=500, meta=None):
        self.calls += 1
        if self.calls == 1:
            raise LLMTimeout("simulated slow model")
        return self.inner.complete(prompt, kind=kind, max_tokens=max_tokens, meta=meta)


class TimeoutRetryTests(unittest.TestCase):

    def test_retry_after_timeout_succeeds(self):
        llm, stats = FlakyLLM(), _Stats()
        first = _execute(llm, "prompt text", STEP, stats)
        second = _execute(llm, "prompt text", STEP, stats)
        self.assertFalse(first["success"])
        self.assertTrue(second["success"])
        self.assertNotIn("timeout", second)
        self.assertEqual(stats.calls, 2)


if __name__ == "__main__":
    unittest.main()
