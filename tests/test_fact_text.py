import unittest

from agents.llm_agents import fact_text

STEP = {"id": "step-2", "code": "K2-4205"}


class FactTextTests(unittest.TestCase):

    def test_long_line_never_loses_the_code(self):
        text = "FACT-2: " + "x " * 150 + "K2-4205 " + "y " * 150
        out = fact_text(text, STEP)
        self.assertIn("K2-4205", out)
        self.assertLessEqual(len(out), 200)


if __name__ == "__main__":
    unittest.main()
