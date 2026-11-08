import unittest

from agents.llm_agents import fact_text

STEP = {"id": "step-2", "code": "K2-4205"}


class FactTextTests(unittest.TestCase):
    def test_code_at_start_keeps_the_rest_of_the_line(self):
        out = fact_text("FACT-2: K2-4205 = result = num1 + num2\n\nMore detail below.", STEP)
        self.assertEqual(out, "FACT-2: K2-4205 = result = num1 + num2")

    def test_code_on_a_later_line_is_still_kept(self):
        out = fact_text("FACT-2: adds numbers\nthe code is K2-4205 here\nignored", STEP)
        self.assertIn("adds numbers", out)
        self.assertIn("K2-4205", out)
        self.assertNotIn("ignored", out)

    def test_long_line_never_loses_the_code(self):
        text = "FACT-2: " + "x " * 150 + "K2-4205 " + "y " * 150
        out = fact_text(text, STEP)
        self.assertIn("K2-4205", out)
        self.assertLessEqual(len(out), 200)

    def test_no_code_falls_back_to_first_line(self):
        out = fact_text("FACT-2: nothing useful\nsecond line", STEP | {"code": "K9-0000"})
        self.assertEqual(out, "FACT-2: nothing useful")


if __name__ == "__main__":
    unittest.main()