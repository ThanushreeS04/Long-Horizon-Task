import tempfile
import unittest
from pathlib import Path

from agents.content_checks import (ContentAuditor, ContentManager, default_constraints,
                                   foreign_syntax, unknown_imports)
from agents.llm import MockLLM
from agents.llm_agents import make_plan
from agents.mea import MEAAgent
from agents.workflow import add_dependencies

GOAL = "Compare options, then write a recommendation"
JS = "FACT-5: K5-5054: if (isNaN(result)) throw new Error('Invalid input');"
LIB = "FACT-2: K2-4205 = 'from calculator_library import *' " + "x" * 80


class RuleTests(unittest.TestCase):
    def test_flags_javascript_in_a_python_task(self):
        self.assertTrue(foreign_syntax(JS, "python"))
        self.assertFalse(foreign_syntax("if x != x: raise ValueError('bad')", "python"))

    def test_flags_invented_module_but_not_stdlib(self):
        self.assertEqual(unknown_imports("from calculator_library import *"), ["calculator_library"])
        self.assertEqual(unknown_imports("import math\nfrom decimal import Decimal"), [])
        self.assertEqual(unknown_imports("import numpy", allowed=["numpy"]), [])
        self.assertEqual(unknown_imports(LIB), ["calculator_library"])      # the real step-2 output
        self.assertEqual(unknown_imports("This step lets us import functions from the module."), [])


class AuditorTests(unittest.TestCase):
    def _run(self, text, final=False):
        steps = add_dependencies(make_plan(MockLLM(), GOAL, 4)[0])
        step = steps[1] if not final else steps[-1]
        contract = {"step_id": step["id"], "goal": "g", "acceptance": [], "depends_on": [],
                    "evidence": {}, "attempt": 1, "last_rejection": None, "final": final}
        state = {"plan": steps, "pending_step_ids": [], "step_history": [], "errors": [], "findings": []}
        base = f"FACT-{step['id'].split('-')[1]} {step['code']} " + text + " " + "y" * 90
        return ContentAuditor().audit(contract, step, {"output": base}, state)

    def test_wrong_language_is_rejected_with_a_reason(self):
        res = self._run("throw new Error('x')")
        self.assertFalse(res["verified"])
        self.assertIn("non-Python", res["reason"])

    def test_invented_import_is_rejected(self):
        self.assertFalse(self._run("from calculator_library import *")["verified"])

    def test_clean_python_still_passes(self):
        self.assertTrue(self._run("import math\nresult = math.sqrt(4)")["verified"])


class IntegrationTests(unittest.TestCase):
    def test_constraints_reach_the_executor_prompt_and_mock_run_still_completes(self):
        steps = add_dependencies(make_plan(MockLLM(), GOAL, 6)[0])
        task = {"task_id": "t", "goal": GOAL, "success_criteria": [], "steps": steps}
        path = Path(tempfile.mkdtemp()) / "m.json"
        mgr = ContentManager(default_constraints("python"))
        d = mgr.decide(MEAAgent(MockLLM(), path).memory.create_or_load(task))
        self.assertIn("use Python only (no JavaScript or other languages)", d["contract"]["extra_acceptance"])
        agent = MEAAgent(MockLLM(), path, manager=mgr, auditor=ContentAuditor())
        self.assertTrue(agent.run(task)["completed"])


if __name__ == "__main__":
    unittest.main()