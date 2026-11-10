import tempfile
import unittest
from pathlib import Path

from agents.llm import MockLLM
from agents.llm_agents import make_plan
from agents.manager import Manager
from agents.mea import Auditor, MEAAgent
from agents.memory_manager import MemoryManager
from agents.taskstate import COMPLETED, PENDING, UNTRUSTED, BLOCKED, build_records
from agents.workflow import add_dependencies
from tests.test_replanner import FailsFor

GOAL = "Compare options, then write a recommendation"


def make_task(n):
    steps, _ = make_plan(MockLLM(), GOAL, n)
    return {"task_id": "t", "goal": GOAL, "success_criteria": [],
            "steps": add_dependencies(steps)}


def tmp():
    return Path(tempfile.mkdtemp()) / "m.json"


def fresh(n=6):
    mem = MemoryManager(tmp())
    state = mem.create_or_load(make_task(n))
    state.update(blocked_step_ids=[], replaced_step_ids=[], replans=[], answers={})
    return mem, state


def good(step):
    return {"success": True, "message": f"FACT-{step['id'][5:]}: ok {step['code']}",
            "output": f"FACT-{step['id'][5:]}: ok {step['code']}.\n" + "work " * 30,
            "timestamp": "t0"}


def run_audit(mem, state, step, result):
    c = Manager().decide(state)["contract"]
    audit = Auditor().audit(c, step, result, state)
    return mem.record_attempt(state, step, result, audit), audit


class RecordTests(unittest.TestCase):

    def test_suspect_claim_is_untrusted_and_not_a_fact(self):
        mem, state = fresh()
        step = state["plan"][0]
        bad = good(step)
        bad["output"] += f" also {state['plan'][4]['code']}"
        state, audit = run_audit(mem, state, step, bad)
        self.assertEqual(audit["integrity"], "suspect")
        r = build_records(state)
        self.assertEqual(r["requirements"][0]["status"], UNTRUSTED)
        self.assertEqual(r["facts"], [])                       # not trusted progress
        self.assertEqual(r["unverified_claims"][0]["status"], UNTRUSTED)
        self.assertEqual(state["findings"], [])                # and never fed to executors


class LoopRecordTests(unittest.TestCase):

    def test_blocked_run_reports_blocked_count(self):
        rep = MEAAgent(FailsFor({1, 7, 8, 9}), tmp()).run(make_task(6))
        self.assertGreaterEqual(rep["records_summary"][BLOCKED], 1)


if __name__ == "__main__":
    unittest.main()
