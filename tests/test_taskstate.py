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
    def test_fresh_state_is_all_pending(self):
        mem, state = fresh()
        r = build_records(state)
        self.assertEqual(r["counts"], {COMPLETED: 0, PENDING: 6, BLOCKED: 0, UNTRUSTED: 0})
        self.assertEqual(r["facts"], [])

    def test_verified_step_is_completed_with_evidence(self):
        mem, state = fresh()
        state, a = run_audit(mem, state, state["plan"][0], good(state["plan"][0]))
        r = build_records(state)
        req = r["requirements"][0]
        self.assertEqual(req["status"], COMPLETED)
        self.assertEqual(req["evidence"]["source"], "step_history")
        self.assertEqual(len(r["facts"]), 1)

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

    def test_blocked_step_is_blocked(self):
        mem, state = fresh()
        state["blocked_step_ids"].append("step-1")
        self.assertEqual(build_records(state)["requirements"][0]["status"], BLOCKED)

    def test_completed_evidence_survives_compaction(self):
        mem, state = fresh(12)
        for i in range(11):
            state, _ = run_audit(mem, state, state["plan"][i], good(state["plan"][i]))
        r = build_records(state)
        done = [q for q in r["requirements"] if q["status"] == COMPLETED]
        self.assertEqual(len(done), 11)
        self.assertTrue(all(q["evidence"] for q in done))
        self.assertTrue(any(q["evidence"]["source"] == "findings" for q in done))

    def test_replaced_step_is_hidden_and_alternative_points_back(self):
        rep_path = tmp()
        MEAAgent(FailsFor({2}), rep_path).run(make_task(6))
        saved = MemoryManager(rep_path).get_task("t")["records"]
        ids = [q["id"] for q in saved["requirements"]]
        self.assertNotIn("step-2", ids)
        alt = next(q for q in saved["requirements"] if q["id"] == "step-7")
        self.assertEqual(alt["replaces"], "step-2")
        self.assertEqual(saved["counts"][COMPLETED], 6)


class LoopRecordTests(unittest.TestCase):
    def test_run_saves_records_and_artifacts(self):
        path = tmp()
        rep = MEAAgent(MockLLM(), path).run(make_task(6))
        saved = MemoryManager(path).get_task("t")
        self.assertEqual(rep["records_summary"][COMPLETED], 6)
        self.assertEqual(len(saved["artifacts"]), 6)
        final = saved["artifacts"][-1]
        self.assertIn("FINAL REPORT", final["text"])
        self.assertEqual(len(final["sha"]), 12)

    def test_artifacts_survive_a_crash(self):
        path, task = tmp(), make_task(6)
        MEAAgent(MockLLM(), path).run(task, stop_after=3)
        self.assertEqual(len(MemoryManager(path).get_task("t")["artifacts"]), 3)
        MEAAgent(MockLLM(), path).run(task)
        self.assertEqual(len(MemoryManager(path).get_task("t")["artifacts"]), 6)

    def test_blocked_run_reports_blocked_count(self):
        rep = MEAAgent(FailsFor({1, 7, 8, 9}), tmp()).run(make_task(6))
        self.assertGreaterEqual(rep["records_summary"][BLOCKED], 1)


if __name__ == "__main__":
    unittest.main()