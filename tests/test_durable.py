import tempfile
import unittest
from pathlib import Path

from agents.durable import (CheckpointStore, DurableMemoryManager, Journal,
                            make_durable, reconcile, validate_state)
from agents.llm import MockLLM
from agents.llm_agents import make_plan
from agents.mea import Executor, MEAAgent
from agents.workflow import add_dependencies
from agents.workspace import Workspace

GOAL = "Compare options, then write a recommendation"


def make_task(n):
    steps, _ = make_plan(MockLLM(), GOAL, n)
    return {"task_id": "t", "goal": GOAL, "success_criteria": [], "steps": add_dependencies(steps)}


def tmpdir():
    return Path(tempfile.mkdtemp())


class JournalTests(unittest.TestCase):
    def test_torn_and_damaged_lines_are_skipped(self):
        j = Journal(tmpdir() / "j.jsonl")
        j.append("begin", round=1, step="step-1")
        j.append("commit", round=1, step="step-1", verified=True)
        with open(j.path, "a") as f:
            f.write('{"crc": 1, "body": {"seq": 9, "event": "evil"}}\n')   # bad crc
            f.write('{"crc": 123, "bo')                                     # torn tail
        events = [e["event"] for e in Journal(j.path).read()]
        self.assertEqual(events, ["begin", "commit"])
        self.assertIsNone(Journal(j.path).in_flight())

    def test_in_flight_round_is_found(self):
        j = Journal(tmpdir() / "j.jsonl")
        j.append("begin", round=1, step="step-1")
        j.append("commit", round=1, step="step-1", verified=True)
        j.append("begin", round=2, step="step-2")
        self.assertEqual(j.in_flight()["step"], "step-2")


class CheckpointTests(unittest.TestCase):
    def test_corrupt_latest_falls_back_and_rotation_keeps_n(self):
        store = CheckpointStore(tmpdir(), keep=3)
        for i in range(5):
            store.save({"task_id": "t", "i": i}, f"r{i}")
        self.assertEqual(len(store.list()), 3)
        newest = sorted(store.dir.glob("ckpt-*.json"))[-1]
        newest.write_text(newest.read_text().replace('"i": 4', '"i": 99'))
        self.assertEqual(store.latest_valid()["state"]["i"], 3)
        with self.assertRaises(ValueError):
            store.restore(5)


class StoreRecoveryTests(unittest.TestCase):
    def test_corrupt_memory_file_is_rebuilt_from_checkpoint(self):
        path = tmpdir() / "m.json"
        make_durable(MEAAgent(MockLLM(), path)).run(make_task(6), stop_after=3)
        path.write_text("{ this is not json")
        rep = make_durable(MEAAgent(MockLLM(), path)).run(make_task(6))
        self.assertTrue(rep["completed"])
        self.assertTrue(list(path.parent.glob("m.json.corrupt-*")))

    def test_no_checkpoint_and_no_backup_still_raises(self):
        path = tmpdir() / "m.json"
        path.write_text("garbage")
        with self.assertRaises(ValueError):
            DurableMemoryManager(path).get_task("t")


class CrashTests(unittest.TestCase):
    def test_crash_mid_round_resumes_without_repeating_verified_steps(self):
        path, task = tmpdir() / "m.json", make_task(8)

        class Dies(Executor):
            calls = 0

            def run(self, goal, contract, step, stats):
                Dies.calls += 1
                if Dies.calls == 4:
                    raise RuntimeError("process killed")
                return super().run(goal, contract, step, stats)

        first = make_durable(MEAAgent(MockLLM(), path, executor=Dies(MockLLM())))
        with self.assertRaises(RuntimeError):
            first.run(task)
        mem = DurableMemoryManager(path)
        self.assertIsNotNone(mem.journal.in_flight("t"))
        before = set(mem.get_task("t")["pending_step_ids"])
        second = make_durable(MEAAgent(MockLLM(), path))
        rep = second.run(task)
        self.assertTrue(rep["completed"])
        self.assertEqual(second.memory.last_report["actions"][0]["action"], "abandoned_round")
        ran = [s for s, _ in rep["executed"]]
        self.assertEqual(len(ran), len(before))                # only unfinished steps ran
        self.assertEqual(len(ran), len(set(ran)))
        self.assertIsNone(mem.journal.in_flight("t"))
        self.assertEqual(len(mem.get_task("t")["errors"]), 0)  # a crash is not a failed attempt

    def test_replan_creates_a_checkpoint_that_can_be_rolled_back(self):
        path, task = tmpdir() / "m.json", make_task(6)
        mem = DurableMemoryManager(path)
        state = mem.create_or_load(task)
        mem.save_task(state)
        old_plan = [s["description"] for s in state["plan"]]
        state["plan"][2]["description"] = "rewritten by a bad replan"
        mem.save_task(state)
        seqs = [s for s, reason, _, ok in mem.checkpoints("t").list() if reason == "replan"]
        self.assertEqual(len(seqs), 1)
        self.assertEqual(mem.checkpoints("t").list()[0][1], "created")
        restored = mem.rollback("t", 1)
        self.assertEqual([s["description"] for s in restored["plan"]], old_plan)
        self.assertEqual([s["description"] for s in mem.get_task("t")["plan"]], old_plan)



class ReconcileTests(unittest.TestCase):
    def _state(self):
        steps = add_dependencies(make_plan(MockLLM(), GOAL, 5)[0], width=1)
        return {"task_id": "t", "goal": GOAL, "plan": steps, "pending_step_ids": [],
                "step_history": [], "errors": [], "findings": [f"FACT-{i}: x" for i in range(1, 5)],
                "status": "completed", "artifacts": []}

    def test_tampered_artifact_demotes_step_and_its_dependents(self):
        ws, j = Workspace(tmpdir()), Journal(tmpdir() / "j.jsonl")
        state = self._state()
        for s in state["plan"]:
            rec = ws.write_step(s["id"], f"content of {s['id']}")
            j.append("commit", task_id="t", round=1, step=s["id"], verified=True,
                     artifact={"path": rec["path"], "sha256": rec["sha256"]})
        (ws.root / "step-2.md").write_text("tampered")
        rep = reconcile(state, j, ws)
        demoted = {a["step"] for a in rep["actions"] if a["action"] == "demoted"}
        self.assertEqual(demoted, {"step-2", "step-3", "step-4", "step-5"})
        self.assertEqual(state["status"], "in_progress")
        self.assertEqual(state["pending_step_ids"], ["step-2", "step-3", "step-4", "step-5"])
        self.assertEqual(state["findings"], ["FACT-1: x"])

    def test_untouched_artifacts_change_nothing(self):
        ws, j = Workspace(tmpdir()), Journal(tmpdir() / "j.jsonl")
        state = self._state()
        rec = ws.write_step("step-1", "ok")
        j.append("commit", task_id="t", round=1, step="step-1", verified=True,
                 artifact={"path": rec["path"], "sha256": rec["sha256"]})
        self.assertEqual(reconcile(state, j, ws)["actions"], [])

    def test_validate_state_flags_broken_structure(self):
        state = self._state()
        state["pending_step_ids"] = ["step-99"]
        state["status"] = "completed"
        problems = validate_state(state)
        self.assertTrue(any("unknown steps" in p for p in problems))
        self.assertTrue(any("still pending" in p for p in problems))
        self.assertTrue(validate_state({"task_id": "t"}))


class MetricsTests(unittest.TestCase):
    def test_probe_reports_sizes_recall_and_ram(self):
        from agents.memory_manager import MemoryManager
        from agents.memory_metrics import MemoryProbe, PeakRAM
        mem = MemoryManager(tmpdir() / "m.json")
        probe = MemoryProbe(mem, budget_chars=1500)
        state = mem.create_or_load(make_task(6))
        state["findings"] = [f"FACT-{i}: " + "x" * 300 for i in range(1, 7)]
        with PeakRAM() as ram:
            probe.save(state)
            probe.load("t")
            ctx = probe.context(state)
        rep = probe.report()
        self.assertEqual(rep["context_over_budget"], 0)
        self.assertGreater(rep["store_chars_last"], rep["context_chars_last"])
        self.assertGreater(ram.peak_kib, 0)
        self.assertEqual(probe.fact_recall(ctx, ["step-6"]), 1.0)


if __name__ == "__main__":
    unittest.main()