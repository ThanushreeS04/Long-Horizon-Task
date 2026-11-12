import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agentic_run
import synthetic_pipeline
from agents.adapter import AgentAdapter, ModelRouter
from agents.cleanup import RunFiles
from agents.cot_executor import CoTExecutor
from agents.env_auditor import EnvironmentAuditor
from agents.hl_planner import HighLevelPlanner
from agents.llm import MockLLM
from agents.llm_agents import make_plan
from agents.mea import MEAAgent
from agents.trajectory import export_trajectory
from agents.workspace import Workspace


class Tamper(CoTExecutor):
    """Misbehaving executor: acts on the workspace ON ITS FIRST ATTEMPT of one step."""

    def __init__(self, *a, step_id, action, **kw):
        super().__init__(*a, **kw)
        self.step_id, self.action, self.done = step_id, action, False

    def run(self, goal, contract, step, stats):
        result = super().run(goal, contract, step, stats)
        if step["id"] == self.step_id and not self.done:
            self.done = True
            self.action(self.workspace, step)
        return result


class Scripted(MockLLM):
    def __init__(self, **by_kind):
        self.by_kind, self.calls = by_kind, {}

    def complete(self, prompt, kind="step", max_tokens=500, meta=None):
        self.calls[kind] = self.calls.get(kind, 0) + 1
        if kind in self.by_kind:
            return {"text": self.by_kind[kind](self.calls[kind]), "input_tokens": 5, "output_tokens": 5}
        return super().complete(prompt, kind, max_tokens, meta)


def run_ws(tmp, executor_cls=CoTExecutor, llm=None, **ekw):
    llm = llm or MockLLM()
    adapter = AgentAdapter(ModelRouter.single(llm))
    steps, _ = make_plan(llm, "Plan a launch, budget, announce", 5)
    ws = Workspace(Path(tmp) / "ws")
    agent = MEAAgent(llm, Path(tmp) / "s.json", adapter=adapter,
                     auditor=EnvironmentAuditor(adapter, ws),
                     executor=executor_cls(adapter.for_role("executor"), workspace=ws, **ekw))
    rep = agent.run({"task_id": "t", "goal": "Plan a launch", "success_criteria": [], "steps": steps})
    return rep, ws


class WorkspaceAudit(unittest.TestCase):
    def test_clean_run_writes_and_verifies_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            rep, ws = run_ws(d)
            self.assertTrue(rep["completed"])
            self.assertEqual(rep["integrity_events"], [])
            self.assertEqual(len(ws.snapshot()), 5)

    def test_artifact_changed_after_receipt_is_a_violation_then_recovers(self):
        tamper = lambda ws, step: ws.path_for(step["id"]).write_text("tampered", encoding="utf-8")
        with tempfile.TemporaryDirectory() as d:
            rep, _ = run_ws(d, Tamper, step_id="step-2", action=tamper)
            self.assertIn(("step-2", "violation"), rep["integrity_events"])
            self.assertTrue(rep["completed"])               # the retry was clean

    def test_stray_file_is_suspect(self):
        stray = lambda ws, step: (ws.root / "stray.txt").write_text("x")
        with tempfile.TemporaryDirectory() as d:
            rep, _ = run_ws(d, Tamper, step_id="step-2", action=stray)
            self.assertIn(("step-2", "suspect"), rep["integrity_events"])

    def test_empty_executor_reply_means_no_artifact_and_a_retry(self):
        llm = Scripted(step=lambda n: "" if n == 1 else MockLLM().complete("", "step", 5, {"index": 1, "code": "K1-8592"})["text"])
        with tempfile.TemporaryDirectory() as d:
            rep, _ = run_ws(d, llm=llm)
            self.assertGreater(rep["rounds"], 4)            # step 1 had to be redone

    def test_workspace_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as d:
            ws = Workspace(Path(d) / "ws")
            self.assertEqual(ws.path_for("../../etc/passwd").parent, ws.root)
            with self.assertRaises(ValueError):
                ws.read_artifact("../outside.md")


class PauseResume(unittest.TestCase):
    def run_cli(self, d, argv):
        runs = Path(d) / "runs"
        with mock.patch.object(agentic_run, "RUNS", runs), mock.patch.object(agentic_run, "ROOT", Path(d)), \
                mock.patch.object(agentic_run, "RunFiles", lambda: RunFiles(runs)), \
                contextlib.redirect_stdout(io.StringIO()):
            return agentic_run.main(argv), runs

    def test_pause_survives_next_run_cleanup_and_resumes(self):
        with tempfile.TemporaryDirectory() as d:
            code, runs = self.run_cli(d, ["Plan a budget", "--steps", "4", "--ask", "off", "--ask-at", "2"])
            self.assertEqual(code, 3)
            state = next(runs.glob("*_state.json"))
            self.assertTrue(state.exists())                  # paused: kept, not tracked for deletion
            code, _ = self.run_cli(d, ["--resume", str(state), "--answer", "USD 5,000"])
            self.assertEqual(code, 0)
            self.assertIn("USD 5,000", json.loads(state.read_text())[next(iter(json.loads(state.read_text())))]["answers"]["step-2"])
            code, _ = self.run_cli(d, ["Another task", "--steps", "3"])      # starts by cleaning the old run
            self.assertEqual(code, 0)
            self.assertFalse(state.exists())

    def test_resume_without_a_pending_question_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            _, runs = self.run_cli(d, ["Plan a thing", "--steps", "3"])
            state = next(runs.glob("*_state.json"))
            with self.assertRaises(SystemExit):
                self.run_cli(d, ["--resume", str(state), "--answer", "x"])


class SyntheticAndTrajectory(unittest.TestCase):
    def test_ungrounded_steps_are_rejected(self):
        ok = synthetic_pipeline.valid_steps([{"title": "a", "action_indices": [0, 1]}], 2)
        bad = synthetic_pipeline.valid_steps([{"title": "a", "action_indices": [5]}], 2)
        self.assertEqual(len(ok), 1)
        self.assertEqual(bad, [])

    def test_pipeline_on_example_file(self):
        with tempfile.TemporaryDirectory() as d:
            argv = ["x", "--input", str(ROOT / "examples" / "trajectories.example.jsonl"), "--output", d,
                    "--llm", "mock", "--expand", "2", "--targeted-augment", "1"]
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                synthetic_pipeline.main()
            report = json.loads((Path(d) / "pipeline_report.json").read_text())
            self.assertEqual(report["accepted_trajectories"], 1)
            self.assertEqual(report["executor_examples"], 3)
            rows = [json.loads(l) for l in (Path(d) / "planner_train.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), report["planner_examples"])
            self.assertTrue(all(r["messages"][-1]["role"] == "assistant" for r in rows))

    def test_trajectory_marks_retried_steps_as_failure_prone(self):
        plan = [{"id": "step-1", "description": "a"}, {"id": "step-2", "description": "b"},
                {"id": "step-3", "description": "report", "final": True}]
        outs = {"step-1": "FACT-1: did a", "step-2": "FACT-2: did b", "step-3": "report"}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.jsonl"
            n = export_trajectory("goal", plan, outs, [{"step_id": "step-2"}], True, p)
            row = json.loads(p.read_text())
            self.assertEqual((n, row["failure_prone_indices"], row["success"]), (2, [1], True))
            self.assertEqual(len(row["trajectory"]), 2)       # final report is not an action

    def test_trained_planner_prompt_accepts_the_training_format(self):
        trained = lambda n: json.dumps({"steps": [{"title": "Draft outline", "action_indices": [0]},
                                                  {"title": "Add milestones", "action_indices": [1]}]})
        adapter = AgentAdapter(ModelRouter.single(Scripted(plan=trained)))
        p = HighLevelPlanner(adapter, cot=False, trained_prompt=True).plan("Make a plan", 6)
        self.assertEqual(p["source"], "model")
        self.assertEqual([s["description"] for s in p["steps"]][:2], ["Draft outline", "Add milestones"])
        self.assertEqual(len(p["steps"]), 3)                  # 2 trained steps + final report


if __name__ == "__main__":
    unittest.main()
