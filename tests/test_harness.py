import contextlib
import io
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agentic_run
from agents.adapter import AgentAdapter, ModelRouter
from agents.cleanup import RunFiles
from agents.cot_executor import CoTExecutor
from agents.env_auditor import EnvironmentAuditor
from agents.harness import ExternalCLIHarness, ToolBox, ToolLoopHarness, parse_action
from agents.llm import MockLLM
from agents.llm_agents import _Stats, make_plan
from agents.mea import MEAAgent
from agents.natural import NaturalRules
from agents.workspace import Workspace

PROSE = "This is the finished work, written out concretely and specifically for the reader. " * 3
STEP = {"id": "step-1", "description": "d", "code": "K1-aaaa", "final": False}


class Seq(MockLLM):
    """Replies come from fn(prompt, kind, n) where n counts calls of that kind; None = default mock."""

    def __init__(self, fn):
        self.fn, self.n, self.prompts = fn, {}, []

    def complete(self, prompt, kind="step", max_tokens=500, meta=None):
        self.n[kind] = self.n.get(kind, 0) + 1
        self.prompts.append((kind, prompt))
        text = self.fn(prompt, kind, self.n[kind])
        if text is None:
            return super().complete(prompt, kind, max_tokens, meta)
        return {"text": text, "input_tokens": 5, "output_tokens": 5}


def pipeline(tmp, llm, harness_factory, executor_cls=CoTExecutor):
    adapter = AgentAdapter(ModelRouter.single(llm))
    steps, _ = make_plan(llm, "Write a small calculator", 3)
    ws = Workspace(Path(tmp) / "ws")
    auditor = EnvironmentAuditor(adapter, ws, use_model=False)
    auditor.rules = NaturalRules()
    ex = executor_cls(adapter.for_role("executor"), cot=False, workspace=ws, natural=True)
    ex.harness = harness_factory(ex.llm, ws)
    agent = MEAAgent(llm, Path(tmp) / "s.json", adapter=adapter, auditor=auditor, executor=ex,
                     manager=agentic_run.Manager(natural=True))
    rep = agent.run({"task_id": "t", "goal": "Write a small calculator", "success_criteria": [], "steps": steps})
    return rep, ws, agent.memory.get_task("t")


WRITE_RUN_FINISH = {
    1: "ACTION: write_file\nPATH: calc.py\nCONTENT:\nprint(2 + 3)\nEND",
    2: "ACTION: run_python\nPATH: calc.py",
    3: "I wrote calc.py and ran it, and it printed 5. " + PROSE}


class Parsing(unittest.TestCase):
    def test_actions_and_plain_answers(self):
        self.assertEqual(parse_action(WRITE_RUN_FINISH[1]), ("write_file", "calc.py", "print(2 + 3)"))
        self.assertEqual(parse_action("ACTION: read_file\nPATH: `a.md`"), ("read_file", "a.md", None))
        self.assertEqual(parse_action("ACTION: finish\nANSWER: done well")[2], "done well")
        self.assertIsNone(parse_action(PROSE))
        self.assertIsNone(parse_action("Plan:\nACTION: Call the supplier\nMore text"))   # not a tool


class ToolBoxSafety(unittest.TestCase):
    def test_paths_reserved_names_and_run_switch(self):
        with tempfile.TemporaryDirectory() as d:
            ws = Workspace(Path(d) / "ws")
            box = ToolBox(ws)
            self.assertTrue(box.call("write_file", "../evil.py", "x").startswith("ERROR"))
            self.assertTrue(box.call("write_file", "step-1.md", "x").startswith("ERROR"))
            self.assertTrue(box.call("read_file", "../../etc/passwd", None).startswith("ERROR"))
            self.assertTrue(box.call("write_file", "a.txt", None).startswith("ERROR"))
            self.assertTrue(box.call("write_file", "sub/a.txt", "hello").startswith("OK"))
            self.assertEqual(list(box.written), ["sub/a.txt"])
            self.assertIn("switched off", box.call("run_python", "sub/a.py", None))
            self.assertFalse((Path(d) / "evil.py").exists())

    def test_run_python_times_out_and_produced_files_are_declared(self):
        with tempfile.TemporaryDirectory() as d:
            ws = Workspace(Path(d) / "ws")
            box = ToolBox(ws, allow_run=True, run_timeout=1)
            box.call("write_file", "loop.py", "while True: pass")
            self.assertIn("stopped after 1 seconds", box.call("run_python", "loop.py", None))
            box.call("write_file", "make.py", "open('out.txt','w').write('hi'); print('ran')")
            self.assertIn("ran", box.call("run_python", "make.py", None))
            self.assertIn("out.txt", box.written)


class ToolLoop(unittest.TestCase):
    def test_write_run_finish_end_to_end_is_verified_clean(self):
        llm = Seq(lambda p, k, n: WRITE_RUN_FINISH.get(n) if k == "step" and n <= 3 else None)
        with tempfile.TemporaryDirectory() as d:
            rep, ws, state = pipeline(d, llm, lambda l, w: ToolLoopHarness(l, w, 6, allow_run=True))
            self.assertTrue(rep["completed"])
            self.assertEqual(rep["integrity_events"], [])
            self.assertEqual((ws.root / "calc.py").read_text(), "print(2 + 3)")
            self.assertEqual(state["step_history"][0]["evidence"]["declared_files"], ["calc.py"])

    def test_model_that_ignores_the_protocol_just_answers_in_one_call(self):
        llm = Seq(lambda p, k, n: PROSE if k == "step" else None)
        with tempfile.TemporaryDirectory() as d:
            rep, _, _ = pipeline(d, llm, lambda l, w: ToolLoopHarness(l, w))
            self.assertTrue(rep["completed"])
            self.assertEqual(llm.n["step"], 2)                  # one call per step, no extra turns

    def test_turn_limit_forces_a_final_answer_turn(self):
        with tempfile.TemporaryDirectory() as d:
            llm = Seq(lambda p, k, n: "ACTION: list_files")
            out = ToolLoopHarness(llm, Workspace(d), max_turns=2).run_episode("do it", STEP, _Stats())
            self.assertEqual((out["harness_turns"], len(out["harness_log"])), (3, 2))
            self.assertIn("out of tool turns", llm.prompts[-1][1])

    def test_only_the_latest_tool_results_stay_in_the_prompt(self):
        with tempfile.TemporaryDirectory() as d:
            ws = Workspace(d)
            for i in range(1, 6):
                (ws.root / f"f{i}.txt").write_text(f"MARK{i}")
            llm = Seq(lambda p, k, n: f"ACTION: read_file\nPATH: f{n}.txt" if n <= 5 else PROSE)
            ToolLoopHarness(llm, ws, max_turns=8, keep_results=3).run_episode("do it", STEP, _Stats())
            last = llm.prompts[-1][1]
            for kept in ("MARK3", "MARK4", "MARK5"):
                self.assertIn(kept, last)
            self.assertNotIn("MARK1", last.split("Tool results so far")[1])


class AuditorCatchesMisbehaviour(unittest.TestCase):
    class After(CoTExecutor):
        """Acts on the workspace right after the episode, once."""
        done, action = False, None

        def run(self, goal, contract, step, stats):
            result = super().run(goal, contract, step, stats)
            if not type(self).done and type(self).action(self.workspace, result):
                type(self).done = True
            return result

    def test_declared_file_changed_after_the_receipt_is_a_violation(self):
        class Tamper(self.After):
            done = False
            action = staticmethod(lambda ws, r: bool(r.get("declared")) and
                                  (ws.root / next(iter(r["declared"]))).write_text("hacked") and True)
        llm = Seq(lambda p, k, n: WRITE_RUN_FINISH.get(n) if k == "step" and n <= 3 else None)
        with tempfile.TemporaryDirectory() as d:
            rep, _, _ = pipeline(d, llm, lambda l, w: ToolLoopHarness(l, w, 6, True), Tamper)
            self.assertIn(("step-1", "violation"), rep["integrity_events"])

    def test_undeclared_stray_file_is_still_suspect(self):
        class Stray(self.After):
            done = False
            action = staticmethod(lambda ws, r: (ws.root / "stray.txt").write_text("x") and True)
        llm = Seq(lambda p, k, n: PROSE if k == "step" else None)
        with tempfile.TemporaryDirectory() as d:
            rep, _, _ = pipeline(d, llm, lambda l, w: ToolLoopHarness(l, w), Stray)
            self.assertIn(("step-1", "suspect"), rep["integrity_events"])


class ExternalCLI(unittest.TestCase):
    CODE = "import sys; sys.stdin.read(); open('extra.txt','w').write('x'); print('A finished deliverable. ' * 8)"

    def test_command_output_is_the_deliverable_and_its_files_are_declared(self):
        cmd = shlex.join([sys.executable, "-c", self.CODE])
        with tempfile.TemporaryDirectory() as d:
            rep, ws, state = pipeline(d, MockLLM(), lambda l, w: ExternalCLIHarness(cmd, w))
            self.assertTrue(rep["completed"])
            self.assertEqual(rep["integrity_events"], [])
            self.assertIn("extra.txt", state["step_history"][0]["evidence"]["declared_files"])

    def test_failures_become_a_rejection_not_a_crash(self):
        with tempfile.TemporaryDirectory() as d:
            ws = Workspace(d)
            bad = ExternalCLIHarness(shlex.join([sys.executable, "-c", "import sys; sys.exit(3)"]), ws)
            out = bad.run_episode("p", STEP, _Stats())
            self.assertEqual((out["success"], out["output"]), (False, ""))
            missing = ExternalCLIHarness("definitely-not-a-command-xyz", ws).run_episode("p", STEP, _Stats())
            self.assertFalse(missing["success"])
            self.assertFalse(missing["harness_log"][0]["ok"])


class RunnerFlags(unittest.TestCase):
    def cli(self, d, argv):
        runs, out = Path(d) / "runs", io.StringIO()
        with mock.patch.object(agentic_run, "RUNS", runs), mock.patch.object(agentic_run, "ROOT", Path(d)), \
                mock.patch.object(agentic_run, "RunFiles", lambda: RunFiles(runs)), \
                mock.patch.object(agentic_run, "make_llm", lambda name, model=None: MockLLM()), \
                contextlib.redirect_stdout(out):
            return agentic_run.main(argv), out.getvalue()

    def test_tools_harness_runs_and_reports(self):
        with tempfile.TemporaryDirectory() as d:
            code, out = self.cli(d, ["Plan a thing", "--steps", "3", "--harness", "tools"])
        self.assertEqual(code, 0)
        self.assertIn("Harness       : tools", out)

    def test_bad_flag_combinations_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            for argv in (["x", "--harness", "cli"], ["x", "--harness", "tools", "--no-workspace"]):
                with self.assertRaises(SystemExit):
                    self.cli(d, argv)


if __name__ == "__main__":
    unittest.main()
