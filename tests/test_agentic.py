import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.adapter import AgentAdapter, ModelRouter
from agents.cleanup import RunFiles
from agents.cot import extract_json, split_think
from agents.cot_executor import CoTExecutor
from agents.hl_planner import HighLevelPlanner
from agents.llm import MockLLM
from agents.llm_auditor import LLMAuditor
from agents.llm_manager import LLMManager
from agents.mea import MEAAgent
from agents.observation_gc import ObservationGC
from agents.state_replanner import StateReplanner


class Scripted(MockLLM):
    """MockLLM whose answer for chosen kinds comes from a function(prompt, meta, call_no)."""

    def __init__(self, **by_kind):
        self.by_kind, self.calls = by_kind, {}

    def complete(self, prompt, kind="step", max_tokens=500, meta=None):
        self.calls[kind] = self.calls.get(kind, 0) + 1
        if kind in self.by_kind:
            text = self.by_kind[kind](prompt, meta or {}, self.calls[kind])
            return {"text": text, "input_tokens": 10, "output_tokens": 10}
        return super().complete(prompt, kind, max_tokens, meta)


def run_pipeline(llm, tmp, n=5, ask_fn=None, needs_input=None, rule_manager=False, rule_auditor=False,
                 replan=True, **kw):
    adapter = AgentAdapter(ModelRouter.single(llm))
    plan = HighLevelPlanner(adapter).plan("Plan a launch, budget it, and announce it", n)
    steps = plan["steps"]
    if needs_input:
        steps[needs_input - 1].update(needs_input=True, question="Audience?")
    gc = ObservationGC()
    manager = LLMManager(adapter) if not rule_manager else None
    auditor = LLMAuditor(adapter, **kw) if not rule_auditor else None
    agent = MEAAgent(llm, Path(tmp) / "s.json", adapter=adapter, manager=manager, auditor=auditor,
                     executor=CoTExecutor(adapter.for_role("executor"), gc=gc),
                     state_replanner=StateReplanner(adapter) if replan else None, ask_fn=ask_fn)
    task = {"task_id": "t", "goal": "Plan a launch", "success_criteria": [], "steps": steps}
    rep = agent.run(task)
    return rep, agent.memory.get_task("t"), manager, auditor, plan


class CoTHelpers(unittest.TestCase):
    def test_split_and_extract(self):
        r, a = split_think("<think>hmm</think>\n[1, 2]")
        self.assertEqual((r, a), ("hmm", "[1, 2]"))
        self.assertEqual(extract_json("<think>x</think>```json\n{\"a\": 1}\n```"), {"a": 1})
        self.assertIsNone(extract_json("no json here"))

    def test_unclosed_think_is_not_an_answer(self):
        r, a = split_think("<think>half a thought [1,2]")
        self.assertEqual(a, "")
        self.assertIn("half a thought", r)


class Cleanup(unittest.TestCase):
    def test_only_registered_files_in_runs_are_deleted(self):
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d) / "runs"
            rf = RunFiles(runs)
            mine, txt = rf.register(runs / "a.json"), rf.register(runs / "b.txt")
            other = runs / "unregistered.csv"
            for p in (mine, txt, other):
                p.write_text("x")
            outside = Path(d) / "keep.json"
            outside.write_text("x")
            with self.assertRaises(ValueError):
                rf.register(outside)
            gone = rf.purge_previous()
            self.assertEqual([Path(g).name for g in gone], ["a.json"])
            self.assertTrue(txt.exists() and other.exists() and outside.exists())
            self.assertFalse(rf.manifest.exists())
            self.assertEqual(rf.purge_previous(), [])


class GarbageCollection(unittest.TestCase):
    def test_final_step_keeps_every_fact(self):
        gc = ObservationGC(max_chars=10, min_keep=1)
        ev = {f"step-{i}": f"FACT-{i}: unrelated text K{i}-abcd" for i in range(1, 6)}
        kept, dropped = gc.select(ev, "anything", final=True)
        self.assertEqual(kept, ev)
        self.assertEqual(dropped, [])

    def test_irrelevant_facts_are_dropped_but_never_cut(self):
        gc = ObservationGC(min_keep=1)
        ev = {"step-1": "FACT-1: budget spreadsheet totals K1-aaaa",
              "step-2": "FACT-2: office plants watering rota K2-bbbb",
              "step-3": "FACT-3: budget approval workflow K3-cccc"}
        kept, dropped = gc.select(ev, "Review the budget approval", step_id="step-4")
        self.assertEqual(dropped, ["step-2"])
        self.assertTrue(all(v == ev[k] for k, v in kept.items()))   # whole, so codes survive
        self.assertGreater(gc.summary()["chars_saved"], 0)


class Planner(unittest.TestCase):
    def test_cot_plan_has_reasoning_and_final_step(self):
        adapter = AgentAdapter(ModelRouter.single(MockLLM()))
        p = HighLevelPlanner(adapter).plan("Do A, then B", 5)
        self.assertEqual(p["source"], "model")
        self.assertTrue(p["reasoning"])
        self.assertEqual(len(p["steps"]), 5)
        self.assertTrue(p["steps"][-1]["final"])

    def test_garbage_falls_back_to_one_shot_planner(self):
        adapter = AgentAdapter(ModelRouter.single(Scripted(cot_plan=lambda p, m, n: "")))
        p = HighLevelPlanner(adapter).plan("Do A, then B", 5)
        self.assertEqual(p["source"], "fallback")
        self.assertEqual(len(p["steps"]), 5)


class FullPipeline(unittest.TestCase):
    def test_everything_on_completes_and_strips_reasoning(self):
        with tempfile.TemporaryDirectory() as d:
            rep, state, *_ = run_pipeline(MockLLM(), d)
            self.assertTrue(rep["completed"])
            self.assertNotIn("<think>", rep["final_output"])
            self.assertTrue(all("<think>" not in h["result"] for h in state["step_history"]))

    def test_ask_is_answered_and_reaches_the_executor(self):
        prompts = []

        def spy(prompt, meta, n):
            prompts.append(prompt)
            return MockLLM().complete(prompt, "step", 500, meta)["text"]
        questions = []

        def ask(q):
            questions.append(q)
            return "Audience is students"
        with tempfile.TemporaryDirectory() as d:
            rep, state, *_ = run_pipeline(Scripted(step=spy), d, ask_fn=ask, needs_input=2)
            self.assertTrue(rep["completed"])
            self.assertEqual(questions, ["Audience?"])
            self.assertEqual(state["answers"], {"step-2": "Audience is students"})
            self.assertTrue(any("Audience is students" in p for p in prompts))

    def test_ask_without_a_human_stops_with_ask_and_can_resume(self):
        with tempfile.TemporaryDirectory() as d:
            rep, state, *_ = run_pipeline(MockLLM(), d, ask_fn=None, needs_input=2)
            self.assertEqual(rep["stopped_because"], "ask")
            self.assertFalse(rep["completed"])

    def test_model_manager_can_raise_ask(self):
        ask = lambda p, m, n: json.dumps({"decision": "ask", "question": "Which region?"}) if n == 2 \
            else '{"decision": "execute"}'
        qs = []
        with tempfile.TemporaryDirectory() as d:
            rep, state, mgr, *_ = run_pipeline(Scripted(manager=ask), d,
                                               ask_fn=lambda q: qs.append(q) or "EU")
            self.assertTrue(rep["completed"])
            self.assertEqual(qs, ["Which region?"])
            self.assertIn("ask", [e["decision"] for e in mgr.log])

    def test_garbage_manager_answer_falls_back_to_rules(self):
        with tempfile.TemporaryDirectory() as d:
            rep, _, mgr, *_ = run_pipeline(Scripted(manager=lambda p, m, n: "lol"), d)
            self.assertTrue(rep["completed"])
            self.assertTrue(all(e["source"] == "fallback" for e in mgr.log))

    def test_model_auditor_rejects_once_then_cannot_deadlock(self):
        fail = lambda p, m, n: '{"verdict": "fail", "reason": "too vague"}'
        with tempfile.TemporaryDirectory() as d:
            rep, state, _, aud, _ = run_pipeline(Scripted(model_audit=fail), d, replan=False)
            self.assertTrue(rep["completed"])
            blocked = [a for a in aud.log if a["blocked"]]
            self.assertEqual(len(blocked), 5)               # once per step, never twice
            self.assertGreater(rep["rounds"], 5)

    def test_model_cannot_pass_what_the_rules_fail(self):
        no_fact = lambda p, m, n: "I did the work, trust me. " * 10
        with tempfile.TemporaryDirectory() as d:
            rep, *_ = run_pipeline(Scripted(step=no_fact), d, replan=False)
            self.assertFalse(rep["completed"])

    def test_replanner_can_add_a_step(self):
        def add(p, m, n):
            d = m["descriptions"]
            return json.dumps(d + ["Check legal requirements"] if n == 1 else d)
        with tempfile.TemporaryDirectory() as d:
            rep, state, *_ = run_pipeline(Scripted(state_replan=add), d)
            added = [s for s in state["plan"] if s.get("added_by_replan")]
            self.assertEqual(len(added), 1)
            self.assertTrue(rep["completed"])
            self.assertIn(added[0]["code"], rep["final_output"])      # final report covers it

    def test_replanner_can_drop_a_step(self):
        def drop(p, m, n):
            d = m["descriptions"]
            return json.dumps(d[:-1] if n == 1 else d)
        with tempfile.TemporaryDirectory() as d:
            rep, state, *_ = run_pipeline(Scripted(state_replan=drop), d)
            self.assertEqual(len(state["replaced_step_ids"]), 1)
            gone = next(s for s in state["plan"] if s["id"] == state["replaced_step_ids"][0])
            self.assertTrue(rep["completed"])
            self.assertNotIn(gone["code"], rep["final_output"])

    def test_replanner_rejects_codes_and_too_many_steps(self):
        bad = lambda p, m, n: json.dumps(m["descriptions"] + ["a", "b", "c", "d"])
        with tempfile.TemporaryDirectory() as d:
            rep, state, *_ = run_pipeline(Scripted(state_replan=bad), d)
            self.assertTrue(rep["completed"])
            self.assertFalse(any(e["applied"] for e in state["llm_replans"]))
            self.assertFalse(any(s.get("added_by_replan") for s in state["plan"]))


if __name__ == "__main__":
    unittest.main()
