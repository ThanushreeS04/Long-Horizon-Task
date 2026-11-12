"""Executor with chain-of-thought and observation garbage collection.

Still a fresh context every round: it sees only the contract, the human's
guidance (if any) and the facts the garbage collector lets through. Its
reasoning is stripped from the output before the Auditor sees it and is never
saved: only the answer survives the round."""
from .cot import THINK_INSTRUCTION, split_think
from .llm_agents import fact_text
from .mea import Executor
from .natural import natural_fact
from .observation_gc import ObservationGC


class _Boost:
    """Gives the model extra output tokens for its <think> block."""

    def __init__(self, llm, extra):
        self.llm, self.extra = llm, extra
        self.name, self.model = getattr(llm, "name", "llm"), getattr(llm, "model", None)

    def complete(self, prompt, kind="step", max_tokens=500, meta=None):
        return self.llm.complete(prompt, kind=kind, max_tokens=max_tokens + self.extra, meta=meta)


class CoTExecutor(Executor):
    def __init__(self, llm, cot=True, gc=None, extra_tokens=400, workspace=None, natural=False,
                 fact_chars=1200, final_budget=9000, harness=None):
        super().__init__(_Boost(llm, extra_tokens) if cot else llm)
        self.cot, self.gc = cot, gc or ObservationGC()
        self.workspace = workspace          # if set, deliverables are written there as artifacts
        self.natural, self.fact_chars, self.final_budget = natural, fact_chars, final_budget
        self._stats = None
        self.harness = harness              # ToolLoopHarness / ExternalCLIHarness, or None = one model call
        self.reasoning_log = []

    def _full_work(self, evidence, final):
        """Replace each short fact by the full earlier deliverable (workspace file, else this run's
        output), cut to a size limit, so a step builds on real work instead of a 200-character stub."""
        out, budget = {}, self.final_budget
        for sid, short in evidence.items():
            text = None
            if self.workspace is not None:
                try:
                    text = self.workspace.read_artifact(self.workspace.path_for(sid).name)
                except (ValueError, OSError):
                    text = None
            if text is None and self._stats is not None:
                text = self._stats.step_outputs.get(sid)
            text = (text or short).strip()
            cap = min(self.fact_chars * (2 if final else 1), max(budget, 200))
            if len(text) > cap:
                text = text[:cap].rsplit(" ", 1)[0] + " ..."
            budget -= len(text)
            out[sid] = text
        return out

    def build_prompt(self, goal, contract, step):
        view = dict(contract)
        view["evidence"], _ = self.gc.select(contract["evidence"], contract["goal"],
                                             final=bool(contract.get("final")),
                                             step_id=contract["step_id"])
        if self.natural:
            view["evidence"] = self._full_work(view["evidence"], bool(contract.get("final")))
        prompt = Executor.build_prompt(goal, view, step)
        return prompt + ("\n\n" + THINK_INSTRUCTION if self.cot else "")

    def run(self, goal, contract, step, stats):
        self._stats = stats
        before = self.workspace.snapshot() if self.workspace else None
        if self.harness is not None:
            result = self.harness.run_episode(self.build_prompt(goal, contract, step), step, stats)
        else:
            result = super().run(goal, contract, step, stats)
        reasoning, answer = split_think(result.get("output", ""))
        if self.cot or reasoning or self.natural:
            result["output"] = answer
            make_fact = natural_fact if self.natural else fact_text
            result["message"] = make_fact(answer, step) if answer else ""
            result["success"] = bool(answer.strip())
        if self.workspace is not None:
            if result.get("output", "").strip():
                result["artifact"] = self.workspace.write_step(step["id"], result["output"])
            result["before_snapshot"] = before
        self.gc.note_discarded_reasoning(reasoning)       # raw trajectory is thrown away
        self.reasoning_log.append({"step": step["id"], "chars": len(reasoning),
                                   "preview": reasoning[:200],
                                   "tool_calls": [f"{l['action']}{'' if l.get('ok', True) else ' (failed)'}"
                                                  for l in result.get("harness_log", [])]})
        return result
