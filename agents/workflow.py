"""Dependencies and replanning for both agents."""
import copy
import json
import time

from .llm_agents import (MAX_ATTEMPTS, OutputAuditor, _Stats, _execute, _report,
                         make_code, now, step_number, step_rules)
from .memory_manager import MemoryManager

MAX_REPLANS = 3


class WorkflowAuditor(OutputAuditor):
    """Same checks as OutputAuditor, except the final report must contain the
    codes of the steps it DEPENDS ON. After a replan, a replaced step is no
    longer required; its alternative is."""

    def verify(self, step, result, environment=None):
        if step.get("final") and step.get("depends_on"):
            text = result.get("output", "")
            required = [make_code(step_number({"id": d})) for d in step["depends_on"]]
            missing = [c for c in required if c not in text]
            ok = len(text.strip()) >= 40 and not missing
            return {"verified": ok,
                    "evidence": {"length": len(text.strip()), "missing_codes": missing},
                    "reason": "Final report present." if ok
                    else "Final report missing or lacks required step codes."}
        return super().verify(step, result, environment)


def add_dependencies(steps, width=3):
    """Turn a linear plan into `width` parallel chains; the final step needs all."""
    steps = copy.deepcopy(steps)
    work = [s for s in steps if not s.get("final")]
    for s in steps:
        i = step_number(s)
        if s.get("final"):
            s["depends_on"] = [w["id"] for w in work]
        elif i > width:
            s["depends_on"] = [f"step-{i - width}"]
        else:
            s["depends_on"] = []
    return steps


def ready_step(plan, verified, blocked, replaced):
    """First step in plan order that can run now, or None."""
    for s in plan:
        sid = s["id"]
        if sid in verified or sid in blocked or sid in replaced:
            continue
        if all(d in verified for d in s.get("depends_on", [])):
            return s
    return None


def make_alternative(plan, failed_step):
    """Insert a replacement step after the failed one and re-point dependents."""
    new_num = max(step_number(s) for s in plan) + 1
    alt = {"id": f"step-{new_num}",
           "description": f"Alternative approach for {failed_step['id']}: {failed_step['description']}",
           "operation": "llm_step", "expected": f"FACT-{new_num}",
           "code": make_code(new_num), "final": False,
           "depends_on": list(failed_step.get("depends_on", [])),
           "replaces": failed_step["id"]}
    for s in plan:
        if failed_step["id"] in s.get("depends_on", []):
            s["depends_on"] = [alt["id"] if d == failed_step["id"] else d
                               for d in s["depends_on"]]
    plan.insert(plan.index(failed_step) + 1, alt)
    return alt


def _finish(agent_name, stats, interrupted, start, mem_s, completed, replans, blocked):
    rep = _report(agent_name, stats, interrupted, time.perf_counter() - start, mem_s)
    rep.update({"completed": completed, "replans": replans, "blocked": sorted(blocked)})
    return rep


class WorkflowBaselineAgent:
    """No persistence. A crash loses the plan, blocked steps and failure counts."""

    def __init__(self, llm, replan=True):
        self.llm = llm
        self.replan = replan

    def run(self, task, stop_after=None):
        start, stats = time.perf_counter(), _Stats()
        plan = copy.deepcopy(task["steps"])
        verified, blocked, replaced, log = set(), set(), set(), []
        attempts, replans, done, interrupted = {}, 0, 0, False
        auditor = WorkflowAuditor()
        while True:
            if stop_after is not None and done >= stop_after:
                interrupted = True
                break
            step = ready_step(plan, verified, blocked, replaced)
            if step is None:
                break
            sid = step["id"]
            if attempts.get(sid, 0) >= MAX_ATTEMPTS:
                if self.replan and replans < MAX_REPLANS:
                    make_alternative(plan, step)
                    replaced.add(sid)
                    replans += 1
                else:
                    blocked.add(sid)
                continue
            history = "\n\n".join(log) if log else "(nothing yet)"
            prompt = (f"Task goal: {task['goal']}\n\nFull work log so far:\n{history}\n\n"
                      f"Now do {sid}: {step['description']}\n{step_rules(step)}")
            result = _execute(self.llm, prompt, step, stats)
            audit = auditor.verify(step, result)
            stats.executed.append((sid, audit["verified"]))
            attempts[sid] = attempts.get(sid, 0) + 1
            if audit["verified"]:
                verified.add(sid)
                log.append(f"[{sid}] {step['description']}\n{result['output']}")
                done += 1
                if step.get("final"):
                    stats.final_output = result["output"]
        final = next(s for s in plan if s.get("final"))["id"]
        return _finish("baseline", stats, interrupted, start, 0.0,
                       final in verified, replans, blocked)


class WorkflowMemoryAgent:
    """Saves plan, blocked steps, replans and failure counts in the JSON memory."""

    def __init__(self, llm, memory_path, ctx_budget=4000, replan=True):
        self.llm = llm
        self.memory = MemoryManager(memory_path)
        self.ctx_budget = ctx_budget
        self.replan = replan

    @staticmethod
    def _verified(state):
        out = set(s["id"] for s in state["plan"])
        out -= set(state["pending_step_ids"])
        out -= set(state["blocked_step_ids"])
        out -= set(state["replaced_step_ids"])
        return out

    def run(self, task, stop_after=None):
        start, stats, mem_s = time.perf_counter(), _Stats(), 0.0
        t0 = time.perf_counter()
        state = self.memory.create_or_load(task)
        state.setdefault("blocked_step_ids", [])
        state.setdefault("replaced_step_ids", [])
        state.setdefault("replans", [])
        mem_s += time.perf_counter() - t0
        auditor, done, interrupted = WorkflowAuditor(), 0, False
        new_replans = 0
        while state["pending_step_ids"]:
            if stop_after is not None and done >= stop_after:
                interrupted = True
                break
            verified = self._verified(state)
            plan_pending = [s for s in state["plan"] if s["id"] in state["pending_step_ids"]]
            step = ready_step(plan_pending, verified, set(state["blocked_step_ids"]),
                              set(state["replaced_step_ids"]))
            if step is None:
                break
            sid = step["id"]
            # failure count comes from the SAVED error log, so it survives a crash
            failures = sum(1 for e in state["errors"] if e["step_id"] == sid)
            if failures >= MAX_ATTEMPTS:
                t0 = time.perf_counter()
                if self.replan and len(state["replans"]) < MAX_REPLANS:
                    alt = make_alternative(state["plan"], step)
                    state["pending_step_ids"] = [alt["id"] if p == sid else p
                                                 for p in state["pending_step_ids"]]
                    state["replaced_step_ids"].append(sid)
                    state["replans"].append({"failed": sid, "alternative": alt["id"],
                                             "at": now()})
                    new_replans += 1
                else:
                    state["blocked_step_ids"].append(sid)
                self.memory.save_task(state)
                mem_s += time.perf_counter() - t0
                continue
            t0 = time.perf_counter()
            ctx = self.memory.build_bounded_context(state, budget_chars=self.ctx_budget)
            mem_s += time.perf_counter() - t0
            prompt = (f"Task goal: {state['goal']}\n\nCompact task memory (JSON):\n{json.dumps(ctx)}\n\n"
                      f"Now do {sid}: {step['description']}\n{step_rules(step)}")
            result = _execute(self.llm, prompt, step, stats)
            audit = auditor.verify(step, result)
            stats.executed.append((sid, audit["verified"]))
            t0 = time.perf_counter()
            state = self.memory.record_attempt(state, step, result, audit)
            mem_s += time.perf_counter() - t0
            if audit["verified"]:
                done += 1
                if step.get("final"):
                    stats.final_output = result["output"]
        final = next(s for s in state["plan"] if s.get("final"))["id"]
        completed = final in self._verified(state)
        return _finish("memory_enabled", stats, interrupted, start, mem_s,
                       completed, new_replans, state["blocked_step_ids"])