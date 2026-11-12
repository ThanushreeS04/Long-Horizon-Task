"""Executor, Auditor and the Manage-Execute-Audit loop (LongHorizon-Harness)."""
import hashlib
import json
import time
from contextlib import nullcontext

from .adapter import AgentAdapter, ModelRouter
from .llm_agents import _Stats, _execute, _report, step_number, step_rules
from .llm_replanner import LLMAuditCheck, LLMReplanner
from .manager import ASK, BLOCKED, DONE, Manager
from .memory_manager import MemoryManager
from .replanner import Replanner
from .taskstate import build_records
from .workflow import WorkflowAuditor
# noqa: F401 (MAX_REPLANS used later)


class Executor:
    """Fresh context every round: sees only the goal and the contract.
    Its output is a report; it does not prove the step is complete."""

    def __init__(self, llm):
        self.llm = llm

    @staticmethod
    def build_prompt(goal, contract, step):
        lines = [f"Task goal: {goal}", "",
                 f"Contract for {contract['step_id']}: {contract['goal']}",
                 "Acceptance criteria:"]
        lines += [f"- {a}" for a in contract["acceptance"]]
        lines += [f"- {a}" for a in contract.get("extra_acceptance", [])]
        if contract.get("human_answer"):
            lines += ["", f"Guidance from the human: {contract['human_answer']}"]
        if contract["evidence"]:
            lines += ["", "Verified facts you may rely on:"]
            lines += [f"[{k}] {v}" for k, v in contract["evidence"].items()]
        if contract["last_rejection"]:
            lines += ["", f"Your previous attempt was rejected. {contract['last_rejection']}"]
        lines += ["", contract.get("rules") or step_rules(step)]   # natural mode supplies its own rules
        return "\n".join(lines)

    def run(self, goal, contract, step, stats):
        return _execute(self.llm, self.build_prompt(goal, contract, step), step, stats)


def _digest(state):
    return hashlib.md5(json.dumps(state, sort_keys=True, default=str).encode()).hexdigest()


class Auditor:
    """Independent and read-only. Returns status (complete / incomplete / blocked)
    and integrity (clean / suspect / violation). Only complete + clean counts."""

    def __init__(self):
        self.rules = WorkflowAuditor()

    def audit(self, contract, step, result, state):
        before = _digest(state)
        base = self.rules.verify(step, result)
        text = result.get("output", "")
        missing_deps = [d for d in contract["depends_on"] if d not in contract["evidence"]]
        foreign = []
        if not step.get("final"):
            allowed = {step["id"], *contract["depends_on"]}
            foreign = [s["id"] for s in state["plan"]
                       if s["id"] not in allowed and s.get("code") and s["code"] in text]
        if missing_deps:
            status = "blocked"
        else:
            status = "complete" if base["verified"] else "incomplete"
        if _digest(state) != before:
            integrity = "violation"
        elif foreign:
            integrity = "suspect"
        else:
            integrity = "clean"
        verified = status == "complete" and integrity == "clean"
        reasons = [base["reason"]] if status == "incomplete" else []
        if status == "blocked":
            reasons.append(f"no verified evidence for {missing_deps}")
        if integrity != "clean":
            reasons.append(f"integrity {integrity}" +
                           (f": codes of unrelated steps {foreign}" if foreign else ""))
        return {"verified": verified, "status": status, "integrity": integrity,
                "evidence": dict(base["evidence"], foreign_codes=foreign,
                                 status=status, integrity=integrity),
                "gaps": reasons,
                "reason": base["reason"] if verified else " ".join(reasons)}


class MEAAgent:
    """Loop controller: Manager -> Executor -> Auditor -> MemoryManager, repeated
    until done, blocked, ask, interrupted or the round budget is used up."""

    def __init__(self, llm, memory_path, n_max=None, replan=True, router=None,
                 llm_replan="off", max_llm_replans=3, llm_audit=False, adapter=None,
                 manager=None, auditor=None, executor=None, state_replanner=None, ask_fn=None):
        self.llm = llm
        self.replan = replan
        self.replanner = Replanner()
        self.memory = MemoryManager(memory_path)
        self.adapter = adapter or AgentAdapter(router or ModelRouter.single(llm))
        self.manager, self.auditor = manager or Manager(), auditor or Auditor()
        self.executor = executor or Executor(self.adapter.for_role("executor"))
        self.ask_fn = ask_fn          # human-in-the-loop: ask_fn(question) -> answer text or None
        self.n_max = n_max
        self.llm_replanner = (LLMReplanner(self.adapter, llm_replan, max_llm_replans)
                              if llm_replan != "off" else None)
        if state_replanner is not None:          # model replans after EVERY audited step
            self.llm_replanner = state_replanner
        self.llm_audit = LLMAuditCheck(self.adapter) if llm_audit else None

    def _timed(self, role, component):
        """Rule-based roles are timed here; model-based ones are timed by the adapter."""
        return (nullcontext() if getattr(component, "uses_model", False)
                else self.adapter.ledger.time_block(role))

    def run(self, task, stop_after=None):
        start, stats, mem_s = time.perf_counter(), _Stats(), 0.0
        t0 = time.perf_counter()
        state = self.memory.create_or_load(task)
        for key, default in (("blocked_step_ids", []), ("replaced_step_ids", []),
                             ("replans", []), ("answers", {}), ("llm_replans", [])):
            state.setdefault(key, default)
        mem_s += time.perf_counter() - t0
        n_max = self.n_max or 3 * len(state["plan"])
        rounds, done, interrupted, stopped = 0, 0, False, "done"
        decisions, integrity_events, new_replans = [], [], []
        new_llm_replans, llm_checks = [], []
        while True:
            if stop_after is not None and done >= stop_after:
                interrupted, stopped = True, "interrupted"
                break
            if rounds >= n_max:
                stopped = "round_budget"
                break
            with self._timed("manager", self.manager):
                d = self.manager.decide(state)
            if d["exhausted"] and self.replan:
                t0 = time.perf_counter()
                changes = self.replanner.replan(state, d["exhausted"])
                if changes:                      # revised plan: decide again
                    self.memory.save_task(state)
                    mem_s += time.perf_counter() - t0
                    new_replans.extend(changes)
                    decisions.append({"round": rounds + 1, "decision": "replan",
                                      "step": changes[0]["failed"],
                                      "reason": f"{len(changes)} step(s) replaced"})
                    continue
            contract = d["contract"]
            decisions.append({"round": rounds + 1, "decision": d["decision"],
                              "step": contract["step_id"] if contract else None,
                              "reason": d["reason"]})
            if d["decision"] == ASK and self.ask_fn is not None:
                answer = self.ask_fn(d["reason"])
                if answer and str(answer).strip():          # answered: the manager re-decides
                    state["answers"][d["step_id"]] = str(answer).strip()
                    state["question"] = None
                    self.memory.save_task(state)
                    decisions.append({"round": rounds + 1, "decision": "ask_answered",
                                      "step": d["step_id"], "reason": str(answer).strip()[:120]})
                    continue
            if d["decision"] == DONE:
                break
            if d["decision"] in (BLOCKED, ASK):
                stopped = d["decision"]
                t0 = time.perf_counter()
                if stopped == BLOCKED:
                    for sid in d["exhausted"]:
                        if sid not in state["blocked_step_ids"]:
                            state["blocked_step_ids"].append(sid)
                else:
                    state["question"], state["question_step"] = d["reason"], d.get("step_id")
                self.memory.save_task(state)
                mem_s += time.perf_counter() - t0
                break
            step = next(s for s in state["plan"] if s["id"] == contract["step_id"])
            result = self.executor.run(state["goal"], contract, step, stats)
            with self._timed("auditor", self.auditor):
                audit = self.auditor.audit(contract, step, result, state)
            if self.llm_audit and audit["verified"]:
                check = self.llm_audit.check(contract, step, result)
                audit["evidence"]["llm_check"] = check
                llm_checks.append({"step": step["id"], **check})
            rounds += 1
            stats.executed.append((step["id"], audit["verified"]))
            if audit["integrity"] != "clean":
                integrity_events.append((step["id"], audit["integrity"]))
            t0 = time.perf_counter()
            state = self.memory.record_attempt(state, step, result, audit)
            mem_s += time.perf_counter() - t0
            if audit["verified"]:
                done += 1
                stats.step_outputs[step["id"]] = result["output"]
                art = {"id": f"output-{step['id']}", "step": step["id"], "status": "completed",
                       "sha": hashlib.md5(result["output"].encode()).hexdigest()[:12],
                       "chars": len(result["output"]), "audited_at": result["timestamp"]}
                if step.get("final"):
                    art["text"] = result["output"]
                    stats.final_output, stats.final_verified = result["output"], True
                t0 = time.perf_counter()
                state.setdefault("artifacts", []).append(art)
                self.memory.save_task(state)
                mem_s += time.perf_counter() - t0
            if self.llm_replanner and self.llm_replanner.should_run(state, audit):
                entry = self.llm_replanner.revise(state, audit, rounds)
                t0 = time.perf_counter()
                self.memory.save_task(state)
                mem_s += time.perf_counter() - t0
                new_llm_replans.append(entry)
                decisions.append({"round": rounds, "decision": "llm_replan", "step": None,
                                  "reason": f"rewrote {entry['changed']}" if entry["applied"]
                                  else f"not applied: {entry['why']}"})
        state["records"] = build_records(state)
        self.memory.save_task(state)
        final = next(s for s in state["plan"] if s.get("final"))["id"]
        completed = final in Manager._sets(state)[3] and not state["pending_step_ids"]
        if completed and not stats.final_verified:      # resumed run: the work was finished earlier
            stats.final_verified = True
            saved = next((a for a in reversed(state.get("artifacts", [])) if a.get("text")), None)
            if saved and not stats.final_output:
                stats.final_output = saved["text"]
        rep = _report("mea", stats, interrupted, time.perf_counter() - start, mem_s)
        rep.update({"completed": completed, "rounds": rounds, "stopped_because": stopped,
                    "decisions": decisions, "integrity_events": integrity_events,
                    "records_summary": state["records"]["counts"], "replans": len(new_replans), "replan_log": new_replans, "blocked": sorted(state["blocked_step_ids"]),
                    "step_outputs": stats.step_outputs,
                    "ledger": self.adapter.summary()})
        rep["calls_by_role"] = {r: v["calls"] for r, v in rep["ledger"].items()}
        # totals cover every role that made a model call (manager and auditor too)
        rep["llm_calls"] = sum(v["calls"] for v in rep["ledger"].values())
        rep["input_tokens"] = sum(v["input_tokens"] for v in rep["ledger"].values())
        rep["output_tokens"] = sum(v["output_tokens"] for v in rep["ledger"].values())
        rep.update({"llm_replans": sum(1 for e in new_llm_replans if e["applied"]),
                    "llm_replan_calls": len(new_llm_replans), "llm_replan_log": new_llm_replans,
                    "llm_audit_checked": len(llm_checks),
                    "llm_audit_disagreements": sum(1 for c in llm_checks if c["agrees"] is False),
                    "llm_audit_log": llm_checks})
        return rep