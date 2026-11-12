"""Manager role (LongHorizon-Harness): reads the saved task state and decides
what happens next. It never runs a step and never sees the environment."""
import re

from .llm_agents import MAX_ATTEMPTS, make_code, step_number
from .natural import natural_acceptance, natural_rules
from .workflow import ready_step

EXECUTE, DONE, BLOCKED, ASK = "execute", "done", "blocked", "ask"


def acceptance_for(step):
    """What the Auditor will check, written down BEFORE the step runs."""
    if step.get("final"):
        if step.get("depends_on"):
            codes = [make_code(step_number({"id": d})) for d in step["depends_on"]]
        else:
            codes = list(step.get("required_codes", []))
        return [f"report contains all {len(codes)} step codes: " + ", ".join(codes),
                "report is at least 40 characters"]
    i = step_number(step)
    return [f"a line starting FACT-{i}",
            f"the exact code {step['code']} inside that line",
            "at least 80 characters of work"]


class Manager:
    def __init__(self, max_attempts=MAX_ATTEMPTS, natural=False):
        self.max_attempts = max_attempts
        self.natural = natural          # natural mode: no FACT labels or codes asked of the model

    @staticmethod
    def _sets(state):
        pending = set(state["pending_step_ids"])
        blocked = set(state.get("blocked_step_ids", []))
        replaced = set(state.get("replaced_step_ids", []))
        verified = {s["id"] for s in state["plan"]} - pending - blocked - replaced
        return pending, blocked, replaced, verified

    @staticmethod
    def _deps(state, step, replaced):
        """Declared dependencies; a final step with none depends on every earlier step."""
        if step.get("final") and not step.get("depends_on"):
            return [s["id"] for s in state["plan"]
                    if not s.get("final") and s["id"] not in replaced]
        return list(step.get("depends_on", []))

    @staticmethod
    def _evidence(state, wanted):
        """Verified facts of the steps this one depends on (nothing else).
        History is compacted over time, so facts are also read from the
        saved findings, whose FACT-n label names the step that produced them."""
        wanted = set(wanted)
        out = {}
        for h in state["step_history"]:
            if h["step_id"] in wanted and h["status"] == "verified":
                out[h["step_id"]] = h["result"]
        for f in state.get("findings", []):
            m = re.search(r"FACT-(\d+)\b", str(f))
            if m and f"step-{m.group(1)}" in wanted:
                out[f"step-{m.group(1)}"] = str(f)
        return out

    def decide(self, state):
        """Return {'decision', 'contract', 'reason', 'exhausted'}."""
        pending, blocked, replaced, verified = self._sets(state)
        if not pending:
            return {"decision": DONE, "contract": None,
                    "reason": "no pending steps", "exhausted": []}
        failures = {}
        for e in state["errors"]:
            failures[e["step_id"]] = failures.get(e["step_id"], 0) + 1
        exhausted = sorted(sid for sid in pending
                           if failures.get(sid, 0) >= self.max_attempts)
        plan_pending = [s for s in state["plan"] if s["id"] in pending]
        step = ready_step(plan_pending, verified, blocked | set(exhausted), replaced)
        if step is None:
            why = (f"steps {exhausted} failed {self.max_attempts} times" if exhausted
                   else "pending steps wait on blocked steps")
            return {"decision": BLOCKED, "contract": None, "reason": why,
                    "exhausted": exhausted}
        if step.get("needs_input") and step["id"] not in state.get("answers", {}):
            return {"decision": ASK, "contract": None, "exhausted": exhausted, "step_id": step["id"],
                    "reason": step.get("question", f"input needed for {step['id']}")}
        last = [e["reason"] for e in state["errors"] if e["step_id"] == step["id"]]
        deps = self._deps(state, step, replaced)
        contract = {"step_id": step["id"], "goal": step["description"],
                    "acceptance": natural_acceptance(step) if self.natural else acceptance_for(step),
                    "rules": natural_rules(step) if self.natural else None,
                    "depends_on": deps,
                    "evidence": self._evidence(state, deps),
                    "attempt": failures.get(step["id"], 0) + 1,
                    "last_rejection": last[-1] if last else None,
                    "final": bool(step.get("final")),
                    "human_answer": state.get("answers", {}).get(step["id"])}
        return {"decision": EXECUTE, "contract": contract,
                "reason": f"{step['id']} is ready", "exhausted": exhausted}