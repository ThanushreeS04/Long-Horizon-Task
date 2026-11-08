"""Replanner: revises the remaining plan from audit evidence (the saved
rejection reasons), not from a fixed template. Rule-based, no model call."""
from .llm_agents import now, step_number
from .workflow import MAX_REPLANS, make_alternative


class Replanner:
    def __init__(self, max_replans=MAX_REPLANS, reason_chars=160):
        self.max_replans = max_replans
        self.reason_chars = reason_chars

    def can_replan(self, state):
        return len(state["replans"]) < self.max_replans

    def replan(self, state, exhausted):
        """Replace each exhausted step by an alternative. Changes `state` in
        place (caller saves it) and returns what was changed."""
        changes = []
        for sid in exhausted:
            if not self.can_replan(state):
                break
            plan = state["plan"]
            step = next(s for s in plan if s["id"] == sid)
            reasons = []
            for e in state["errors"]:
                if e["step_id"] == sid and e["reason"] not in reasons:
                    reasons.append(e["reason"])
            last = reasons[-1][:self.reason_chars] if reasons else "no reason recorded"
            alt = make_alternative(plan, step)
            # the reason mentions the OLD label and code; point it at the new step
            last = last.replace(step.get("code", "\0"), alt["code"]).replace(
                f"FACT-{step_number(step)}", f"FACT-{step_number(alt)}")
            alt["description"] += f" Earlier attempts were rejected: {last}"
            alt["failed_reasons"] = reasons
            if step.get("final"):                      # the report step itself is replaced
                step["final"], alt["final"] = False, True
                alt["required_codes"] = list(step.get("required_codes", []))
            for s in plan:                             # linear plans list codes explicitly
                if s.get("required_codes") and step.get("code") in s["required_codes"]:
                    s["required_codes"] = [alt["code"] if c == step["code"] else c
                                           for c in s["required_codes"]]
            state["pending_step_ids"] = [alt["id"] if p == sid else p
                                         for p in state["pending_step_ids"]]
            state["replaced_step_ids"].append(sid)
            record = {"failed": sid, "alternative": alt["id"], "reasons": reasons, "at": now()}
            state["replans"].append(record)
            changes.append(record)
        return changes