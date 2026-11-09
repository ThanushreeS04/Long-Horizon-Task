"""Optional model-based roles (both off by default): LLM replanning and an advisory LLM audit check."""
import re

from .llm_agents import now, parse_steps

CODE_LIKE = re.compile(r"K\d+-[0-9a-f]{4}|FACT-\d+")


class LLMReplanner:
    """Plan-and-Act idea: after an audited round, a model rewrites the descriptions of the
    remaining steps from the audit evidence.

    Only descriptions change. Step ids, codes, the final report step and the auditors' rules
    stay as they are, so the model cannot alter what the Auditor checks. A bad answer
    (wrong number of steps, empty text, or a code or FACT label) leaves the plan untouched.
    Every model call counts toward max_calls, and the count is kept in the saved state so a
    crash and restart cannot reset it."""

    def __init__(self, adapter, mode="on_reject", max_calls=3, desc_chars=200):
        if mode not in ("on_reject", "always"):
            raise ValueError("mode must be 'on_reject' or 'always'")
        self.adapter, self.mode = adapter, mode
        self.max_calls, self.desc_chars = max_calls, desc_chars

    @staticmethod
    def remaining(state):
        pending = set(state["pending_step_ids"])
        return [s for s in state["plan"] if s["id"] in pending and not s.get("final")]

    def calls_used(self, state):
        return len(state.get("llm_replans", []))

    def should_run(self, state, audit):
        if self.calls_used(state) >= self.max_calls or not self.remaining(state):
            return False
        return self.mode == "always" or not audit["verified"]

    def _prompt(self, state, audit, steps):
        facts = state["findings"][-5:]
        lines = [f"Task goal: {state['goal']}", "", "Verified so far:"]
        lines += [f"- {f[:120]}" for f in facts] or ["- (nothing yet)"]
        lines += ["", f"Latest audit: status={audit.get('status', '?')}, "
                      f"integrity={audit.get('integrity', '?')}. {audit.get('reason', '')[:200]}",
                  "", "Remaining steps:"]
        lines += [f"{i}. {s['description']}" for i, s in enumerate(steps, 1)]
        lines += ["", "If the audit shows a problem, rewrite the remaining steps so they avoid it; "
                      f"otherwise keep them. Keep exactly {len(steps)} steps in the same order. "
                      f"Do not add codes or FACT labels. Return ONLY a JSON array of {len(steps)} strings."]
        return "\n".join(lines)

    def revise(self, state, audit, round_no):
        steps = self.remaining(state)
        res = self.adapter.run_episode(
            "manager", self._prompt(state, audit, steps), {"max_tokens": 400}, kind="replan",
            meta={"goal": state["goal"], "descriptions": [s["description"] for s in steps]})
        new = parse_steps(res["text"], len(steps))
        why = None
        if len(new) != len(steps):
            why = f"expected {len(steps)} steps, got {len(new)}"
        elif any(not d.strip() for d in new):
            why = "empty description"
        elif any(CODE_LIKE.search(d) for d in new):
            why = "description contains a code or FACT label"
        entry = {"round": round_no, "trigger": self.mode, "applied": why is None,
                 "why": why or "ok", "changed": [], "at": now()}
        if why is None:
            for step, text in zip(steps, new):
                text = " ".join(text.split())[:self.desc_chars]
                if text != step["description"]:
                    step.setdefault("original_description", step["description"])
                    step["description"] = text
                    entry["changed"].append(step["id"])
        state.setdefault("llm_replans", []).append(entry)
        return entry


class LLMAuditCheck:
    """Advisory second opinion from a model. It is recorded and counted, but it never
    blocks a step: a small model's false rejections would otherwise dominate."""

    def __init__(self, adapter, max_chars=800):
        self.adapter, self.max_chars = adapter, max_chars

    def check(self, contract, step, result):
        text = result.get("output", "")[:self.max_chars]
        prompt = ("You are an independent reviewer.\n"
                  f"Step goal: {contract['goal']}\n\nStep output:\n{text}\n\n"
                  "Does the output actually carry out the step, rather than only repeating a label "
                  "or code? Answer on the first line with exactly YES or NO, then one short reason.")
        res = self.adapter.run_episode("auditor", prompt, {"max_tokens": 120}, kind="audit",
                                       meta={"step": step["id"]})
        first = res["text"].strip().upper()
        agrees = True if first.startswith("YES") else False if first.startswith("NO") else None
        return {"agrees": agrees, "note": res["text"].strip()[:200]}