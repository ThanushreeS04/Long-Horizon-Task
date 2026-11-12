"""Dynamic replanning after EVERY audited step (Plan-and-Act, Sec. 3.3).

After each round the model sees the goal, the verified facts, the plan that is
still pending, the last action and its audit result, and returns the new
remaining plan. It may keep, rewrite, drop or add steps. Every change goes
through the same bookkeeping the rest of the system relies on (required codes,
dependencies, pending and replaced lists), and the Auditor's rules are never
touched. A bad answer leaves the plan unchanged."""
from .cot import extract_json, split_think
from .llm_agents import make_code, now, step_number
from .llm_replanner import CODE_LIKE, LLMReplanner

PROMPT = """You are the PLANNER, updating the plan after one step finished.

Goal: {goal}

Verified facts so far:
{facts}
{human}
Last action: {last}

Current plan still to do (in order):
{plan}

{cot}Return ONLY a JSON array of strings: the new remaining plan.
- Keep a step word-for-word if it is still right.
- Rewrite it if the new facts change what it should do.
- Leave a step out if it is no longer needed.
- Add at most {max_add} new step(s) if something important is missing.
Do not include codes or FACT labels. Return between 1 and {max_len} items."""

COT = "First think briefly inside <think>...</think> about what the new facts change. Then answer.\n"


class StateReplanner(LLMReplanner):
    def __init__(self, adapter, mode="always", max_calls=50, max_add_per_call=2,
                 max_total_added=4, cot=True, desc_chars=200):
        super().__init__(adapter, mode, max_calls, desc_chars)
        self.max_add, self.max_total_added, self.cot = max_add_per_call, max_total_added, cot

    def _prompt(self, state, audit, steps):
        last = (state.get("step_history") or [{}])[-1]
        facts = "\n".join(f"- {str(f)[:120]}" for f in state["findings"][-6:]) or "- (nothing yet)"
        human = ("Human answers: " + "; ".join(f"{k}: {v}" for k, v in state["answers"].items()) + "\n"
                 if state.get("answers") else "")
        last_txt = (f"{last.get('step_id', '?')} -> {last.get('status', '?')} "
                    f"(audit: {audit.get('status', '?')}/{audit.get('integrity', '?')}; "
                    f"{str(audit.get('reason', ''))[:160]}) {str(last.get('result', ''))[:120]}")
        plan = "\n".join(f"{i}. {s['description']}" for i, s in enumerate(steps, 1))
        return PROMPT.format(goal=state["goal"], facts=facts, human=human, last=last_txt, plan=plan,
                             cot=COT if self.cot else "", max_add=self.max_add,
                             max_len=len(steps) + self.max_add)

    def _check(self, state, steps, new):
        if not new:
            return "empty plan"
        if len(new) > len(steps) + self.max_add:
            return f"too many new steps ({len(new)} for {len(steps)})"
        if any(not d.strip() for d in new):
            return "empty description"
        if any(CODE_LIKE.search(d) for d in new):
            return "description contains a code or FACT label"
        already = sum(1 for s in state["plan"] if s.get("added_by_replan"))
        if already + max(0, len(new) - len(steps)) > self.max_total_added:
            return "too many steps added overall"
        return None

    @staticmethod
    def apply(state, steps, new, desc_chars=200):
        """Rewrite, drop and add steps. Returns (rewritten, dropped, added) id lists."""
        plan, pending = state["plan"], state["pending_step_ids"]
        final = next(s for s in plan if s.get("final"))
        rewritten, dropped, added = [], [], []
        for step, text in zip(steps, new):
            text = " ".join(text.split())[:desc_chars]
            if text != step["description"]:
                step.setdefault("original_description", step["description"])
                step["description"] = text
                rewritten.append(step["id"])
        for step in steps[len(new):]:                              # no longer needed
            sid = step["id"]
            state["pending_step_ids"] = pending = [p for p in pending if p != sid]
            state.setdefault("replaced_step_ids", []).append(sid)
            step["dropped_by_replan"] = True
            if step.get("code") in final.get("required_codes", []):
                final["required_codes"] = [c for c in final["required_codes"] if c != step["code"]]
            for s in plan:
                if sid in s.get("depends_on", []):
                    s["depends_on"] = [d for d in s["depends_on"] if d != sid]
            dropped.append(sid)
        for text in new[len(steps):]:                              # something was missing
            num = max(step_number(s) for s in plan) + 1
            step = {"id": f"step-{num}", "description": " ".join(text.split())[:desc_chars],
                    "operation": "llm_step", "expected": f"FACT-{num}", "code": make_code(num),
                    "final": False, "added_by_replan": True}
            plan.insert(plan.index(final), step)                   # runs before the final report
            pending.append(step["id"])
            if "required_codes" in final:
                final["required_codes"] = list(final["required_codes"]) + [step["code"]]
            if final.get("depends_on"):
                final["depends_on"] = list(final["depends_on"]) + [step["id"]]
            added.append(step["id"])
        return rewritten, dropped, added

    def revise(self, state, audit, round_no):
        steps = self.remaining(state)
        res = self.adapter.run_episode(
            "replanner", self._prompt(state, audit, steps), {"max_tokens": 700 if self.cot else 400},
            kind="state_replan", meta={"goal": state["goal"],
                                       "descriptions": [s["description"] for s in steps]})
        reasoning, _ = split_think(res["text"])
        obj = extract_json(res["text"])
        new = [str(x) for x in obj] if isinstance(obj, list) else []
        why = self._check(state, steps, new) if new else "unparseable answer"
        entry = {"round": round_no, "trigger": self.mode, "applied": why is None,
                 "why": why or "ok", "changed": [], "dropped": [], "added": [],
                 "reasoning": reasoning[:300], "at": now()}
        if why is None:
            entry["changed"], entry["dropped"], entry["added"] = self.apply(
                state, steps, new, self.desc_chars)
        state.setdefault("llm_replans", []).append(entry)
        return entry
