"""Model-based Manager (LongHorizon-Harness, Sec. 2.2).

The rule-based Manager stays in charge of what is SAFE: which step is ready,
dependency order, retry limits, done / blocked. A model then does the judgement
part on top of it: it writes the subtask contract in its own words, may add
extra acceptance criteria, and may decide the step needs information only a
human has (the ASK route). A bad or missing answer falls back to the rules."""
import re

from .cot import extract_json, split_think
from .llm_agents import MAX_ATTEMPTS
from .llm_replanner import CODE_LIKE
from .manager import ASK, EXECUTE, Manager
from .timeouts import LLMTimeout

PROMPT = """You are the MANAGER of a long task. You never do the work; you decide the next move.

Goal: {goal}

Verified facts so far:
{facts}

Next step that is ready: {sid}: {desc}
{extra}
{cot}Reply with ONLY a JSON object:
{{"decision": "execute" or "ask",
 "contract_goal": "<one sentence saying exactly what the executor must achieve>",
 "extra_acceptance": ["<at most 2 extra checkable criteria, or none>"],
 "question": "<only if decision is ask: one short question for the human>"}}
Use "ask" only if the step cannot be done sensibly without information that only the human can give."""

COT = "First think briefly inside <think>...</think>. Then answer.\n"


class LLMManager(Manager):
    uses_model = True

    def __init__(self, adapter, allow_ask=True, max_asks=2, cot=True, max_attempts=MAX_ATTEMPTS, natural=False):
        super().__init__(max_attempts, natural)
        self.adapter, self.allow_ask, self.max_asks, self.cot = adapter, allow_ask, max_asks, cot
        self.log = []

    def _prompt(self, state, contract):
        facts = "\n".join(f"- {str(f)[:120]}" for f in state.get("findings", [])[-5:]) or "- (nothing yet)"
        extra = []
        if contract.get("last_rejection"):
            extra.append(f"The last attempt was rejected: {contract['last_rejection'][:200]}")
        if state.get("answers"):
            extra.append("Human answers so far: " + "; ".join(
                f"{k}: {v}" for k, v in state["answers"].items()))
        return PROMPT.format(goal=state["goal"], facts=facts, sid=contract["step_id"],
                             desc=contract["goal"], extra="\n".join(extra),
                             cot=COT if self.cot else "")

    def _note(self, sid, decision, source, reasoning=""):
        self.log.append({"step": sid, "decision": decision, "source": source,
                         "reasoning": reasoning[:300]})

    def decide(self, state):
        d = super().decide(state)                       # rules decide what is allowed
        if d["decision"] != EXECUTE:
            return d
        contract = d["contract"]
        sid = contract["step_id"]
        try:
            res = self.adapter.run_episode(
                "manager", self._prompt(state, contract),
                {"max_tokens": 600 if self.cot else 300}, kind="manager",
                meta={"goal": state["goal"], "desc": contract["goal"]})
        except LLMTimeout:
            self._note(sid, "execute", "fallback", "model timed out")
            return d
        reasoning, _ = split_think(res["text"])
        obj = extract_json(res["text"])
        if not isinstance(obj, dict):
            self._note(sid, "execute", "fallback", "unparseable answer")
            return d
        answers = state.get("answers", {})
        if (str(obj.get("decision", "")).lower() == "ask" and self.allow_ask
                and sid not in answers and len(answers) < self.max_asks):
            step = next(s for s in state["plan"] if s["id"] == sid)
            step["needs_input"] = True
            step["question"] = str(obj.get("question") or "").strip()[:300] or \
                f"What should I know before doing {sid} ({contract['goal'][:80]})?"
            self._note(sid, "ask", "model", reasoning)
            return super().decide(state)                # now returns ASK
        goal = " ".join(str(obj.get("contract_goal") or "").split())
        if goal and len(goal) <= 300 and not CODE_LIKE.search(goal):
            contract["goal"] = goal
        extra = obj.get("extra_acceptance")
        contract["extra_acceptance"] = [
            " ".join(str(x).split())[:140] for x in (extra if isinstance(extra, list) else [])
            if str(x).strip() and not CODE_LIKE.search(str(x))][:2]
        contract["manager_reasoning"] = reasoning[:300]
        self._note(sid, "execute", "model", reasoning)
        return d
