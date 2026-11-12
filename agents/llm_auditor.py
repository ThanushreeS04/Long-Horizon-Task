"""Model-based Auditor (LongHorizon-Harness, Sec. 2.4).

The rule checks (FACT label, code, length, dependencies, read-only integrity)
always run first and can never be overruled in the executor's favour: a model
cannot turn a rule failure into a pass. After the rules pass, a model with a
fresh context judges the output against the contract (plus any extra criteria
the Manager added) and may reject it. To stop a small model from deadlocking a
step, it can reject each step at most `max_model_rejections` times; an
unparseable answer counts as no opinion."""
from .cot import extract_json, split_think
from .mea import Auditor
from .timeouts import LLMTimeout

PROMPT = """You are an independent AUDITOR. You did not write this output; judge it only against the contract.

Step goal: {goal}
Acceptance criteria:
{criteria}

Output to audit:
<<<
{text}
>>>

{cot}Reply with ONLY a JSON object: {{"verdict": "pass" or "fail", "reason": "<one short sentence>"}}
Fail only if the output clearly does not carry out the step or breaks a criterion. Do not fail for style."""

COT = "First think briefly inside <think>...</think>. Then answer.\n"


class LLMAuditor(Auditor):
    uses_model = True

    def __init__(self, adapter, cot=True, max_model_rejections=1, max_chars=1200):
        super().__init__()
        self.adapter, self.cot = adapter, cot
        self.max_model_rejections, self.max_chars = max_model_rejections, max_chars
        self.rejections, self.log = {}, []

    def _judge(self, contract, result):
        criteria = list(contract["acceptance"]) + list(contract.get("extra_acceptance", []))
        prompt = PROMPT.format(goal=contract["goal"], criteria="\n".join(f"- {c}" for c in criteria),
                               text=result.get("output", "")[:self.max_chars],
                               cot=COT if self.cot else "")
        try:
            res = self.adapter.run_episode("auditor", prompt, {"max_tokens": 500 if self.cot else 200},
                                           kind="model_audit", meta={"step": contract["step_id"]})
        except LLMTimeout:
            return None, "model timed out", ""
        reasoning, _ = split_think(res["text"])
        obj = extract_json(res["text"])
        verdict = str(obj.get("verdict", "")).lower() if isinstance(obj, dict) else ""
        if verdict not in ("pass", "fail"):
            return None, "unparseable answer", reasoning
        return verdict == "pass", str(obj.get("reason", ""))[:200], reasoning

    def audit(self, contract, step, result, state):
        res = super().audit(contract, step, result, state)      # rules first, read-only check included
        if not res["verified"]:
            return res
        ok, reason, reasoning = self._judge(contract, result)
        sid = step["id"]
        info = {"verdict": {True: "pass", False: "fail", None: "no opinion"}[ok],
                "reason": reason, "blocked": False}
        if ok is False and self.rejections.get(sid, 0) < self.max_model_rejections:
            self.rejections[sid] = self.rejections.get(sid, 0) + 1
            info["blocked"] = True
            res.update(verified=False, status="incomplete",
                       reason=f"model auditor: {reason}", gaps=[f"model auditor: {reason}"])
        res["evidence"]["model_audit"] = info
        self.log.append({"step": sid, **info, "reasoning": reasoning[:300]})
        return res
