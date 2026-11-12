"""Natural mode: the model just answers, like Claude does when you ask it a question.

The old checks demanded a 'FACT-n' line and a code like K3-1a2b inside every reply. That is a
test harness trick, not how a person wants an answer written. In natural mode:
  * the model is never asked for labels or codes (the harness adds its own FACT-n label to the
    saved summary, so memory, records and resume keep working);
  * the rule check looks at substance: long enough, no refusal, no unfilled placeholder;
  * the model Auditor judges the real content against plain-language criteria;
  * each step sees the FULL text of the earlier work it depends on, the final step sees all of it."""
import re

REFUSAL = re.compile(r"(\bas an ai\b|\bi (?:cannot|can't|am unable to|'m unable to) (?:help|assist|complete|do|provide)"
                     r"|\bi'?m sorry, but\b|\[insert|\[your [^\]]*\]|lorem ipsum|\btbd\b|<placeholder>)", re.I)


def natural_acceptance(step, min_chars=100, min_final=150):
    if step.get("final"):
        return ["one coherent, well-organised answer to the overall goal that combines the earlier results",
                f"at least {min_final} characters",
                "no refusals and no unfilled placeholders such as [insert ...]"]
    return ["the finished deliverable for this step, concrete and specific (not a plan to do it later)",
            f"at least {min_chars} characters",
            "consistent with the earlier verified results",
            "no refusals and no unfilled placeholders such as [insert ...]"]


def natural_rules(step):
    if step.get("final"):
        return ("Write the final answer to the overall goal as one coherent, well-organised piece that "
                "combines the verified work above. Do not just list the steps. Do not mention that you are "
                "an AI, do not ask questions and do not leave placeholders.")
    return ("Write the deliverable for this step directly, as finished work. Be concrete and specific, and "
            "use the verified work above where it is relevant. Do not mention that you are an AI, do not "
            "ask questions and do not leave placeholders.")


def natural_fact(text, step, max_chars=220):
    """Short summary kept in memory: the first paragraph, cut at a word boundary, with the
    harness's own FACT-n label (the model never has to write it)."""
    para = next((p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()), "")
    flat = " ".join(para.split())
    if len(flat) > max_chars:
        flat = flat[:max_chars].rsplit(" ", 1)[0] + "..."
    n = int(step["id"].split("-")[1])
    return f"FACT-{n}: {flat}"


class NaturalRules:
    """Drop-in for the Auditor's rule check (same verify() signature)."""

    def __init__(self, min_chars=100, min_final=150):
        self.min_chars, self.min_final = min_chars, min_final

    def verify(self, step, result, environment=None):
        text = result.get("output", "").strip()
        need = self.min_final if step.get("final") else self.min_chars
        reasons = []
        if len(text) < need:
            reasons.append(f"too short ({len(text)} of at least {need} characters)")
        if REFUSAL.search(text):
            reasons.append("contains a refusal or an unfilled placeholder")
        ok = not reasons
        return {"verified": ok, "evidence": {"chars": len(text)},
                "reason": "Substantive answer." if ok else "Rejected: " + "; ".join(reasons)}
