"""Chain-of-thought helpers (Plan-and-Act, Sec. 3.4).

A model is asked to reason inside <think>...</think> and then give its answer.
The reasoning is kept for the report and then DISCARDED: it never reaches the
saved task state and is never audited, only the answer is."""
import json
import re

_THINK = re.compile(r"<think>(.*?)</think>", re.S | re.I)

THINK_INSTRUCTION = ("First reason briefly inside <think>...</think> (this part is discarded). "
                     "Then give your answer.")


def split_think(text):
    """Return (reasoning, answer). A truncated, never-closed <think> block counts
    as reasoning only, so a half-finished thought is never mistaken for an answer."""
    text = text or ""
    reasoning = [m.strip() for m in _THINK.findall(text)]
    rest = _THINK.sub("", text)
    i = rest.lower().find("<think>")
    if i != -1:
        reasoning.append(rest[i + len("<think>"):].strip())
        rest = rest[:i]
    return "\n".join(r for r in reasoning if r), rest.strip()


def extract_json(text):
    """First JSON object or array in the answer part of `text`, else None."""
    _, rest = split_think(text)
    rest = re.sub(r"```(?:json)?", "", rest)
    decoder = json.JSONDecoder()
    for i, ch in enumerate(rest):
        if ch in "{[":
            try:
                return decoder.raw_decode(rest[i:])[0]
            except json.JSONDecodeError:
                continue
    return None
