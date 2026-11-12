"""Executor 'garbage collection' of irrelevant observations (Plan-and-Act, Sec. 3.2).

The paper's executor removes redundant HTML before choosing its next action.
Here the observations are the verified facts the Manager hands over. Before each
round, facts that do not help the current step are dropped from the Executor's
prompt. Facts are only dropped or kept whole, never cut, so a code is never lost.

The final report step is exempt: it needs every fact. The contract itself is not
changed, so the Auditor still sees every dependency the step really has."""
import re

_STOP = {"this", "that", "with", "from", "step", "task", "plan", "have", "will", "into",
         "work", "then", "than", "them", "their", "about", "which", "each", "fact"}


def _words(text):
    return set(re.findall(r"[a-z]{4,}", str(text).lower())) - _STOP


def _num(key):
    m = re.search(r"(\d+)$", str(key))
    return int(m.group(1)) if m else 0


class ObservationGC:
    def __init__(self, max_chars=1500, min_keep=2, drop_irrelevant=True):
        self.max_chars, self.min_keep, self.drop_irrelevant = max_chars, min_keep, drop_irrelevant
        self.log = []
        self.discarded_reasoning_chars = 0

    def select(self, evidence, goal, final=False, step_id=None):
        """Return (kept, dropped_ids). Order of `evidence` is preserved."""
        before = sum(len(str(v)) for v in evidence.values())
        if final or not evidence:
            kept, dropped = dict(evidence), []
        else:
            goal_words = _words(goal)
            ranked = sorted(evidence.items(),
                            key=lambda kv: (len(goal_words & _words(kv[1])), _num(kv[0])),
                            reverse=True)
            keep, used = set(), 0
            for i, (k, v) in enumerate(ranked):
                relevant = bool(goal_words & _words(v)) or not self.drop_irrelevant
                size = len(str(v))
                if i < self.min_keep or (relevant and used + size <= self.max_chars):
                    keep.add(k)
                    used += size
            kept = {k: v for k, v in evidence.items() if k in keep}
            dropped = [k for k in evidence if k not in keep]
        self.log.append({"step": step_id, "kept": list(kept), "dropped": dropped,
                         "chars_before": before,
                         "chars_after": sum(len(str(v)) for v in kept.values())})
        return kept, dropped

    def note_discarded_reasoning(self, text):
        """The executor's raw reasoning is thrown away after each round; count it."""
        self.discarded_reasoning_chars += len(text or "")

    def summary(self):
        return {"rounds": len(self.log),
                "facts_dropped": sum(len(e["dropped"]) for e in self.log),
                "chars_saved": sum(e["chars_before"] - e["chars_after"] for e in self.log),
                "reasoning_chars_discarded": self.discarded_reasoning_chars}
