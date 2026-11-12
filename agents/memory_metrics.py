"""M6 - memory instrumentation: store size, context size, read/write latency,
Python peak RAM, and whether the context still contains the facts the next steps need."""
import json
import time
import tracemalloc
from statistics import mean

from .taskstate import _step_of


class MemoryProbe:
    """Wrap a MemoryManager; every load/save/context build is timed and sized."""

    def __init__(self, memory, budget_chars=2000):
        self.memory, self.budget = memory, budget_chars
        self.reads, self.writes, self.ctx_builds = [], [], []
        self.samples = []

    def _timed(self, bucket, fn, *a, **kw):
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        bucket.append(time.perf_counter() - t0)
        return out

    def load(self, task_id):
        return self._timed(self.reads, self.memory.get_task, task_id)

    def save(self, state):
        self._timed(self.writes, self.memory.save_task, state)

    def context(self, state):
        ctx = self._timed(self.ctx_builds, self.memory.build_bounded_context, state, self.budget)
        store_chars = len(json.dumps(state, default=str))
        self.samples.append({"store_chars": store_chars, "context_chars": len(json.dumps(ctx)),
                             "within_budget": len(json.dumps(ctx)) <= self.budget,
                             "facts_dropped": ctx.get("older_facts_dropped", 0)})
        return ctx

    @staticmethod
    def fact_recall(context, needed_steps):
        """Share of the facts the next steps depend on that are visible in the context."""
        if not needed_steps:
            return 1.0
        seen = {_step_of(f) for f in context.get("key_facts", [])}
        return sum(1 for s in needed_steps if s in seen) / len(needed_steps)

    def report(self):
        ms = lambda xs: round(mean(xs) * 1000, 3) if xs else 0.0
        last = self.samples[-1] if self.samples else {}
        return {"read_ms_avg": ms(self.reads), "write_ms_avg": ms(self.writes),
                "context_build_ms_avg": ms(self.ctx_builds),
                "store_chars_last": last.get("store_chars", 0),
                "context_chars_last": last.get("context_chars", 0),
                "context_over_budget": sum(1 for s in self.samples if not s["within_budget"]),
                "max_facts_dropped": max((s["facts_dropped"] for s in self.samples), default=0)}


class PeakRAM:
    """with PeakRAM() as r: ...  then r.peak_kib (Python allocations only)."""

    def __enter__(self):
        tracemalloc.start()
        return self

    def __exit__(self, *exc):
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_kib = round(peak / 1024, 1)