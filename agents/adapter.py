"""AgentAdapter, ModelRouter and the per-role cost ledger (LongHorizon-Harness, Fig. 5 style).

Every model call made by a role goes through AgentAdapter.run_episode(), which
  * asks the ModelRouter for the model assigned to that role,
  * passes the role's budget (max output tokens, time limit),
  * records calls, tokens and seconds for that role in a Ledger.
A blocking local call cannot be cancelled half-way, so a time overrun is recorded
(over_budget) instead of being interrupted.
"""
import time
from contextlib import contextmanager

ROLES = ("manager", "executor", "auditor")

# The paper gives the executor 1800 s per round and the manager and auditor 300 s.
DEFAULT_BUDGETS = {"manager": {"max_tokens": 500, "seconds": 300},
                   "executor": {"max_tokens": 700, "seconds": 1800},
                   "auditor": {"max_tokens": 500, "seconds": 300}}


def model_name(llm):
    return getattr(llm, "model", None) or getattr(llm, "name", type(llm).__name__)


class ModelRouter:
    """Gives each role its own model; any role without one uses the default."""

    def __init__(self, default, **role_models):
        self.default = default
        self.role_models = {r: m for r, m in role_models.items() if m is not None}

    @classmethod
    def single(cls, llm):
        return cls(llm)

    def for_role(self, role):
        return self.role_models.get(role, self.default)

    def describe(self):
        return {r: model_name(self.for_role(r)) for r in ROLES}


class Ledger:
    """Calls, tokens and seconds per role."""

    def __init__(self):
        self.rows = {r: {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                         "seconds": 0.0, "over_budget": 0} for r in ROLES}

    def _row(self, role):
        return self.rows.setdefault(role, {"calls": 0, "input_tokens": 0,
                                           "output_tokens": 0, "seconds": 0.0, "over_budget": 0})

    def add_call(self, role, input_tokens, output_tokens, seconds, over_budget=False):
        row = self._row(role)
        row["calls"] += 1
        row["input_tokens"] += input_tokens
        row["output_tokens"] += output_tokens
        row["seconds"] += seconds
        row["over_budget"] += int(over_budget)

    @contextmanager
    def time_block(self, role):
        """Time spent by a rule-based role (no model call, so no tokens)."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._row(role)["seconds"] += time.perf_counter() - t0

    def summary(self, models=None):
        total = sum(r["input_tokens"] + r["output_tokens"] for r in self.rows.values())
        out = {}
        for role, r in self.rows.items():
            tokens = r["input_tokens"] + r["output_tokens"]
            out[role] = {**r, "seconds": round(r["seconds"], 4), "tokens": tokens,
                         "token_share": round(tokens / total, 4) if total else 0.0,
                         "model": (models or {}).get(role)}
        return out


class AgentAdapter:
    """One entry point for every model call a role makes."""

    def __init__(self, router, budgets=None):
        self.router = router
        self.budgets = {r: dict(b) for r, b in (budgets or DEFAULT_BUDGETS).items()}
        self.ledger = Ledger()

    def run_episode(self, role, prompt, budget=None, kind="step", meta=None):
        b = dict(self.budgets.get(role, {}))
        b.update(budget or {})
        t0 = time.perf_counter()
        result = self.router.for_role(role).complete(
            prompt, kind=kind, max_tokens=b.get("max_tokens", 500), meta=meta)
        seconds = time.perf_counter() - t0
        over = b.get("seconds") is not None and seconds > b["seconds"]
        self.ledger.add_call(role, result["input_tokens"], result["output_tokens"], seconds, over)
        return dict(result, role=role, seconds=round(seconds, 4), over_budget=over)

    def for_role(self, role):
        """A view that behaves like an LLM object but runs every call as this role."""
        return RoleLLM(self, role)

    def summary(self):
        return self.ledger.summary(self.router.describe())


class RoleLLM:
    """Same complete() signature as MockLLM / OllamaLLM, routed through the adapter."""

    def __init__(self, adapter, role):
        self.adapter, self.role = adapter, role
        base = adapter.router.for_role(role)
        self.name = getattr(base, "name", "llm")
        self.model = getattr(base, "model", None)

    def complete(self, prompt, kind="step", max_tokens=500, meta=None):
        return self.adapter.run_episode(self.role, prompt, {"max_tokens": max_tokens}, kind, meta)


def merge_ledgers(summaries):
    """Add up the ledger summaries of several sessions (crash + resume)."""
    summaries = [s for s in summaries if s]
    if not summaries:
        return None
    merged = {}
    for role in summaries[0]:
        rows = [s[role] for s in summaries if role in s]
        merged[role] = {k: (round(sum(r[k] for r in rows), 4) if k == "seconds"
                            else sum(r[k] for r in rows))
                        for k in ("calls", "input_tokens", "output_tokens", "seconds",
                                  "over_budget", "tokens")}
        merged[role]["model"] = rows[-1].get("model")
    total = sum(r["tokens"] for r in merged.values())
    for r in merged.values():
        r["token_share"] = round(r["tokens"] / total, 4) if total else 0.0
    return merged


def format_ledger(ledger):
    lines = [f"{'Role':<10}{'Model calls':>12}{'In tokens':>11}{'Out tokens':>12}"
             f"{'% of tokens':>13}{'Seconds':>10}  Model",
             "-" * 78]
    for role in ROLES:
        r = ledger.get(role)
        if r:
            lines.append(f"{role:<10}{r['calls']:>12}{r['input_tokens']:>11}{r['output_tokens']:>12}"
                         f"{100 * r['token_share']:>12.1f}%{r['seconds']:>10.2f}  {r.get('model') or ''}")
    return "\n".join(lines)