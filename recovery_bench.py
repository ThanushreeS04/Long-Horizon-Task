"""Step 3 - failure-recovery benchmark: baseline vs memory agent.

Usage:
  python recovery_bench.py
  python recovery_bench.py --sizes 10 20 40 --trials 5 --crashes 3
"""
import argparse
import copy
import csv
import json
import random
import tempfile
from datetime import datetime
from pathlib import Path

from agents.llm import make_llm, estimate_tokens
from agents.llm_agents import (LLMBaselineAgent, LLMMemoryAgent, make_plan,
                               make_code)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEFAULT_TASK = "Research three database options, compare their costs, and write a recommendation"


class FaultyLLM:
    """Wraps a model. For chosen steps the first N outputs are wrong."""

    def __init__(self, inner, faults):
        self.inner = inner
        self.faults = dict(faults)          # {step_number: bad_attempts}
        self.used = {}
        self.name = inner.name
        self.model = getattr(inner, "model", None)
        self.injected = 0

    def complete(self, prompt, kind="step", max_tokens=500, meta=None):
        r = self.inner.complete(prompt, kind=kind, max_tokens=max_tokens, meta=meta)
        if kind == "step":
            i = meta["index"]
            if self.used.get(i, 0) < self.faults.get(i, 0):
                self.used[i] = self.used.get(i, 0) + 1
                self.injected += 1
                bad = (f"Step {i} is finished and everything looks fine. "
                       f"The work was done carefully and nothing is left to do. ") * 2
                r = dict(r, text=bad, output_tokens=estimate_tokens(bad))
        return r


def pick_faults(n, rate, rng):
    """Choose which work steps get 1 or 2 bad outputs (final step excluded)."""
    work = list(range(1, n))
    count = round(len(work) * rate)
    chosen = rng.sample(work, count) if count else []
    return {i: rng.choice([1, 2]) for i in chosen}


def recall(final_output, n):
    if not final_output:
        return 0, n - 1
    found = sum(1 for i in range(1, n) if make_code(i) in final_output)
    return found, n - 1


def run_agent(kind, llm, task, crash_sizes, budget, mem_path):
    sessions = []

    def make():
        return (LLMBaselineAgent(llm) if kind == "baseline"
                else LLMMemoryAgent(llm, mem_path, budget))

    for k in crash_sizes:
        sessions.append(make().run(copy.deepcopy(task), stop_after=k))
    sessions.append(make().run(copy.deepcopy(task)))
    return sessions


def summarise(sessions, n):
    executed = [e for s in sessions for e in s["executed"]]
    seen, repeated, failed = set(), 0, 0
    for step_id, ok in executed:
        if ok:
            if step_id in seen:
                repeated += 1
            seen.add(step_id)
        else:
            failed += 1
    final_out = next((s["final_output"] for s in sessions if s["final_output"]), None)
    found, total = recall(final_out, n)
    series = [t for s in sessions for t in s["prompt_series"]]
    return {
        "completed": int(len(seen) == n),
        "unique_verified": len(seen),
        "llm_calls": sum(s["llm_calls"] for s in sessions),
        "repeated_steps": repeated,
        "rejected_attempts": failed,
        "input_tokens": sum(s["input_tokens"] for s in sessions),
        "output_tokens": sum(s["output_tokens"] for s in sessions),
        "peak_prompt_tokens": max(series) if series else 0,
        "recall_pct": round(100 * found / total, 1),
        "memory_seconds": round(sum(s["memory_seconds"] for s in sessions), 3),
    }


def mean(values):
    return round(sum(values) / len(values), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 40])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--crashes", type=int, default=3, help="number of crashes per run")
    ap.add_argument("--fault-rate", type=float, default=0.2,
                    help="fraction of work steps that get 1-2 bad outputs")
    ap.add_argument("--ctx-budget", type=int, default=4000)
    ap.add_argument("--llm", choices=["mock", "anthropic", "ollama"], default="mock")
    ap.add_argument("--model", default=None)
    ap.add_argument("--task", default=DEFAULT_TASK)
    args = ap.parse_args()

    base_llm = make_llm(args.llm, args.model)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tmp = tempfile.TemporaryDirectory(prefix="lh_", ignore_cleanup_errors=True)
    run_dir = Path(tmp.name)

    scenarios = [
        ("crashes only", args.crashes, 0.0),
        ("faults only", 0, args.fault_rate),
        ("crashes + faults", args.crashes, args.fault_rate),
    ]

    raw, summary = [], []
    for n in args.sizes:
        steps, _ = make_plan(base_llm, args.task, n)
        for label, n_crash, rate in scenarios:
            crash_every = max(1, (n - 1) // (n_crash + 1)) if n_crash else 0
            crash_sizes = [crash_every] * n_crash
            print(f"\n--- {n} steps | {label} | crash every {crash_every or '-'} | "
                  f"{args.trials} trial(s) ---", flush=True)
            per_agent = {"baseline": [], "memory": []}
            for t in range(1, args.trials + 1):
                faults = pick_faults(n, rate, random.Random(1000 * n + t))
                task = {"task_id": f"rec_{n}_{t}", "goal": args.task,
                        "success_criteria": [], "steps": steps}
                for kind in ("baseline", "memory"):
                    llm = FaultyLLM(base_llm, faults)
                    mem_path = run_dir / f"{stamp}_{n}_{label.replace(' ', '')}_{t}.json"
                    sessions = run_agent(kind, llm, task, crash_sizes, args.ctx_budget, mem_path)
                    row = summarise(sessions, n)
                    row["faults_injected"] = llm.injected
                    per_agent[kind].append(row)
                    raw.append({"steps": n, "scenario": label, "trial": t,
                                "agent": kind, "faults": faults, **row})
            for kind, rows in per_agent.items():
                agg = {"steps": n, "scenario": label, "agent": kind, "trials": len(rows),
                       "completion_pct": round(100 * mean([r["completed"] for r in rows]), 1)}
                for key in rows[0]:
                    if key != "completed":
                        agg[key] = mean([r[key] for r in rows])
                summary.append(agg)

    cols = [("Steps", "steps", 7), ("Scenario", "scenario", 18), ("Agent", "agent", 10),
            ("Done%", "completion_pct", 8), ("Calls", "llm_calls", 8),
            ("Repeat", "repeated_steps", 8), ("Rejected", "rejected_attempts", 10),
            ("InTok", "input_tokens", 10), ("PeakPrm", "peak_prompt_tokens", 9),
            ("Recall%", "recall_pct", 9)]
    print("\n" + "".join(f"{c:>{w}}" for c, _, w in cols))
    print("-" * sum(w for _, _, w in cols))
    for row in summary:
        print("".join(f"{str(row[k]):>{w}}" for _, k, w in cols))

    out_json = DATA / f"recovery_{stamp}.json"
    out_csv = DATA / f"recovery_{stamp}.csv"
    out_json.write_text(json.dumps({"backend": base_llm.name, "settings": vars(args),
                                    "summary": summary, "trials": raw}, indent=2), encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\nSaved: {out_json}\nSaved: {out_csv}")
    if tmp:
        tmp.cleanup()


if __name__ == "__main__":
    main()