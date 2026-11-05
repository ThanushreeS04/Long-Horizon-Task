"""Repeatable benchmark: baseline vs memory agent over several task lengths.

Usage:
  python bench.py                                   (mock model, fast)
  python bench.py --sizes 6 10 --trials 2 --llm ollama --model qwen2.5:3b
"""
import argparse
import csv
import json
import tracemalloc
from datetime import datetime
from pathlib import Path

from agents.llm import make_llm
from agents.llm_agents import LLMBaselineAgent, LLMMemoryAgent, make_plan
from solve import merge, recall

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEFAULT_TASK = "Research three database options, compare their costs, and write a recommendation"


def measured(fn):
    """Run fn() and return (result, peak Python memory in KB)."""
    tracemalloc.start()
    try:
        result = fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, round(peak / 1024, 1)


def one_trial(llm, goal, n, k, budget, mem_path):
    steps, _ = make_plan(llm, goal, n)
    task = {"task_id": mem_path.stem, "goal": goal, "success_criteria": [], "steps": steps}

    def run_baseline():
        sessions = []
        if k:
            sessions.append(LLMBaselineAgent(llm).run(task, stop_after=k))
        sessions.append(LLMBaselineAgent(llm).run(task))
        return merge(sessions)

    def run_memory():
        sessions = []
        if k:
            sessions.append(LLMMemoryAgent(llm, mem_path, budget).run(task, stop_after=k))
        sessions.append(LLMMemoryAgent(llm, mem_path, budget).run(task))
        return merge(sessions)

    rows = {}
    for name, fn in (("baseline", run_baseline), ("memory", run_memory)):
        r, peak_kb = measured(fn)
        found, total = recall(r["final_output"], n - 1)
        rows[name] = {"completed": int(r["final_verified"]), "verified_steps": r["verified_steps"],
                      "llm_calls": r["llm_calls"], "repeated_steps": r["repeated_steps"],
                      "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
                      "peak_prompt_tokens": r["peak_prompt_tokens"],
                      "avg_prompt_tokens": r["avg_prompt_tokens"],
                      "recall_pct": round(100 * found / total, 1),
                      "memory_seconds": r["memory_seconds"], "elapsed_seconds": r["elapsed_seconds"],
                      "peak_python_kb": peak_kb}
    return rows


def mean(values):
    return round(sum(values) / len(values), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 40, 80])
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--interrupt-frac", type=float, default=0.5,
                    help="crash after this fraction of the steps (0 = never)")
    ap.add_argument("--llm", choices=["mock", "anthropic", "ollama"], default="mock")
    ap.add_argument("--model", default=None)
    ap.add_argument("--ctx-budget", type=int, default=2000)
    ap.add_argument("--task", default=DEFAULT_TASK)
    args = ap.parse_args()

    llm = make_llm(args.llm, args.model)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = DATA / "bench_runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    raw, summary = [], []
    for n in args.sizes:
        k = min(int(n * args.interrupt_frac), n - 1)
        print(f"\n--- {n} steps, crash after {k or 'never'}, {args.trials} trial(s) ---", flush=True)
        per_agent = {"baseline": [], "memory": []}
        for t in range(1, args.trials + 1):
            rows = one_trial(llm, args.task, n, k, args.ctx_budget, run_dir / f"{stamp}_{n}_{t}.json")
            for name, row in rows.items():
                per_agent[name].append(row)
                raw.append({"steps": n, "trial": t, "agent": name, **row})
            print(f"  trial {t} done", flush=True)
        for name, rows in per_agent.items():
            agg = {"steps": n, "agent": name, "trials": len(rows),
                   "completion_pct": round(100 * mean([r["completed"] for r in rows]), 1)}
            for key in rows[0]:
                if key != "completed":
                    agg[key] = mean([r[key] for r in rows])
            summary.append(agg)

    cols = [("Steps", "steps"), ("Agent", "agent"), ("Done%", "completion_pct"),
            ("Calls", "llm_calls"), ("Repeat", "repeated_steps"), ("InTok", "input_tokens"),
            ("PeakPrompt", "peak_prompt_tokens"), ("Recall%", "recall_pct"),
            ("MemSec", "memory_seconds"), ("PeakKB", "peak_python_kb")]
    print("\n" + "".join(f"{c:>12}" for c, _ in cols))
    print("-" * 12 * len(cols))
    for row in summary:
        print("".join(f"{str(row[k]):>12}" for _, k in cols))

    out_json = DATA / f"benchmark_{stamp}.json"
    out_csv = DATA / f"benchmark_{stamp}.csv"
    out_json.write_text(json.dumps({"backend": llm.name, "model": getattr(llm, "model", None),
                                    "settings": vars(args), "summary": summary, "trials": raw},
                                   indent=2), encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\nSaved: {out_json}\nSaved: {out_csv}")


if __name__ == "__main__":
    main()