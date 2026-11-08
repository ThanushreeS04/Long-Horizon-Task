"""Failure-injection benchmark: baseline vs memory agent vs MEA agent.

Uses the same faults as workflow_bench.py: some steps always fail ("dead"), some
fail 1-2 times, and the process crashes part-way. Same plan, same faults, same
crash points for every agent.

Usage:
  python mea_bench.py
  python mea_bench.py --sizes 10 20 40 --trials 5 --dead-steps 2 --crashes 2
"""
import argparse
import copy
import csv
import json
import random
import tempfile
from datetime import datetime
from pathlib import Path

from agents.llm import make_llm
from agents.llm_agents import make_plan
from agents.mea import MEAAgent
from agents.workflow import WorkflowBaselineAgent, WorkflowMemoryAgent, add_dependencies
from recovery_bench import FaultyLLM
from workflow_bench import mean, pick_faults, summarise

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEFAULT_TASK = "Research three database options, compare their costs, and write a recommendation"

AGENTS = {
    "baseline": lambda llm, p, b: WorkflowBaselineAgent(llm, replan=True),
    "memory+replan": lambda llm, p, b: WorkflowMemoryAgent(llm, p, b, replan=True),
    "mea no-replan": lambda llm, p, b: MEAAgent(llm, p, replan=False),
    "mea+replan": lambda llm, p, b: MEAAgent(llm, p, replan=True),
}


def run_agent(make, task, crash_sizes):
    sessions = []
    for k in crash_sizes:
        sessions.append(make().run(copy.deepcopy(task), stop_after=k))
    sessions.append(make().run(copy.deepcopy(task)))
    return sessions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 40])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--dead-steps", type=int, default=1, help="steps that always fail")
    ap.add_argument("--fault-rate", type=float, default=0.1, help="transient faults on other steps")
    ap.add_argument("--crashes", type=int, default=1)
    ap.add_argument("--ctx-budget", type=int, default=4000, help="memory agent only")
    ap.add_argument("--llm", choices=["mock", "anthropic", "ollama"], default="mock")
    ap.add_argument("--model", default=None)
    ap.add_argument("--task", default=DEFAULT_TASK)
    args = ap.parse_args()

    base_llm = make_llm(args.llm, args.model)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    DATA.mkdir(exist_ok=True)
    tmp = tempfile.TemporaryDirectory(prefix="lh_mea_", ignore_cleanup_errors=True)
    run_dir = Path(tmp.name)

    raw, summary = [], []
    try:
        for n in args.sizes:
            steps, _ = make_plan(base_llm, args.task, n)
            steps = add_dependencies(steps)
            crash_every = max(1, (n - 1) // (args.crashes + 1)) if args.crashes else 0
            crash_sizes = [crash_every] * args.crashes
            print(f"\n--- {n} steps | {args.dead_steps} dead step(s) | fault rate {args.fault_rate} | "
                  f"crash every {crash_every or '-'} | {args.trials} trial(s) ---", flush=True)
            per_agent = {name: [] for name in AGENTS}
            for t in range(1, args.trials + 1):
                faults = pick_faults(n, args.dead_steps, args.fault_rate,
                                     random.Random(1000 * n + t))
                task = {"task_id": f"mb_{n}_{t}", "goal": args.task,
                        "success_criteria": [], "steps": steps}
                for name, factory in AGENTS.items():
                    llm = FaultyLLM(base_llm, faults)
                    path = run_dir / f"{stamp}_{n}_{t}_{name.replace('+', '_').replace(' ', '_')}.json"
                    sessions = run_agent(lambda: factory(llm, path, args.ctx_budget), task, crash_sizes)
                    row = summarise(sessions, n)
                    per_agent[name].append(row)
                    raw.append({"steps": n, "trial": t, "agent": name, "faults": faults, **row})
            for name, rows in per_agent.items():
                agg = {"steps": n, "agent": name, "trials": len(rows),
                       "completion_pct": round(100 * mean([r["completed"] for r in rows]), 1)}
                for key in rows[0]:
                    if key != "completed":
                        agg[key] = mean([r[key] for r in rows])
                summary.append(agg)
    finally:
        tmp.cleanup()

    cols = [("Steps", "steps", 7), ("Agent", "agent", 16), ("Done%", "completion_pct", 8),
            ("Verified", "unique_verified", 10), ("Calls", "llm_calls", 8),
            ("Repeat", "repeated_steps", 8), ("Rejected", "rejected_attempts", 10),
            ("Replans", "replan_events", 9), ("Blocked", "blocked_steps", 9),
            ("InTok", "input_tokens", 10), ("Recall%", "recall_pct", 9)]
    print("\n" + "".join(f"{c:>{w}}" for c, _, w in cols))
    print("-" * sum(w for _, _, w in cols))
    for row in summary:
        print("".join(f"{str(row[k]):>{w}}" for _, k, w in cols))

    out_json = DATA / f"meabench_{stamp}.json"
    out_csv = DATA / f"meabench_{stamp}.csv"
    out_json.write_text(json.dumps({"backend": base_llm.name, "settings": vars(args),
                                    "summary": summary, "trials": raw}, indent=2), encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\nSaved: {out_json}\nSaved: {out_csv}")


if __name__ == "__main__":
    main()