"""Step 5 - dependencies and replanning benchmark.

Usage:
  python workflow_bench.py
  python workflow_bench.py --sizes 10 20 --trials 5 --dead-steps 1 --crashes 1
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
from agents.llm_agents import make_plan, make_code
from agents.workflow import (WorkflowBaselineAgent, WorkflowMemoryAgent,
                             add_dependencies)
from recovery_bench import FaultyLLM

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEFAULT_TASK = "Research three database options, compare their costs, and write a recommendation"
PERMANENT = 99


def pick_faults(n, dead, rate, rng):
    """`dead` steps always fail; a fraction `rate` of the others fail 1-2 times."""
    work = list(range(1, n))
    pool = work[: max(1, (2 * len(work)) // 3)]
    dead_steps = rng.sample(pool, min(dead, len(pool)))
    faults = {i: PERMANENT for i in dead_steps}
    others = [i for i in work if i not in faults]
    for i in rng.sample(others, round(len(others) * rate)):
        faults[i] = rng.choice([1, 2])
    return faults


def run_agent(kind, llm, task, crash_sizes, budget, replan, mem_path):
    sessions = []

    def make():
        if kind == "baseline":
            return WorkflowBaselineAgent(llm, replan=replan)
        return WorkflowMemoryAgent(llm, mem_path, budget, replan=replan)

    for k in crash_sizes:
        sessions.append(make().run(copy.deepcopy(task), stop_after=k))
    sessions.append(make().run(copy.deepcopy(task)))
    return sessions


def summarise(sessions, n):
    executed = [e for s in sessions for e in s["executed"]]
    seen, repeated, failed = set(), 0, 0
    for step_id, ok in executed:
        if ok:
            repeated += step_id in seen
            seen.add(step_id)
        else:
            failed += 1
    final_out = next((s["final_output"] for s in sessions if s["final_output"]), None)
    work_ids = [int(x.split("-")[1]) for x in seen]
    found = sum(1 for i in work_ids if final_out and make_code(i) in final_out)
    expected = len([i for i in work_ids if i != n])
    series = [t for s in sessions for t in s["prompt_series"]]
    last = sessions[-1]
    return {
        "completed": int(last["completed"]),
        "unique_verified": len(seen),
        "llm_calls": sum(s["llm_calls"] for s in sessions),
        "repeated_steps": repeated,
        "rejected_attempts": failed,
        "replan_events": sum(s["replans"] for s in sessions),
        "blocked_steps": len(last["blocked"]),
        "input_tokens": sum(s["input_tokens"] for s in sessions),
        "peak_prompt_tokens": max(series) if series else 0,
        "recall_pct": round(100 * found / expected, 1) if expected and final_out else 0.0,
    }


def mean(values):
    return round(sum(values) / len(values), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 40])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--dead-steps", type=int, default=1, help="steps that always fail")
    ap.add_argument("--fault-rate", type=float, default=0.1, help="transient faults on other steps")
    ap.add_argument("--crashes", type=int, default=1)
    ap.add_argument("--ctx-budget", type=int, default=4000)
    ap.add_argument("--llm", choices=["mock", "anthropic", "ollama"], default="mock")
    ap.add_argument("--model", default=None)
    ap.add_argument("--task", default=DEFAULT_TASK)
    args = ap.parse_args()

    base_llm = make_llm(args.llm, args.model)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tmp = tempfile.TemporaryDirectory(prefix="lh_", ignore_cleanup_errors=True)
    run_dir = Path(tmp.name)

    raw, summary = [], []
    for n in args.sizes:
        steps, _ = make_plan(base_llm, args.task, n)
        steps = add_dependencies(steps)
        crash_every = max(1, (n - 1) // (args.crashes + 1)) if args.crashes else 0
        crash_sizes = [crash_every] * args.crashes
        for replan in (True, False):
            tag = "replan ON" if replan else "replan OFF"
            print(f"\n--- {n} steps | {tag} | {args.dead_steps} dead step(s) | "
                  f"crash every {crash_every or '-'} | {args.trials} trial(s) ---", flush=True)
            per_agent = {"baseline": [], "memory": []}
            for t in range(1, args.trials + 1):
                faults = pick_faults(n, args.dead_steps, args.fault_rate,
                                     random.Random(1000 * n + t))
                task = {"task_id": f"wf_{n}_{t}", "goal": args.task,
                        "success_criteria": [], "steps": steps}
                for kind in ("baseline", "memory"):
                    llm = FaultyLLM(base_llm, faults)
                    mem_path = run_dir / f"{stamp}_{n}_{'on' if replan else 'off'}_{t}.json"
                    sessions = run_agent(kind, llm, task, crash_sizes,
                                         args.ctx_budget, replan, mem_path)
                    row = summarise(sessions, n)
                    per_agent[kind].append(row)
                    raw.append({"steps": n, "replan": replan, "trial": t, "agent": kind,
                                "faults": faults, **row})
            for kind, rows in per_agent.items():
                agg = {"steps": n, "replan": tag, "agent": kind, "trials": len(rows),
                       "completion_pct": round(100 * mean([r["completed"] for r in rows]), 1)}
                for key in rows[0]:
                    if key != "completed":
                        agg[key] = mean([r[key] for r in rows])
                summary.append(agg)

    cols = [("Steps", "steps", 7), ("Replan", "replan", 12), ("Agent", "agent", 10),
            ("Done%", "completion_pct", 8), ("Verified", "unique_verified", 10),
            ("Calls", "llm_calls", 8), ("Repeat", "repeated_steps", 8),
            ("Rejected", "rejected_attempts", 10), ("Replans", "replan_events", 9),
            ("Blocked", "blocked_steps", 9), ("Recall%", "recall_pct", 9)]
    print("\n" + "".join(f"{c:>{w}}" for c, _, w in cols))
    print("-" * sum(w for _, _, w in cols))
    for row in summary:
        print("".join(f"{str(row[k]):>{w}}" for _, k, w in cols))

    out_json = DATA / f"workflow_{stamp}.json"
    out_csv = DATA / f"workflow_{stamp}.csv"
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