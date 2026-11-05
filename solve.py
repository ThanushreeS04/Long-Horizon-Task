"""Give any task to a baseline agent and a memory agent and compare them.

Usage:
  python solve.py "Your task here" --steps 12 --interrupt-at 6 --llm mock
  python solve.py                      (asks you to type the task)
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from agents.llm import make_llm
from agents.llm_agents import (LLMBaselineAgent, LLMMemoryAgent, make_code, make_plan)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def merge(sessions):
    """Combine the sessions (before and after the interruption) of one agent."""
    executed = [e for s in sessions for e in s["executed"]]
    series = [t for s in sessions for t in s["prompt_series"]]
    final = next((s["final_output"] for s in reversed(sessions) if s["final_output"]), None)
    final_ok = any(s["final_verified"] for s in sessions)
    verified_ever, repeated = set(), 0
    for step_id, ok in executed:
        if step_id in verified_ever:
            repeated += 1            # work redone on a step that was already verified
        if ok:
            verified_ever.add(step_id)
    return {"llm_calls": sum(s["llm_calls"] for s in sessions),
            "input_tokens": sum(s["input_tokens"] for s in sessions),
            "output_tokens": sum(s["output_tokens"] for s in sessions),
            "peak_prompt_tokens": max(series) if series else 0,
            "avg_prompt_tokens": round(sum(series) / len(series), 1) if series else 0,
            "prompt_series": series, "verified_steps": len(verified_ever),
            "repeated_steps": repeated, "final_output": final, "final_verified": final_ok,
            "elapsed_seconds": round(sum(s["elapsed_seconds"] for s in sessions), 4),
            "memory_seconds": round(sum(s["memory_seconds"] for s in sessions), 4)}


def recall(final_output, n_facts):
    if not final_output:
        return 0, n_facts
    found = sum(make_code(i) in final_output for i in range(1, n_facts + 1))
    return found, n_facts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?", help="the task to solve")
    ap.add_argument("--steps", type=int, default=10, help="total steps incl. final report")
    ap.add_argument("--interrupt-at", type=int, default=None,
                    help="crash after this many steps (0 = no interruption)")
    ap.add_argument("--llm", choices=["mock", "anthropic", "ollama"], default="mock")
    ap.add_argument("--model", default=None)
    ap.add_argument("--ctx-budget", type=int, default=2000,
                    help="max characters of memory context in each prompt")
    args = ap.parse_args()

    goal = args.task or input("Enter your task: ").strip()
    if not goal:
        sys.exit("No task given.")
    n = max(3, args.steps)
    k = n // 2 if args.interrupt_at is None else args.interrupt_at
    k = min(k, n - 1)

    llm = make_llm(args.llm, args.model)
    steps, plan_usage = make_plan(llm, goal, n)
    task = {"task_id": "solve-" + datetime.now().strftime("%Y%m%d-%H%M%S%f"), "goal": goal,
            "success_criteria": [], "steps": steps}
    print(f"\nTask: {goal}\nBackend: {llm.name}   Steps: {n}   Interrupt after: {k or 'never'}\n")
    for s in steps:
        print(f"  {s['id']}: {s['description']}")

    # Baseline: after the crash it has nothing saved, so it starts over.
    base_sessions = []
    if k:
        base_sessions.append(LLMBaselineAgent(llm).run(task, stop_after=k))
    base_sessions.append(LLMBaselineAgent(llm).run(task))

    # Memory agent: after the crash it reloads the JSON state and continues.
    DATA.mkdir(exist_ok=True)
    mem_path = DATA / f"{task['task_id']}_memory.json"
    mem_sessions = []
    if k:
        mem_sessions.append(LLMMemoryAgent(llm, mem_path, args.ctx_budget).run(task, stop_after=k))
    mem_sessions.append(LLMMemoryAgent(llm, mem_path, args.ctx_budget).run(task))

    base, mem = merge(base_sessions), merge(mem_sessions)
    for r in (base, mem):
        r["completed"] = r["final_verified"]
        r["facts_recalled"], r["facts_total"] = recall(r["final_output"], n - 1)

    rows = [("Completed", "completed"), ("Verified steps", "verified_steps"),
            ("LLM calls", "llm_calls"), ("Repeated steps after crash", "repeated_steps"),
            ("Total input tokens", "input_tokens"), ("Total output tokens", "output_tokens"),
            ("Peak prompt tokens", "peak_prompt_tokens"), ("Avg prompt tokens", "avg_prompt_tokens"),
            ("Memory overhead (s)", "memory_seconds"), ("Elapsed (s)", "elapsed_seconds")]
    print(f"\n{'Metric':<30}{'Baseline':>14}{'Memory agent':>16}")
    print("-" * 60)
    for label, key in rows:
        print(f"{label:<30}{str(base[key]):>14}{str(mem[key]):>16}")
    rb = f"{base['facts_recalled']}/{base['facts_total']}"
    rm = f"{mem['facts_recalled']}/{mem['facts_total']}"
    print(f"{'Early facts in final report':<30}{rb:>14}{rm:>16}")

    out = DATA / f"{task['task_id']}_comparison.json"
    out.write_text(json.dumps({"task": task, "backend": llm.name, "interrupt_after": k,
                               "plan_tokens": plan_usage["input_tokens"] + plan_usage["output_tokens"],
                               "baseline": base, "memory_enabled": mem}, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}\nMemory file: {mem_path}")


if __name__ == "__main__":
    main()