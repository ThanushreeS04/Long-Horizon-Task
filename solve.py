"""Give a task to the baseline agent and/or the memory agent, watch them do the
work, and see the results and statistics.

Examples:
  python solve.py "Plan a one-week study schedule for learning Pandas" --llm ollama --model qwen2.5:3b --steps 6
  python solve.py "Plan a launch" --llm ollama --model qwen2.5:3b --only memory
  python solve.py "..." --llm ollama --model qwen2.5:3b --baseline-window 3 --out my_guide.md
  python solve.py                      (asks you to type the task; uses the mock model)
"""
import argparse
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from agents.guide import build_guide, slugify, write_guide
from agents.llm import make_llm
from agents.adapter import format_ledger, merge_ledgers
from agents.llm_agents import LLMBaselineAgent, LLMMemoryAgent, make_code, make_plan
from agents.mea import MEAAgent

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def merge(sessions):
    """Combine the sessions (before and after the interruption) of one agent."""
    executed = [e for s in sessions for e in s["executed"]]
    series = [t for s in sessions for t in s["prompt_series"]]
    final = next((s["final_output"] for s in reversed(sessions) if s["final_output"]), None)
    final_ok = any(s["final_verified"] for s in sessions)
    outputs = {}
    for s in sessions:
        outputs.update(s.get("step_outputs", {}))
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
            "rejected_attempts": sum(1 for _, ok in executed if not ok),
            "step_outputs": outputs,
            "replans": sum(s.get("replans", 0) for s in sessions),
            "replan_map": {r["failed"]: r["alternative"]
                           for s in sessions for r in s.get("replan_log", [])},
            "integrity_flags": sum(len(s.get("integrity_events", [])) for s in sessions),
            "llm_replans": sum(s.get("llm_replans", 0) for s in sessions),
            "llm_audit_disagreements": sum(s.get("llm_audit_disagreements", 0) for s in sessions),
            "ledger": merge_ledgers([s.get("ledger") for s in sessions]),
            "elapsed_seconds": round(sum(s["elapsed_seconds"] for s in sessions), 4),
            "memory_seconds": round(sum(s["memory_seconds"] for s in sessions), 4)}


def recall(final_output, n_facts, replaced=None):
    """Share of the original step codes present in the final report. If a step was
    replaced by a replan, its replacement's code counts for it."""
    if not final_output:
        return 0, n_facts
    replaced, found = replaced or {}, 0
    for i in range(1, n_facts + 1):
        codes, nxt = [make_code(i)], replaced.get(f"step-{i}")
        while nxt:
            codes.append(make_code(int(nxt.split("-")[1])))
            nxt = replaced.get(nxt)
        found += any(c in final_output for c in codes)
    return found, n_facts


def banner(text):
    print("\n" + "=" * 72 + f"\n{text}\n" + "=" * 72)


def show_work(title, merged, full):
    """Print what the agent actually produced, step by step, then the final report."""
    banner(title)
    outputs = merged["step_outputs"]
    final_text = merged["final_output"]
    for step_id in sorted(outputs, key=lambda s: int(s.split("-")[1])):
        text = outputs[step_id].strip()
        if text == (final_text or "").strip():
            continue
        shown = text if full or len(text) <= 400 else text[:400] + " ..."
        print(f"\n[{step_id}]\n{shown}")
    print("\n[FINAL REPORT]\n" + (final_text.strip() if final_text else "(no final report was produced)"))


def main():
    if hasattr(sys.stdout, "reconfigure"):          # model text may contain symbols the Windows console rejects
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Run a task through the baseline and/or memory agent.")
    ap.add_argument("task", nargs="?", help="the task to solve")
    ap.add_argument("--steps", type=int, default=10, help="total steps including the final report")
    ap.add_argument("--only", choices=["both", "baseline", "memory", "mea", "all"], default="both",
                    help="which agents to run: both = baseline+memory (default), all = baseline+memory+mea, or just one")
    ap.add_argument("--interrupt-at", type=int, default=None,
                    help="simulate a crash after this many steps (0 = never). Default: halfway "
                         "when comparing both agents, never when running one.")
    ap.add_argument("--llm", choices=["mock", "anthropic", "ollama"], default="mock")
    ap.add_argument("--model", default=None)
    ap.add_argument("--ctx-budget", type=int, default=2000,
                    help="max characters of memory context in each prompt")
    ap.add_argument("--full", action="store_true", help="print each step's output in full")
    ap.add_argument("--baseline-window", type=int, default=0,
                    help="give the baseline a sliding window: only its last N step outputs stay in the "
                         "prompt (0 = keep its whole log)")
    ap.add_argument("--out", default=None,
                    help="where to save the how-to guide (default: outputs/<task>-guide.md; "
                         "use a .txt name for a text file)")
    ap.add_argument("--mea-replan", action="store_true",
                    help="let the MEA agent replace steps that keep failing (the baseline and memory agents cannot)")
    ap.add_argument("--llm-replan", choices=["off", "on_reject", "always"], default="off",
                    help="MEA only: let a model rewrite the remaining step descriptions after an "
                         "audited round (on_reject = only after a rejection; max 3 calls)")
    ap.add_argument("--llm-audit", action="store_true",
                    help="MEA only: add an advisory model check of each verified step (never blocks)")
    args = ap.parse_args()

    goal = args.task or input("Enter your task: ").strip()
    if not goal:
        sys.exit("No task given.")
    n = max(3, args.steps)
    k = (n // 2 if args.only in ("both", "all") else 0) if args.interrupt_at is None else args.interrupt_at
    k = min(k, n - 1)

    llm = make_llm(args.llm, args.model)
    print(f"\nTask: {goal}\nBackend: {llm.name}   Steps: {n}   Crash after: {k or 'never'}")
    print("\nPlanning the task ...", flush=True)
    steps, plan_usage = make_plan(llm, goal, n)
    print()
    for s in steps:
        print(f"  {s['id']}: {s['description']}")
    task = {"task_id": "solve-" + datetime.now().strftime("%Y%m%d-%H%M%S%f"), "goal": goal,
            "success_criteria": [], "steps": steps}

    results = {}
    tmp = tempfile.TemporaryDirectory(prefix="lh_solve_", ignore_cleanup_errors=True)
    try:
        if args.only in ("both", "all", "baseline"):
            kept = (f"keeps only its last {args.baseline_window} outputs" if args.baseline_window
                    else "keeps its whole log")
            banner(f"BASELINE AGENT  (no memory: {kept}, saves nothing)")
            sessions = []
            if k:
                print(f"Session 1: working, will crash after {k} steps ", end="", flush=True)
                sessions.append(LLMBaselineAgent(llm, args.baseline_window).run(task, stop_after=k))
                print("\nSession 2: restarted with nothing saved, starts again from step 1 ", end="", flush=True)
            else:
                print("Working ", end="", flush=True)
            sessions.append(LLMBaselineAgent(llm, args.baseline_window).run(task))
            print()
            results["baseline"] = merge(sessions)

        if args.only in ("both", "all", "memory"):
            banner("MEMORY AGENT  (saves progress to JSON, compact context, resumes)")
            mem_path = Path(tmp.name) / f"{task['task_id']}_memory.json"
            sessions = []
            if k:
                print(f"Session 1: working, will crash after {k} steps ", end="", flush=True)
                sessions.append(LLMMemoryAgent(llm, mem_path, args.ctx_budget).run(task, stop_after=k))
                print("\nSession 2: reloads the saved state and continues ", end="", flush=True)
            else:
                print("Working ", end="", flush=True)
            sessions.append(LLMMemoryAgent(llm, mem_path, args.ctx_budget).run(task))
            print()
            results["memory"] = merge(sessions)

        if args.only in ("all", "mea"):
            banner("MEA AGENT  (Manager -> Executor -> Auditor loop: contracts, fresh context, replanning "
                   + ("ON)" if args.mea_replan else "off)"))
            mea_path = Path(tmp.name) / f"{task['task_id']}_mea.json"
            sessions = []
            if k:
                print(f"Session 1: working, will crash after {k} steps ", end="", flush=True)
                sessions.append(MEAAgent(llm, mea_path, replan=args.mea_replan, llm_replan=args.llm_replan,
                                      llm_audit=args.llm_audit).run(task, stop_after=k))
                print("\nSession 2: reloads the saved state and continues ", end="", flush=True)
            else:
                print("Working ", end="", flush=True)
            sessions.append(MEAAgent(llm, mea_path, replan=args.mea_replan, llm_replan=args.llm_replan,
                                      llm_audit=args.llm_audit).run(task))
            print()
            results["mea"] = merge(sessions)
    finally:
        tmp.cleanup()

    for r in results.values():
        r["completed"] = r["final_verified"]
        r["facts_recalled"], r["facts_total"] = recall(r["final_output"], n - 1, r["replan_map"])

    shown = "memory" if "memory" in results else ("mea" if "mea" in results else "baseline")
    show_work(f"WORK PRODUCED BY THE {shown.upper()} AGENT", results[shown], args.full)

    banner("STATISTICS")
    names = {"baseline": "Baseline", "memory": "Memory agent", "mea": "MEA agent"}
    rows = [("Completed", lambda r: "yes" if r["completed"] else "NO"),
            ("Steps verified", lambda r: f"{r['verified_steps']}/{n}"),
            ("LLM calls", lambda r: r["llm_calls"]),
            ("Attempts rejected by auditor", lambda r: r["rejected_attempts"]),
            ("Replans", lambda r: r["replans"]),
            ("LLM replans (applied)", lambda r: r["llm_replans"]),
            ("LLM audit disagreements", lambda r: r["llm_audit_disagreements"]),
            ("Integrity flags", lambda r: r["integrity_flags"]),
            ("Steps redone after crash", lambda r: r["repeated_steps"]),
            ("Input tokens (estimated)" if llm.name != "anthropic" else "Input tokens",
             lambda r: r["input_tokens"]),
            ("Output tokens", lambda r: r["output_tokens"]),
            ("Peak prompt tokens", lambda r: r["peak_prompt_tokens"]),
            ("Avg prompt tokens", lambda r: r["avg_prompt_tokens"]),
            ("Early facts in final report", lambda r: f"{r['facts_recalled']}/{r['facts_total']}"),
            ("Memory overhead (s)", lambda r: r["memory_seconds"]),
            ("Total time (s)", lambda r: r["elapsed_seconds"])]
    cols = list(results)
    print(f"{'Metric':<34}" + "".join(f"{names[c]:>16}" for c in cols))
    print("-" * (34 + 16 * len(cols)))
    for label, fn in rows:
        print(f"{label:<34}" + "".join(f"{str(fn(results[c])):>16}" for c in cols))
    for c in cols:
        if not results[c]["completed"]:
            print(f"\nNote: the {names[c].lower()} did NOT finish. The auditor rejected too many attempts "
                  f"or the final report was missing codes.")

    if results.get("mea", {}).get("ledger"):
        banner("COST BY ROLE  (MEA agent)")
        print(format_ledger(results["mea"]["ledger"]))
        print("Manager and auditor are rule-based here, so they use 0 model tokens; "
              "their seconds are Python work.")

    chosen = results[shown]
    guide_path = Path(args.out) if args.out else ROOT / "outputs" / f"{slugify(goal)}-guide.md"
    stats_line = (f"Run summary: {'completed' if chosen['completed'] else 'NOT completed'}, "
                  f"{chosen['verified_steps']}/{n} steps verified, {chosen['llm_calls']} LLM calls, "
                  f"{chosen['elapsed_seconds']}s ({llm.name}{'/' + llm.model if getattr(llm, 'model', None) else ''}).")
    guide = build_guide(goal, task["steps"], chosen["step_outputs"], names[shown].lower(),
                        getattr(llm, "model", None) or llm.name, chosen["completed"], stats_line)
    write_guide(guide_path, guide)

    out = DATA / f"{task['task_id']}_comparison.json"
    DATA.mkdir(exist_ok=True)
    out.write_text(json.dumps({"task": task, "backend": llm.name, "model": getattr(llm, "model", None),
                               "crash_after": k,
                               "plan_tokens": plan_usage["input_tokens"] + plan_usage["output_tokens"],
                               **results}, indent=2), encoding="utf-8")
    print(f"\nHow-to guide saved to: {guide_path}")
    print(f"Full results saved to: {out}")


if __name__ == "__main__":
    main()