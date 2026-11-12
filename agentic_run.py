"""Plan-and-Act + Manage-Execute-Audit, all features on.

  * Planner turns the query into high-level steps (with chain-of-thought)
  * Model-based Manager (writes contracts, may ASK the human)
  * Human-in-the-loop ASK: answer now (--ask cli) or later (--resume STATE --answer "...")
  * Executor with chain-of-thought and garbage collection; deliverables are written to a workspace
  * Environment-grounded Auditor: re-reads the file, checks its SHA-256, diffs the workspace, then a model judge
  * Dynamic replanning after every audited step (keep / rewrite / drop / add steps)
  * Optional LoRA adapters for planner / executor, and trajectory export for synthetic_pipeline.py
  * Files from the PREVIOUS run are deleted automatically when a new run starts

Examples
  python agentic_run.py "Plan a product launch" --steps 6                       (offline mock model)
  python agentic_run.py "Plan a product launch" --ask cli --ask-at 2            (you answer step 2's question)
  python agentic_run.py "Plan a budget" --ask off --ask-at 2                    (pauses; prints the resume command)
  python agentic_run.py --resume runs/<id>_state.json --answer "USD 5,000"      (continue the paused run)
  python agentic_run.py "Plan a calculator" --llm ollama --model qwen2.5:3b --small --steps 5   (small local model)
  python agentic_run.py "..." --planner-adapter adapters/planner --base-model meta-llama/Llama-3.1-8B-Instruct
  python -m agents.cleanup                                                      (delete last run's files now)
"""
import argparse
import contextlib
import csv
import io
import json
import sys
from datetime import datetime
from pathlib import Path

from agents.adapter import DEFAULT_BUDGETS, AgentAdapter, ModelRouter
from agents.cleanup import RUNS, RunFiles
from agents.cot_executor import CoTExecutor
from agents.env_auditor import EnvironmentAuditor
from agents.harness import ExternalCLIHarness, ToolLoopHarness
from agents.hl_planner import HighLevelPlanner
from agents.human import make_ask_fn
from agents.llm import make_llm
from agents.llm_auditor import LLMAuditor
from agents.llm_manager import LLMManager
from agents.manager import Manager
from agents.mea import Auditor, MEAAgent
from agents.memory_manager import MemoryManager
from agents.natural import NaturalRules
from agents.observation_gc import ObservationGC
from agents.state_replanner import StateReplanner
from agents.trajectory import export_trajectory
from agents.workspace import Workspace

ROOT = Path(__file__).resolve().parent


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", nargs="?", help="the task (asked for if omitted; not needed with --resume)")
    ap.add_argument("--steps", type=int, default=6, help="plan length including the final report")
    ap.add_argument("--llm", default="mock", choices=["mock", "ollama", "anthropic"])
    ap.add_argument("--model", help="model name for every role")
    ap.add_argument("--manager-model", help="different model for manager, planner and replanner")
    ap.add_argument("--executor-model", help="different model for the executor")
    ap.add_argument("--auditor-model", help="different model for the auditor")
    ap.add_argument("--planner-adapter", help="LoRA adapter directory for the planner (see train_lora.py)")
    ap.add_argument("--executor-adapter", help="LoRA adapter directory for the executor")
    ap.add_argument("--base-model", default="meta-llama/Llama-3.3-70B-Instruct",
                    help="base model the adapters were trained on")
    ap.add_argument("--ask", default="auto", choices=["auto", "cli", "off"],
                    help="when the manager would ask a person: auto = decide alone and state the assumption "
                         "(default, no typing needed), cli = ask on the terminal, off = pause with the state "
                         "saved, resume with --resume/--answer")
    ap.add_argument("--ask-at", type=int, metavar="N", help="force the manager to ask about step N (demo)")
    ap.add_argument("--ask-question", default="Any preferences or constraints for this step "
                                              "(audience, tools, budget, deadline)?")
    ap.add_argument("--resume", metavar="STATE", help="state file of a paused run (runs/<id>_state.json)")
    ap.add_argument("--answer", help="answer to the paused run's question (use with --resume)")
    ap.add_argument("--legacy-codes", action="store_true",
                    help="old test-harness checks: every reply must contain a FACT-n line and a K-code. "
                         "Default is natural mode: the model just answers")
    ap.add_argument("--quiet", action="store_true", help="print only the final answer")
    ap.add_argument("--small", action="store_true",
                    help="preset for small local models such as qwen2.5:3b: no chain-of-thought, rule-based "
                         "manager and auditor (file and content checks still run), replan only after a "
                         "rejection. About 1 model call per step instead of 4")
    ap.add_argument("--replan-mode", default="always", choices=["always", "on_reject"],
                    help="model replanning after every step, or only after a rejected step")
    ap.add_argument("--harness", default="none", choices=["none", "tools", "cli"],
                    help="how the executor works: none = one model call (default), tools = built-in loop that "
                         "can list/read/write files in the workspace, cli = hand each step to --harness-cmd")
    ap.add_argument("--allow-run", action="store_true",
                    help="with --harness tools: let the model run .py files it wrote in the workspace. "
                         "NOT a sandbox: the code runs with your permissions")
    ap.add_argument("--harness-turns", type=int, default=6, help="max tool turns per step (tools harness)")
    ap.add_argument("--harness-cmd", help='external agent command, prompt on stdin, e.g. "claude -p"')
    ap.add_argument("--no-workspace", action="store_true",
                    help="do not write artifacts; audit the executor's text only (old behaviour)")
    ap.add_argument("--no-cot", action="store_true", help="turn off chain-of-thought")
    ap.add_argument("--rule-manager", action="store_true", help="use the rule-based manager only")
    ap.add_argument("--rule-auditor", action="store_true", help="rules + file checks only, no model judge")
    ap.add_argument("--no-replan", action="store_true", help="turn off replanning after each step")
    ap.add_argument("--max-replans", type=int, default=50, help="cap on model replans per run")
    ap.add_argument("--gc-chars", type=int, default=1500, help="facts shown to one executor round (chars)")
    ap.add_argument("--export-trajectory", action="store_true",
                    help="append this run to data/trajectories/trajectories.jsonl for synthetic_pipeline.py")
    ap.add_argument("--keep-files", action="store_true",
                    help="do NOT delete the previous run's json/csv files at start")
    ap.add_argument("--clean-now", action="store_true", help="delete this run's json/csv files when it ends")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.small:
        args.no_cot = args.rule_manager = args.rule_auditor = True
        args.replan_mode = "on_reject"
    if not args.quiet:
        return _run(args)[0]
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code, final = _run(args)
    print(final.strip() if code == 0 and final else buffer.getvalue().strip())   # answer only; details on failure
    return code


def _run(args):
    files = RunFiles()
    if not args.keep_files:                                   # automatic cleanup of the previous run
        gone = files.purge_previous()
        print(f"Cleaned up {len(gone)} file(s) from the previous run." if gone
              else "No files from a previous run to clean up.")

    cot, ask_fn = not args.no_cot, make_ask_fn(args.ask)
    natural = not args.legacy_codes
    role_llm = lambda m: make_llm(args.llm, m) if m else None
    llm = make_llm(args.llm, args.model)
    planner_llm = role_llm(args.manager_model)
    executor_llm = role_llm(args.executor_model)
    if args.planner_adapter or args.executor_adapter:
        from agents.hf_lora import HuggingFaceLoRABackend
        if args.planner_adapter:
            planner_llm = HuggingFaceLoRABackend(args.planner_adapter, args.base_model)
        if args.executor_adapter:
            executor_llm = HuggingFaceLoRABackend(args.executor_adapter, args.base_model)
    router = ModelRouter(llm, manager=role_llm(args.manager_model), planner=planner_llm,
                         replanner=role_llm(args.manager_model), executor=executor_llm,
                         auditor=role_llm(args.auditor_model))
    budgets = {r: dict(b) for r, b in DEFAULT_BUDGETS.items()}
    budgets["planner"] = budgets["replanner"] = {"max_tokens": 900, "seconds": 300}
    adapter = AgentAdapter(router, budgets)

    run_id = "run-" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    RUNS.mkdir(exist_ok=True)
    plan = None
    if args.resume:                                           # continue a paused (ASK) run
        state_path = Path(args.resume).resolve()
        mem = MemoryManager(state_path)
        saved = mem._read_all()
        if not saved:
            sys.exit(f"{state_path} holds no saved task.")
        task_id = next(iter(saved))
        st = saved[task_id]
        goal, steps = st["goal"], st["plan"]
        if args.answer:
            sid = st.get("question_step")
            if not sid:
                sys.exit("This run is not waiting for an answer.")
            st.setdefault("answers", {})[sid] = args.answer.strip()
            st["question"] = None
            mem.save_task(st)
            print(f"Answer saved for {sid}; resuming '{goal}'.")
        files.register(state_path)                            # tracked again; deleted at the next run's start
    else:
        goal = args.task or input("Enter your task: ").strip()
        if not goal:
            sys.exit("No task given.")
        n = max(3, args.steps)
        print(f"\nTask: {goal}\nBackend: {llm.name}  Steps: {n}  CoT: {'on' if cot else 'off'}\n\nPlanning ...",
              flush=True)
        plan = HighLevelPlanner(adapter, cot=cot and not args.planner_adapter,
                                trained_prompt=bool(args.planner_adapter)).plan(goal, n)
        steps = plan["steps"]
        if args.ask_at:
            if not 1 <= args.ask_at < len(steps):
                sys.exit(f"--ask-at must be between 1 and {len(steps) - 1}")
            steps[args.ask_at - 1].update(needs_input=True, question=args.ask_question)
        print(f"(planner source: {plan['source']})")
        task_id = run_id
        state_path = files.register(RUNS / f"{task_id}_state.json")
    for s in steps:
        print(f"  {s['id']}: {s['description']}")
    report_path = files.register(RUNS / f"{run_id}_report.json")
    rounds_path = files.register(RUNS / f"{run_id}_rounds.csv")
    task = {"task_id": task_id, "goal": goal, "success_criteria": [], "steps": steps}

    workspace = None if args.no_workspace else Workspace(ROOT / "workspace" / task_id)
    gc = ObservationGC(max_chars=args.gc_chars)
    executor = CoTExecutor(adapter.for_role("executor"), cot=cot, gc=gc, workspace=workspace, natural=natural)
    if args.harness != "none":
        if workspace is None:
            sys.exit("--harness needs the workspace (remove --no-workspace).")
        if args.harness == "cli":
            if not args.harness_cmd:
                sys.exit('--harness cli needs --harness-cmd, for example --harness-cmd "claude -p"')
            executor.harness = ExternalCLIHarness(args.harness_cmd, workspace)
        else:
            executor.harness = ToolLoopHarness(executor.llm, workspace, args.harness_turns, args.allow_run)
    manager = (Manager(natural=natural) if args.rule_manager
               else LLMManager(adapter, allow_ask=ask_fn is not None, cot=cot, natural=natural))
    if workspace is not None:
        auditor = EnvironmentAuditor(adapter, workspace, use_model=not args.rule_auditor, cot=cot)
    else:
        auditor = Auditor() if args.rule_auditor else LLMAuditor(adapter, cot=cot)
    if natural:
        auditor.rules = NaturalRules()          # judge substance, not labels and codes
    replanner = None if args.no_replan else StateReplanner(adapter, args.replan_mode, args.max_replans, cot=cot)
    agent = MEAAgent(llm, state_path, replan=True, adapter=adapter, manager=manager, auditor=auditor,
                     executor=executor, state_replanner=replanner, ask_fn=ask_fn)

    print("\nWorking ...", flush=True)
    rep = agent.run(task)
    state = agent.memory.get_task(task_id) or {}

    paused = rep["stopped_because"] == "ask"
    print(f"\n{'=' * 70}\nRESULT: {'COMPLETED' if rep['completed'] else 'PAUSED (waiting for you)' if paused else 'NOT COMPLETED'} "
          f"(stopped because: {rep['stopped_because']}) after {rep['rounds']} executor rounds")
    print(f"Human answers : {state.get('answers') or 'none'}")
    llm_rp = [e for e in state.get("llm_replans", []) if e["applied"]]
    print(f"Model replans : {len(llm_rp)} applied of {len(state.get('llm_replans', []))} calls "
          f"(rewritten {sum(len(e['changed']) for e in llm_rp)}, "
          f"dropped {sum(len(e['dropped']) for e in llm_rp)}, added {sum(len(e['added']) for e in llm_rp)})")
    alog = getattr(auditor, "log", [])
    print(f"Model auditor : {len(alog)} checks, {sum(1 for a in alog if a.get('blocked'))} rejection(s)")
    print(f"Integrity     : {rep['integrity_events'] or 'all clean'}")
    print(f"Garbage coll. : {gc.summary()}")
    if args.harness != "none":
        calls = [c for r in executor.reasoning_log for c in r.get("tool_calls", [])]
        print(f"Harness       : {args.harness}, {len(calls)} tool call(s) {calls[:8]}")
    if workspace is not None:
        print(f"Workspace     : {workspace.root}  ({len(workspace.snapshot())} artifact file(s))")
    print(f"\n{'Role':<11}{'calls':>7}{'in tok':>9}{'out tok':>9}{'seconds':>9}")
    for role, r in rep["ledger"].items():
        print(f"{role:<11}{r['calls']:>7}{r['input_tokens']:>9}{r['output_tokens']:>9}{r['seconds']:>9.2f}")
    if rep.get("final_output"):
        print(f"\nFINAL OUTPUT\n{rep['final_output'][:1500]}")

    report = {"run_id": run_id, "task_id": task_id, "goal": goal, "backend": llm.name,
              "planner": {"source": plan["source"], "reasoning": plan["reasoning"][:1000]} if plan else "resumed",
              "completed": rep["completed"], "stopped_because": rep["stopped_because"],
              "rounds": rep["rounds"], "decisions": rep["decisions"], "answers": state.get("answers", {}),
              "manager_log": getattr(manager, "log", []), "auditor_log": alog,
              "integrity_events": rep["integrity_events"], "replans": state.get("llm_replans", []),
              "executor_reasoning": executor.reasoning_log, "gc": {"summary": gc.summary(), "log": gc.log},
              "ledger": rep["ledger"], "final_output": rep.get("final_output")}
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    with open(rounds_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["round", "decision", "step", "reason"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rep["decisions"])

    if args.export_trajectory and not paused:          # a paused run is not a finished trajectory
        out = ROOT / "data" / "trajectories" / "trajectories.jsonl"
        k = export_trajectory(goal, state.get("plan", steps), rep["step_outputs"], state.get("errors", []),
                              rep["completed"], out)
        print(f"\nTrajectory: {k} step(s) appended to {out}" if k else "\nTrajectory: nothing verified to export.")

    if paused:                                                # keep the state so the run can be resumed
        files.unregister(state_path)
        print(f"\nThe run is paused and its state is kept. Question: {state.get('question')}\n"
              f"Resume with:\n  python agentic_run.py --resume \"{state_path}\" --answer \"<your answer>\"")
    elif args.clean_now:
        print(f"\nDeleted {len(files.purge_current())} file(s) from this run (--clean-now).")
    else:
        print("\nFiles from this run (deleted automatically when the next run starts):")
        for p in (state_path, report_path, rounds_path):
            print(f"  {p}")
    return (0 if rep["completed"] else (3 if paused else 1)), rep.get("final_output")


if __name__ == "__main__":
    sys.exit(main())
