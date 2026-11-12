"""M6/M7 memory baselines: how much of what a step NEEDS does each way of building
its context actually contain?

A finished run is replayed step by step. For each step we know which earlier steps it
depends on (the final report needs all of them) and which code each of those steps
produced. A strategy's recall at a step is the share of those codes present in the
context it would hand the model.

Strategies
  full_log    every earlier fact, in full            (upper bound, grows without limit)
  window_k    only the last `window` facts           (the usual sliding-window baseline)
  bounded     MemoryManager.build_bounded_context    (hard character budget)
  dependency  Manager._evidence: only the facts the step depends on (what MEA really sends)

Also: io_cost() replays the saves to measure write/read time, context-build time,
store growth and peak Python RAM.

Usage:  python memory_baselines.py
        python memory_baselines.py --steps 20 40 80 --budget 700 --window 5
"""
import argparse
import copy
import json
import tempfile
from pathlib import Path

from agents.llm import MockLLM
from agents.llm_agents import make_code, make_plan, step_number
from agents.manager import Manager
from agents.mea import MEAAgent
from agents.memory_manager import MemoryManager
from agents.memory_metrics import MemoryProbe, PeakRAM
from agents.taskstate import _step_of
from agents.workflow import add_dependencies

GOAL = "Research three database options, compare their costs, and write a recommendation"


def run_agent(llm, n, tmpdir):
    """Run an n-step MEA task to completion. Returns (task, agent, report)."""
    steps, _ = make_plan(llm, GOAL, n)
    task = {"task_id": f"mb-{n}", "goal": GOAL, "success_criteria": [], "steps": add_dependencies(steps)}
    agent = MEAAgent(llm, Path(tmpdir) / f"{task['task_id']}.json")
    return task, agent, agent.run(task)


def _state_before(state, index):
    """The saved state as it stood just before step number `index` ran."""
    earlier = lambda sid: sid is not None and step_number({"id": sid}) < index
    partial = copy.deepcopy(state)
    partial["findings"] = [f for f in state.get("findings", []) if earlier(_step_of(f))]
    partial["step_history"] = [h for h in state.get("step_history", []) if earlier(h.get("step_id"))]
    return partial


def _needed_codes(state, step):
    replaced = set(state.get("replaced_step_ids", []))
    return [make_code(step_number({"id": d})) for d in Manager._deps(state, step, replaced)]


def _recall(text, codes):
    return sum(1 for c in codes if c in text) / len(codes) if codes else 1.0


def evaluate(task, state, budget=2000, window=5):
    """Replay the run; per strategy report min/mean/final recall and context size."""
    mem = MemoryManager(Path(tempfile.mkdtemp()) / "unused.json")
    out = {k: {"recalls": [], "sizes": [], "final": None}
           for k in ("full_log", "window_k", "bounded", "dependency")}
    for step in state["plan"]:
        if step["id"] in set(state.get("replaced_step_ids", [])):
            continue
        codes = _needed_codes(state, step)
        before = _state_before(state, step_number(step))
        facts = [str(f) for f in before["findings"]]
        contexts = {
            "full_log": "\n".join(facts),
            "window_k": "\n".join(facts[-window:]) if window > 0 else "",
            "bounded": json.dumps(mem.build_bounded_context(before, budget)),
            "dependency": json.dumps(Manager._evidence(before, Manager._deps(
                before, step, set(before.get("replaced_step_ids", []))))),
        }
        for name, text in contexts.items():
            r = _recall(text, codes)
            if codes:
                out[name]["recalls"].append(r)
            out[name]["sizes"].append(len(text))
            if step.get("final"):
                out[name]["final"] = r
    result = {}
    for name, d in out.items():
        rec = d["recalls"] or [1.0]
        result[name] = {"min_recall": round(min(rec), 3), "mean_recall": round(sum(rec) / len(rec), 3),
                        "final_recall": round(d["final"] if d["final"] is not None else rec[-1], 3),
                        "max_context_chars": max(d["sizes"]),
                        "steps_over_budget": sum(1 for s in d["sizes"] if s > budget)}
    return result


def io_cost(task, state, budget, tmpdir):
    """Replay the saves of a growing state and time them."""
    probe = MemoryProbe(MemoryManager(Path(tmpdir) / "io_probe.json"), budget)
    growth = []
    with PeakRAM() as ram:
        for step in state["plan"]:
            partial = _state_before(state, step_number(step) + 1)
            partial["task_id"] = "io-probe"
            probe.save(partial)
            probe.load("io-probe")
            ctx = probe.context(partial)
            growth.append((step["id"], len(json.dumps(partial, default=str)), len(json.dumps(ctx))))
    rep = probe.report()
    return {"write_ms_avg": rep["write_ms_avg"], "read_ms_avg": rep["read_ms_avg"],
            "context_build_ms_avg": rep["context_build_ms_avg"], "peak_ram_kib": ram.peak_kib,
            "state_chars_final": growth[-1][1], "context_chars_final": growth[-1][2], "growth": growth}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", default=[20, 40, 80])
    ap.add_argument("--budget", type=int, default=700)
    ap.add_argument("--window", type=int, default=5)
    args = ap.parse_args()
    print(f"budget={args.budget} chars, window={args.window}   (recall = share of needed codes in context)\n")
    head = f"{'Steps':>6} {'Strategy':<12} {'MinRecall':>10} {'FinalRecall':>12} {'MaxCtxChars':>12} {'OverBudget':>11}"
    print(head + "\n" + "-" * len(head))
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for n in args.steps:
            task, agent, rep = run_agent(MockLLM(), n, tmp)
            state = agent.memory.get_task(task["task_id"])
            for name, r in evaluate(task, state, args.budget, args.window).items():
                print(f"{n:>6} {name:<12} {r['min_recall']:>10} {r['final_recall']:>12} "
                      f"{r['max_context_chars']:>12} {r['steps_over_budget']:>11}")
            io = io_cost(task, state, args.budget, tmp)
            print(f"{'':>6} io: write {io['write_ms_avg']}ms  read {io['read_ms_avg']}ms  "
                  f"state {io['state_chars_final']} chars  peak RAM {io['peak_ram_kib']} KiB\n")


if __name__ == "__main__":
    main()