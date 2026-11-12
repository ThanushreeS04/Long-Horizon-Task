"""M9/M10 ablation: MEA vs MEA + durable recovery under HARD crashes.

Scenarios : midround       process killed mid-round, state file intact
            midround+torn  killed mid-round AND the saved JSON is truncated
Arms      : mea            plain MEAAgent
            mea+durable    make_durable(MEAAgent)
Measures  : completed, crash_detected, store_recovered, repeated_steps, extra_errors

Usage:  python durable_bench.py
        python durable_bench.py --sizes 10 20 40 --trials 5
"""
import argparse
import copy
import json
import tempfile
from datetime import datetime
from pathlib import Path

from agents.durable import make_durable
from agents.llm import MockLLM
from agents.llm_agents import make_plan
from agents.mea import Executor, MEAAgent
from agents.memory_manager import MemoryManager
from agents.workflow import add_dependencies

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
GOAL = "Research three database options, compare their costs, and write a recommendation"


class Dies(Executor):
    """Raises on its k-th call: the process is killed in the middle of a round."""

    def __init__(self, llm, k):
        super().__init__(llm)
        self.k, self.calls = k, 0

    def run(self, goal, contract, step, stats):
        self.calls += 1
        if self.calls == self.k:
            raise RuntimeError("process killed")
        return super().run(goal, contract, step, stats)


def build(arm, path, k=None):
    llm = MockLLM()
    agent = MEAAgent(llm, path, executor=Dies(llm, k) if k else None)
    return make_durable(agent) if arm == "mea+durable" else agent


def trial(arm, scenario, task, kill_at, workdir):
    path = workdir / f"{arm.replace('+', '_')}_{scenario.replace('+', '_')}.json"
    try:
        build(arm, path, kill_at).run(copy.deepcopy(task))
    except RuntimeError:
        pass
    state = MemoryManager(path).get_task(task["task_id"])
    verified_before = {s["id"] for s in state["plan"]} - set(state["pending_step_ids"])
    errors_before = len(state["errors"])
    if scenario == "midround+torn":
        raw = path.read_text(encoding="utf-8")
        path.write_text(raw[: len(raw) // 2], encoding="utf-8")
    row = {"completed": False, "crash_detected": False, "store_recovered": False,
           "repeated_steps": 0, "extra_errors": 0, "failure": None}
    try:
        agent = build(arm, path)
        rep = agent.run(copy.deepcopy(task))
    except ValueError as exc:
        row["failure"] = str(exc)[:80]
        return row
    ran = [s for s, _ in rep["executed"]]
    row["completed"] = bool(rep["completed"])
    row["repeated_steps"] = sum(1 for s in ran if s in verified_before)
    final = MemoryManager(path).get_task(task["task_id"])
    row["extra_errors"] = len(final["errors"]) - errors_before
    if arm == "mea+durable":
        row["crash_detected"] = any(a["action"] == "abandoned_round"
                                    for a in (agent.memory.last_report or {}).get("actions", []))
        row["store_recovered"] = any(e["event"] == "store_recovered" for e in agent.memory.journal.read())
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 40])
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    DATA.mkdir(exist_ok=True)
    tmp = tempfile.TemporaryDirectory(prefix="lh_durable_", ignore_cleanup_errors=True)
    raw, summary = [], []
    try:
        for n in args.sizes:
            steps, _ = make_plan(MockLLM(), GOAL, n)
            for scenario in ("midround", "midround+torn"):
                for arm in ("mea", "mea+durable"):
                    rows = []
                    for t in range(1, args.trials + 1):
                        task = {"task_id": f"db_{n}_{t}", "goal": GOAL, "success_criteria": [],
                                "steps": add_dependencies(steps)}
                        kill_at = max(2, (n * t) // (args.trials + 1))      # different crash point per trial
                        wd = Path(tmp.name) / f"{n}_{t}"
                        wd.mkdir(exist_ok=True)
                        rows.append(trial(arm, scenario, task, kill_at, wd))
                        raw.append({"steps": n, "trial": t, "scenario": scenario, "arm": arm, **rows[-1]})
                    pct = lambda key: round(100 * sum(bool(r[key]) for r in rows) / len(rows), 1)
                    summary.append({"steps": n, "scenario": scenario, "arm": arm,
                                    "completed_pct": pct("completed"),
                                    "crash_detected_pct": pct("crash_detected"),
                                    "store_recovered_pct": pct("store_recovered"),
                                    "repeated_steps": sum(r["repeated_steps"] for r in rows),
                                    "extra_errors": sum(r["extra_errors"] for r in rows)})
    finally:
        tmp.cleanup()
    cols = [("Steps", "steps", 7), ("Scenario", "scenario", 16), ("Arm", "arm", 13),
            ("Done%", "completed_pct", 8), ("Detect%", "crash_detected_pct", 9),
            ("Recover%", "store_recovered_pct", 10), ("Repeat", "repeated_steps", 8),
            ("ExtraErr", "extra_errors", 9)]
    print("".join(f"{c:>{w}}" for c, _, w in cols))
    print("-" * sum(w for _, _, w in cols))
    for r in summary:
        print("".join(f"{str(r[k]):>{w}}" for _, k, w in cols))
    out = DATA / f"durablebench_{stamp}.json"
    out.write_text(json.dumps({"settings": vars(args), "summary": summary, "trials": raw}, indent=2),
                   encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()