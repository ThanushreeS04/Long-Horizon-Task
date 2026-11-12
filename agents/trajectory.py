"""Exports a finished run as a trajectory line for synthetic_pipeline.py (Plan-and-Act, Sec. 4.1).

One JSON object per line: {"query", "success", "failure_prone_indices", "trajectory": [{"observation",
"action"}]}. Steps that needed more than one attempt are marked failure-prone, which the pipeline's
targeted augmentation then focuses on (Sec. 4.3). An 'action' here is the verified result of a step."""
import json
from pathlib import Path


def export_trajectory(goal, plan, step_outputs, errors, success, path):
    errs = {}
    for e in errors:
        errs[e["step_id"]] = errs.get(e["step_id"], 0) + 1
    rows, facts, flaky = [], [], []
    for s in plan:
        out = step_outputs.get(s["id"])
        if not out or s.get("final"):
            continue
        action = " ".join(str(out).split())[:300]
        rows.append({"observation": f"Goal: {goal}. Done so far: {'; '.join(facts[-3:]) or 'nothing yet'}. "
                                    f"Next: {s['description']}", "action": action})
        facts.append(action[:80])
        if errs.get(s["id"], 0) > 0:
            flaky.append(len(rows) - 1)
    if not rows:
        return None
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"query": goal, "success": bool(success), "failure_prone_indices": flaky,
                            "trajectory": rows}, ensure_ascii=False) + "\n")
    return len(rows)
