import time
from .planner import Planner
from .auditor import Auditor

class BaselineAgent:
    # No persistent task-state file is read or written.
    def __init__(self, environment):
        self.environment = environment
        self.planner = Planner()
        self.auditor = Auditor()

    def run_task(self, task, max_steps=None):
        start = time.perf_counter()
        verified, history, errors, count = set(), [], [], 0
        while True:
            step = self.planner.next_step(task, verified)
            if step is None or (max_steps is not None and count >= max_steps):
                break
            result = self.environment.execute(step)
            audit = self.auditor.verify(step, result, self.environment)
            status = "verified" if audit["verified"] else "failed"
            history.append({"step_id": step["id"], "description": step["description"],
                            "status": status, "result": result.get("message", ""),
                            "evidence": audit["evidence"]})
            if audit["verified"]:
                verified.add(step["id"])
            else:
                errors.append({"step_id": step["id"], "reason": audit["reason"]})
                break
            count += 1
        complete = len(verified) == len(task["steps"])
        return {"agent": "baseline", "completed": complete,
                "status": "completed" if complete else "incomplete",
                "execution_steps": count, "verified_steps": len(verified),
                "failures": len(errors), "elapsed_seconds": round(time.perf_counter()-start, 6),
                "history_size_chars": len(str(history)), "memory_processing_seconds": 0.0,
                "step_history": history, "errors": errors}
