import time
from .planner import Planner
from .auditor import Auditor
from .memory_manager import MemoryManager

class MemoryEnabledAgent:
    def __init__(self, environment, memory_path):
        self.environment = environment
        self.memory = MemoryManager(memory_path)
        self.planner = Planner()
        self.auditor = Auditor()

    def run_task(self, task, max_steps=None):
        state = self.memory.create_or_load(task)
        return self._run(task, state, max_steps)

    def resume_task(self, task_id, max_steps=None):
        state = self.memory.get_task(task_id)
        if state is None:
            raise KeyError(f"No saved task found for {task_id}")
        task = {"task_id": task_id, "goal": state["goal"],
                "success_criteria": state.get("success_criteria", []),
                "steps": state["plan"]}
        return self._run(task, state, max_steps)

    def _run(self, task, state, max_steps):
        start, memory_seconds, count = time.perf_counter(), 0.0, 0
        while state.get("pending_step_ids"):
            if max_steps is not None and count >= max_steps:
                break
            t0 = time.perf_counter()
            self.memory.build_compact_context(state)
            memory_seconds += time.perf_counter() - t0
            verified = {s["step_id"] for s in state["step_history"]
                        if s["status"] == "verified"}
            # Steps no longer pending have already been verified, including compacted ones.
            verified.update(s["id"] for s in task["steps"]
                            if s["id"] not in state["pending_step_ids"])
            step = self.planner.next_step(task, verified)
            if step is None or step["id"] not in state["pending_step_ids"]:
                break
            result = self.environment.execute(step)
            audit = self.auditor.verify(step, result, self.environment)
            t0 = time.perf_counter()
            state = self.memory.record_attempt(state, step, result, audit)
            memory_seconds += time.perf_counter() - t0
            count += 1
            if not audit["verified"]:
                break
        completed = state.get("status") == "completed"
        context_chars = len(str(self.memory.build_compact_context(state)))
        return {"agent": "memory_enabled", "completed": completed,
                "status": state.get("status", "in_progress"),
                "execution_steps_this_session": count,
                "pending_steps": state.get("pending_step_ids", []),
                "failures_total": len(state.get("errors", [])),
                "elapsed_seconds": round(time.perf_counter()-start, 6),
                "compact_context_size_chars": context_chars,
                "memory_processing_seconds": round(memory_seconds, 6),
                "memory_file": str(self.memory.path)}
