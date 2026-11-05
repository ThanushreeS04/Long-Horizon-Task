
import json
import time
from datetime import datetime, timezone
from pathlib import Path


def now():
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MemoryManager:
    """Persist task state and build compact working context."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read_all(self):
        if not self.path.exists():
            return {}

        try:
            return json.loads(
                self.path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(
                f"Invalid or unreadable memory JSON: {exc}"
            ) from exc

    def _write_all(self, data):
        temp = self.path.with_suffix(
            self.path.suffix + ".tmp"
        )
        temp.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )
        # Windows can briefly lock a freshly written file (antivirus,
        # indexer), so retry the atomic swap a few times.
        for attempt in range(8):
            try:
                temp.replace(self.path)
                return
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.05 * (attempt + 1))

    def create_or_load(self, task):
        all_tasks = self._read_all()
        task_id = task["task_id"]

        if task_id not in all_tasks:
            all_tasks[task_id] = {
                "task_id": task_id,
                "goal": task["goal"],
                "success_criteria": task.get(
                    "success_criteria", []
                ),
                "status": "in_progress",
                "plan": task["steps"],
                "pending_step_ids": [
                    step["id"] for step in task["steps"]
                ],
                "step_history": [],
                "errors": [],
                "findings": [],
                "summary": "",
                "created_at": now(),
                "updated_at": now(),
            }
            self._write_all(all_tasks)

        return all_tasks[task_id]

    def get_task(self, task_id):
        return self._read_all().get(task_id)

    def save_task(self, state):
        all_tasks = self._read_all()
        state["updated_at"] = now()
        all_tasks[state["task_id"]] = state
        self._write_all(all_tasks)

    @staticmethod
    def _limit_text(value, max_chars):
        """Shorten text to a maximum character count."""
        text = str(value)

        if max_chars <= 0:
            return ""

        if len(text) <= max_chars:
            return text

        if max_chars <= 3:
            return text[:max_chars]

        return text[:max_chars - 3] + "..."

    def build_compact_context(
        self,
        state,
        keep_recent=3,
        max_summary_chars=1200,
        max_result_chars=250,
        max_goal_chars=500,
        max_criteria=10,
        max_criteria_chars=200,
        max_errors=3,
        max_findings=5,
        max_finding_chars=200,
    ):
        """Build bounded context while preserving task progress."""
        history = state.get("step_history", [])
        recent = history[-keep_recent:] if keep_recent > 0 else []

        summary = self._limit_text(
            state.get("summary", ""),
            max_summary_chars,
        )

        recent_steps = []

        for step in recent:
            recent_steps.append({
                "step_id": step.get("step_id"),
                "status": step.get("status"),
                "result": self._limit_text(
                    step.get("result", ""),
                    max_result_chars,
                ),
            })

        criteria = [
            self._limit_text(item, max_criteria_chars)
            for item in state.get("success_criteria", [])[
                :max_criteria
            ]
        ]

        pending = state.get("pending_step_ids", [])

        return {
            "goal": self._limit_text(
                state.get("goal", ""),
                max_goal_chars,
            ),
            "success_criteria": criteria,
            "status": state.get("status", "in_progress"),
            "summary": summary,
            "pending_step_ids": list(pending),
            "recent_steps": recent_steps,
            "recent_errors": (
                state.get("errors", [])[-max_errors:]
                if max_errors > 0 else []
            ),
            "findings": [
                self._limit_text(item, max_finding_chars)
                for item in (
                    state.get("findings", [])[-max_findings:]
                    if max_findings > 0 else []
                )
            ],
        }

    @staticmethod
    def _shorten_middle(text, limit):
        """Shorten text but keep both its beginning and its end."""
        text = str(text)
        if limit <= 0:
            return ""
        if len(text) <= limit:
            return text
        if limit <= 6:
            return text[:limit]
        keep = limit - 3
        head = keep // 2
        return text[:head] + "..." + text[len(text) - (keep - head):]

    def build_bounded_context(
        self,
        state,
        budget_chars=2000,
        recent_facts=5,
        recent_fact_chars=160,
        old_fact_chars=48,
        keep_recent_steps=2,
        max_errors=2,
        error_chars=120,
        goal_chars=300,
    ):
        """Context that stays under budget_chars however long the task is.

        Always kept: goal, status, pending count + next pending ids, recent
        step statuses, recent errors. Key facts: the newest ones in full, older
        ones shortened. If the budget is still exceeded, the OLDEST facts are
        dropped and the number dropped is reported (never hidden).
        """
        findings = [str(f) for f in state.get("findings", [])]
        pending = list(state.get("pending_step_ids", []))
        history = state.get("step_history", [])
        errors = state.get("errors", [])

        base = {
            "goal": self._limit_text(state.get("goal", ""), goal_chars),
            "status": state.get("status", "in_progress"),
            "pending_count": len(pending),
            "next_pending": pending[:3],
            "recent_steps": [
                {"step_id": s.get("step_id"), "status": s.get("status")}
                for s in history[-keep_recent_steps:]
            ] if keep_recent_steps > 0 else [],
            "recent_errors": [
                {"step_id": e.get("step_id"),
                 "reason": self._limit_text(e.get("reason", ""), error_chars)}
                for e in errors[-max_errors:]
            ] if max_errors > 0 else [],
        }

        total = len(findings)
        entries = []
        for index, text in enumerate(findings):
            age = total - 1 - index
            limit = recent_fact_chars if age < recent_facts else old_fact_chars
            entries.append(self._shorten_middle(text, limit))

        dropped = 0
        while True:
            context = dict(base, key_facts=entries[dropped:],
                           older_facts_dropped=dropped)
            if len(json.dumps(context)) <= budget_chars or dropped >= len(entries):
                return context
            dropped += 1

    def record_attempt(self, state, step, result, audit):
        if audit["verified"]:
            status = "verified"
        elif not result.get("success"):
            status = "failed"
        else:
            status = "attempted"

        state["step_history"].append({
            "step_id": step["id"],
            "description": step["description"],
            "status": status,
            "result": result.get("message", ""),
            "evidence": audit.get("evidence", {}),
            "audit_reason": audit.get("reason", ""),
            "timestamp": result.get("timestamp", now()),
        })

        if status == "verified":
            state["pending_step_ids"] = [
                step_id
                for step_id in state["pending_step_ids"]
                if step_id != step["id"]
            ]
            state["findings"].append(
                result.get("message", "")
            )
        else:
            state["errors"].append({
                "step_id": step["id"],
                "reason": audit.get(
                    "reason",
                    result.get("message", ""),
                ),
                "timestamp": now(),
            })

        state["status"] = (
            "completed"
            if not state["pending_step_ids"]
            else "in_progress"
        )

        self._compact_if_needed(state)
        self.save_task(state)

        return state

    def _compact_if_needed(
        self,
        state,
        keep_recent=3,
        max_summary_chars=1200,
    ):
        """Summarize older verified steps and preserve unresolved work."""
        history = state.get("step_history", [])

        if keep_recent < 0:
            raise ValueError("keep_recent cannot be negative")

        if max_summary_chars <= 0:
            state["summary"] = ""
        existing_summary = state.get("summary", "")

        # Only older verified steps are eligible for compaction.
        older = history[:-keep_recent] if keep_recent else history
        resolved = [
            step for step in older
            if step.get("status") == "verified"
        ]

        if not resolved:
            state["summary"] = self._limit_text(
                existing_summary,
                max_summary_chars,
            )
            return

        summary_lines = [
            (
                f'{step.get("step_id", "unknown")}: '
                f'{step.get("description", "")} - '
                f'{step.get("result", "")}'
            )
            for step in resolved
        ]

        # Add newly compacted steps after the old summary so the newest
        # information is at the end and can be retained preferentially.
        combined = "\n".join(
            part
            for part in [existing_summary, *summary_lines]
            if part
        )

        state["summary"] = self._limit_text_from_end(
            combined,
            max_summary_chars,
        )

        compacted_ids = {
            step.get("step_id") for step in resolved
        }

        # Keep recent steps and every unresolved step in history.
        state["step_history"] = [
            step for step in history
            if step.get("step_id") not in compacted_ids
        ]

    @staticmethod
    def _limit_text_from_end(value, max_chars):
        """Keep the newest (rightmost) text within the character limit."""
        text = str(value)

        if max_chars <= 0:
            return ""

        if len(text) <= max_chars:
            return text

        if max_chars <= 3:
            return text[-max_chars:]

        return "..." + text[-(max_chars - 3):]
