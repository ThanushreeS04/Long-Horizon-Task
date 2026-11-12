"""Environment-grounded Auditor: judges what is ON DISK, not what the Executor says.

For every round it
  1. re-reads the artifact from the workspace (the Executor's text reply is ignored),
  2. compares its SHA-256 with the Executor's write receipt (changed since = violation),
  3. diffs the workspace before and after the executor ran (any other file touched = suspect),
  4. runs the usual rule checks and, if enabled, the model judge on the DISK content.
Status and integrity are combined, so only complete + clean counts as verified."""
from .llm_auditor import LLMAuditor
from .mea import Auditor

_RANK = {"clean": 0, "suspect": 1, "violation": 2}


class EnvironmentAuditor(LLMAuditor):
    def __init__(self, adapter, workspace, use_model=True, **kw):
        super().__init__(adapter, **kw)
        self.workspace, self.use_model = workspace, use_model

    def _inner(self, contract, step, result, state):
        base = LLMAuditor if self.use_model else Auditor
        return base.audit(self, contract, step, result, state)

    def audit(self, contract, step, result, state):
        art = result.get("artifact")
        if not art:                                   # nothing was written: nothing to verify
            res = Auditor.audit(self, contract, step, dict(result, output=""), state)
            res.update(verified=False, status="incomplete", reason="executor wrote no artifact")
            res["gaps"] = ["executor wrote no artifact"]
            return res
        ws, rel = self.workspace, art["path"]
        problems, integrity = [], "clean"
        try:
            content = ws.read_artifact(rel)
            actual = ws.file_hash(rel)
        except (ValueError, OSError) as exc:
            res = Auditor.audit(self, contract, step, dict(result, output=""), state)
            res.update(verified=False, status="blocked", integrity="violation",
                       reason=f"artifact unreadable: {exc}", gaps=[f"artifact unreadable: {exc}"])
            return res
        if actual != art["sha256"]:
            integrity = "violation"
            problems.append("artifact changed after the executor wrote it")
        declared = result.get("declared") or {}
        for path, sha in declared.items():                    # files the harness wrote on purpose
            try:
                same = ws.file_hash(path) == sha
            except OSError:
                same = False
            if not same:
                integrity = "violation"
                problems.append(f"declared file {path} changed or vanished after the executor wrote it")
        before = result.get("before_snapshot") or {}
        now = ws.snapshot()
        changed = sorted(k for k in set(before) | set(now)
                         if before.get(k) != now.get(k) and k != rel and k not in declared)
        if changed:
            integrity = max(integrity, "suspect", key=_RANK.get)
            problems.append(f"unexpected workspace changes: {changed}")
        res = self._inner(contract, step, dict(result, output=content), state)   # judge the DISK copy
        if _RANK[integrity] > _RANK[res["integrity"]]:
            res["integrity"] = integrity
        if problems:
            res["gaps"] = list(res.get("gaps", [])) + problems
            res["reason"] = ("; ".join(problems) + ". " + str(res.get("reason", ""))).strip()
        res["verified"] = res["status"] == "complete" and res["integrity"] == "clean"
        res["evidence"].update(artifact=rel, sha256=actual, bytes=len(content.encode()),
                               workspace_changes=changed, declared_files=sorted(declared))
        return res
