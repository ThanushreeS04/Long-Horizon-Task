"""M5 - Recovery / checkpointing for the memory store.

  Journal               append-only, checksummed log of rounds (begin / executed / commit / ...)
  CheckpointStore       atomic, checksummed snapshots of one task's state, rotated, rollback-able
  DurableMemoryManager  MemoryManager + fsync + .bak + corruption recovery + auto checkpoints
                        + reconcile-on-resume
  JournalingExecutor    wraps the Executor so every round is logged before and after it runs
  reconcile()           safe resumption: validates state, flags the in-flight round, re-checks
                        committed artifacts on disk and demotes steps whose evidence went stale
  make_durable(agent)   one-line wiring for an existing MEAAgent
"""
import hashlib
import json
import os
import re
import shutil
import time
import zlib
from pathlib import Path

from .manager import Manager
from .memory_manager import MemoryManager, now
from .taskstate import _step_of


def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _sha(obj):
    return hashlib.sha256(_canon(obj).encode()).hexdigest()


def atomic_write(path, text):
    """Write to a temp file, fsync, then swap in. Retries because Windows can lock files briefly."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    for attempt in range(8):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.05 * (attempt + 1))


# --------------------------------------------------------------------------- journal
class Journal:
    """JSON-lines log. Each line carries a CRC, so a torn last line or a damaged
    line is skipped on read instead of breaking recovery."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = max((e["seq"] for e in self.read()), default=0)

    def append(self, event, **data):
        self._seq += 1
        body = {"seq": self._seq, "ts": now(), "event": event, **data}
        line = _canon(body)
        record = {"crc": zlib.crc32(line.encode()), "body": body}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return body

    def read(self):
        if not self.path.exists():
            return []
        out = []
        for raw in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rec = json.loads(raw)
                if rec["crc"] == zlib.crc32(_canon(rec["body"]).encode()):
                    out.append(rec["body"])
            except (ValueError, KeyError, TypeError):
                continue                      # torn or damaged line
        return out

    def in_flight(self, task_id=None):
        """The last 'begin' that has no 'commit' or 'abandon' after it, else None."""
        open_round = None
        for e in self.read():
            if task_id and e.get("task_id") not in (None, task_id):
                continue
            if e["event"] == "begin":
                open_round = e
            elif e["event"] in ("commit", "abandon") and open_round \
                    and e.get("round") == open_round.get("round"):
                open_round = None
        return open_round


# --------------------------------------------------------------------------- checkpoints
class CheckpointStore:
    def __init__(self, directory, keep=5):
        self.dir, self.keep = Path(directory), keep
        self.dir.mkdir(parents=True, exist_ok=True)

    def _files(self):
        return sorted(self.dir.glob("ckpt-*.json"))

    def save(self, state, reason):
        files = self._files()
        seq = int(files[-1].stem.split("-")[1]) + 1 if files else 1
        envelope = {"seq": seq, "created": now(), "reason": reason,
                    "sha256": _sha(state), "state": state}
        atomic_write(self.dir / f"ckpt-{seq:06d}.json", json.dumps(envelope, default=str))
        for old in self._files()[:-self.keep]:
            old.unlink(missing_ok=True)
        return seq

    def _load(self, path):
        try:
            env = json.loads(path.read_text(encoding="utf-8"))
            return env if env["sha256"] == _sha(env["state"]) else None
        except (ValueError, KeyError, OSError):
            return None

    def list(self):
        """[(seq, reason, created, valid)] oldest first."""
        out = []
        for p in self._files():
            env = self._load(p)
            seq = int(p.stem.split("-")[1])
            out.append((seq, env["reason"] if env else "?", env["created"] if env else "?", env is not None))
        return out

    def latest_valid(self):
        for p in reversed(self._files()):
            env = self._load(p)
            if env:
                return env
        return None

    def restore(self, seq):
        env = self._load(self.dir / f"ckpt-{seq:06d}.json")
        if env is None:
            raise ValueError(f"checkpoint {seq} missing or corrupt")
        return env["state"]


# --------------------------------------------------------------------------- state checks
def validate_state(state):
    """Return a list of problems (empty list = consistent)."""
    problems = []
    for key in ("task_id", "goal", "plan", "pending_step_ids", "step_history", "errors", "findings"):
        if key not in state:
            problems.append(f"missing key: {key}")
    if problems:
        return problems
    ids = [s["id"] for s in state["plan"]]
    known = set(ids)
    if len(ids) != len(known):
        problems.append("duplicate step ids in plan")
    for key in ("pending_step_ids", "blocked_step_ids", "replaced_step_ids"):
        extra = [i for i in state.get(key, []) if i not in known]
        if extra:
            problems.append(f"{key} names unknown steps: {extra}")
    if len(state["pending_step_ids"]) != len(set(state["pending_step_ids"])):
        problems.append("duplicate pending ids")
    if state.get("status") == "completed" and state["pending_step_ids"]:
        problems.append("status completed but steps are still pending")
    return problems


def _dependents(state, sid):
    """Every step that (transitively) needs `sid`."""
    replaced = set(state.get("replaced_step_ids", []))
    out, frontier = set(), {sid}
    while frontier:
        nxt = {s["id"] for s in state["plan"]
               if s["id"] not in out and frontier & set(Manager._deps(state, s, replaced))}
        out |= nxt
        frontier = nxt
    return out


def _demote(state, sid, why, log):
    """Put a 'completed' step back to pending and drop the facts it produced."""
    if sid in state["pending_step_ids"]:
        return
    state["pending_step_ids"].append(sid)
    state["findings"] = [f for f in state["findings"] if _step_of(f) != sid]
    state["artifacts"] = [a for a in state.get("artifacts", []) if a.get("step") != sid]
    state["status"] = "in_progress"
    order = {s["id"]: i for i, s in enumerate(state["plan"])}
    state["pending_step_ids"].sort(key=lambda i: order.get(i, 0))
    log.append({"action": "demoted", "step": sid, "why": why})


def reconcile(state, journal, workspace=None):
    """Make a loaded state safe to resume from. Changes `state` in place (caller saves)
    and returns a report. Three checks:
      1. structure       - validate_state
      2. in-flight round - a 'begin' without a 'commit' is abandoned; the step is still
                           pending, and a crash is not counted as a failed attempt
      3. stale evidence  - each committed artifact is re-hashed on disk; if it changed or
                           vanished the step and everything built on it goes back to pending
    """
    log = []
    problems = validate_state(state)
    task_id = state.get("task_id")
    open_round = journal.in_flight(task_id)
    if open_round:
        journal.append("abandon", task_id=task_id, round=open_round.get("round"),
                       step=open_round.get("step"), why="crash before commit")
        log.append({"action": "abandoned_round", "step": open_round.get("step"),
                    "round": open_round.get("round")})
    if workspace is not None and not problems:
        last_commit = {}
        for e in journal.read():
            if e["event"] == "commit" and e.get("task_id") == task_id and e.get("verified") and e.get("artifact"):
                last_commit[e["step"]] = e["artifact"]
        completed = (set(s["id"] for s in state["plan"]) - set(state["pending_step_ids"])
                     - set(state.get("blocked_step_ids", [])) - set(state.get("replaced_step_ids", [])))
        for sid, art in last_commit.items():
            if sid not in completed:
                continue
            try:
                same = workspace.file_hash(art["path"]) == art["sha256"]
            except OSError:
                same = False
            if not same:
                _demote(state, sid, f"artifact {art['path']} changed or missing on disk", log)
                for dep in sorted(_dependents(state, sid) & completed):
                    _demote(state, dep, f"depends on {sid}, whose evidence went stale", log)
    if log:
        state.setdefault("recovery_log", []).extend({**e, "at": now()} for e in log)
    return {"problems": problems, "actions": log}


# --------------------------------------------------------------------------- durable manager
class DurableMemoryManager(MemoryManager):
    def __init__(self, path, workspace=None, keep_checkpoints=5):
        super().__init__(path)
        stem = self.path.with_suffix("")
        self.journal = Journal(f"{stem}.journal.jsonl")
        self._ckpt_root = Path(f"{stem}.ckpt")
        self._keep = keep_checkpoints
        self.workspace = workspace
        self.last_report = None

    # ---- checkpoints
    def checkpoints(self, task_id):
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", str(task_id))
        return CheckpointStore(self._ckpt_root / safe, self._keep)

    def checkpoint(self, state, reason):
        seq = self.checkpoints(state["task_id"]).save(state, reason)
        self.journal.append("checkpoint", task_id=state["task_id"], ckpt=seq, reason=reason)
        return seq

    def rollback(self, task_id, seq):
        """Restore a checkpoint as the live state (e.g. undo a bad replan)."""
        state = self.checkpoints(task_id).restore(seq)
        MemoryManager.save_task(self, state)
        self.journal.append("rollback", task_id=task_id, ckpt=seq)
        return state

    # ---- storage with recovery
    def _read_all(self):
        try:
            data = super()._read_all()
            if not isinstance(data, dict):
                raise ValueError("memory file root is not an object")
            return data
        except ValueError:
            return self._recover_store()

    def _recover_store(self):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        if self.path.exists():
            shutil.move(str(self.path), f"{self.path}.corrupt-{stamp}")
        recovered = {}
        if self._ckpt_root.exists():
            for d in sorted(p for p in self._ckpt_root.iterdir() if p.is_dir()):
                env = CheckpointStore(d, self._keep).latest_valid()
                if env:
                    recovered[env["state"]["task_id"]] = env["state"]
        bak = Path(f"{self.path}.bak")
        if bak.exists():
            try:
                for tid, st in json.loads(bak.read_text(encoding="utf-8")).items():
                    recovered.setdefault(tid, st)
            except (ValueError, AttributeError):
                pass
        if not recovered:
            raise ValueError("memory file is corrupt and no checkpoint or backup could be used")
        atomic_write(self.path, json.dumps(recovered, indent=2, default=str))
        self.journal.append("store_recovered", tasks=sorted(recovered))
        return recovered

    def _write_all(self, data):
        if self.path.exists():
            shutil.copy2(self.path, f"{self.path}.bak")
        atomic_write(self.path, json.dumps(data, indent=2, default=str))

    # ---- hooks the agent already calls
    def create_or_load(self, task):
        existed = self.get_task(task["task_id"]) is not None
        state = super().create_or_load(task)
        if existed:                                    # resuming: make it safe first
            self.last_report = reconcile(state, self.journal, self.workspace)
            if self.last_report["actions"]:
                self.save_task(state)
        else:
            self.checkpoint(state, "created")
        return state

    def save_task(self, state):
        # A replan can rewrite a step, add one, or drop one. A dropped step keeps its text and
        # only gets a flag plus an entry in replaced_step_ids, so those are part of the hash.
        plan_hash = _sha([[(s["id"], s.get("description"), bool(s.get("dropped_by_replan")))
                           for s in state["plan"]], sorted(state.get("replaced_step_ids", []))])
        changed = state.get("_plan_hash") not in (None, plan_hash)
        state["_plan_hash"] = plan_hash
        super().save_task(state)
        if changed:                                    # the plan was revised
            self.checkpoint(state, "replan")

    def record_attempt(self, state, step, result, audit):
        state = super().record_attempt(state, step, result, audit)
        ev = audit.get("evidence", {})
        art = {"path": ev["artifact"], "sha256": ev["sha256"]} if ev.get("artifact") else None
        self.journal.append("commit", task_id=state["task_id"], round=self._round(),
                            step=step["id"], verified=bool(audit["verified"]), artifact=art)
        if audit["verified"]:
            self.checkpoint(state, f"verified {step['id']}")
        return state

    def _round(self):
        return sum(1 for e in self.journal.read() if e["event"] == "begin")


class JournalingExecutor:
    """Logs 'begin' before the model runs and 'executed' after, so a crash between
    the two (or before commit) is visible on resume."""

    def __init__(self, inner, journal):
        self.inner, self.journal = inner, journal

    def run(self, goal, contract, step, stats):
        rnd = sum(1 for e in self.journal.read() if e["event"] == "begin") + 1
        self.journal.append("begin", round=rnd, step=step["id"], attempt=contract.get("attempt"))
        result = self.inner.run(goal, contract, step, stats)
        self.journal.append("executed", round=rnd, step=step["id"],
                            sha=hashlib.sha256(str(result.get("output", "")).encode()).hexdigest()[:16])
        return result

    def __getattr__(self, name):
        return getattr(self.inner, name)


def make_durable(agent, workspace=None, keep_checkpoints=5):
    """Wire recovery into an existing MEAAgent (no edits to mea.py needed)."""
    mem = DurableMemoryManager(agent.memory.path, workspace, keep_checkpoints)
    agent.memory = mem
    agent.executor = JournalingExecutor(agent.executor, mem.journal)
    return agent