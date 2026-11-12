"""Executor harnesses: HOW the Executor does its work inside one bounded episode.

LongHorizon-Harness keeps the native agent loop of whatever backend executes a step; the harness only
controls the context, the tools, the budget and the report. Two backends here, same interface:

  ToolLoopHarness   built-in loop. The model may list / read / write files in the workspace (and,
                    only if you allow it, run a Python file there) before giving its answer.
  ExternalCLIHarness  hands the step to a command you choose (any CLI agent) and uses its stdout.

run_episode(prompt, step, stats) returns the same result dict a plain model call does, plus
  declared      {file: sha256} of every file this episode legitimately wrote or changed
  harness_log   what it did, one entry per tool call (the raw trajectory itself is discarded)
The Auditor treats declared files as expected, verifies their hashes, and flags everything else.

Safety: file tools cannot leave the workspace and cannot touch step-N.md (written by the harness).
run_python is OFF by default. It is NOT a real sandbox: model-written code runs with your user's
permissions. Only turn it on (--allow-run) for tasks and models you trust."""
import re
import shlex
import subprocess
import sys

from .llm import estimate_tokens
from .llm_agents import _execute, now
from .workspace import sha256_bytes

RESERVED = re.compile(r"^step-\d+\.md$")
TOOLS = ("list_files", "read_file", "write_file", "run_python")
ACTION_RE = re.compile(r"^\s*ACTION:\s*([A-Za-z_]+)\s*$", re.M)

TOOL_HELP = """You can use tools before you answer. To use one, reply with ONLY these lines:
ACTION: list_files
  or
ACTION: read_file
PATH: <file>
  or
ACTION: write_file
PATH: <file>
CONTENT:
<the file text>
END{run}
When you have what you need, reply with the finished deliverable as plain text. If you do not need a
tool, just answer. Never reply with a tool call and an answer together."""
RUN_HELP = "\n  or\nACTION: run_python\nPATH: <file.py>"


def parse_action(text):
    """(name, path, content) for a tool call, ('finish', None, answer), or None for a plain answer.
    An unknown ACTION word (for example inside a plan that lists 'ACTION: ...') is a plain answer."""
    m = ACTION_RE.search(text)
    if not m:
        return None
    name, rest = m.group(1).lower(), text[m.end():]
    if name == "finish":
        return "finish", None, re.sub(r"^\s*ANSWER:\s*", "", rest.strip(), flags=re.I)
    if name not in TOOLS:
        return None
    pm = re.search(r"^\s*PATH:\s*(.+?)\s*$", rest, re.M)
    cm = re.search(r"^\s*CONTENT:[ \t]*\n(.*?)(?:^\s*END\s*$|\Z)", rest, re.S | re.M)
    return name, pm.group(1).strip("\"'` ") if pm else "", cm.group(1).rstrip("\n") if cm else None


class ToolBox:
    """The tools, confined to one workspace. Every failure is returned as text for the model to read."""

    def __init__(self, workspace, allow_run=False, run_timeout=20, max_read=2000, max_out=1500,
                 max_write=20000):
        self.ws, self.allow_run, self.run_timeout = workspace, allow_run, run_timeout
        self.max_read, self.max_out, self.max_write = max_read, max_out, max_write
        self.written = {}                                   # relative path -> sha256 of what this episode wrote

    def _resolve(self, rel):
        p = (self.ws.root / (rel or "")).resolve()
        if not rel or self.ws.root not in p.parents:
            return None, "ERROR: the path must be a file inside the workspace"
        return p, None

    def call(self, name, path, content):
        try:
            return getattr(self, "_" + name)(path, content)
        except OSError as exc:
            return f"ERROR: {exc}"

    def _list_files(self, path, content):
        files = sorted(self.ws.snapshot())
        return "\n".join(files[:50]) or "(the workspace is empty)"

    def _read_file(self, path, content):
        p, err = self._resolve(path)
        if err or not p.is_file():
            return err or f"ERROR: no such file: {path}"
        text = p.read_text(encoding="utf-8", errors="replace")
        return text[:self.max_read] + ("\n...[cut]" if len(text) > self.max_read else "")

    def _write_file(self, path, content):
        p, err = self._resolve(path)
        if err:
            return err
        if RESERVED.match(p.name):
            return "ERROR: step-N.md files are written by the harness, choose another file name"
        if content is None:
            return "ERROR: write_file needs a CONTENT: block ending with a line END"
        if len(content) > self.max_write:
            return f"ERROR: file too long (limit {self.max_write} characters)"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        rel = p.relative_to(self.ws.root).as_posix()
        self.written[rel] = sha256_bytes(p.read_bytes())
        return f"OK wrote {rel} ({len(content)} characters)"

    def _run_python(self, path, content):
        if not self.allow_run:
            return "ERROR: running code is switched off in this run"
        p, err = self._resolve(path)
        if err or not p.is_file() or p.suffix != ".py":
            return err or "ERROR: give the path of an existing .py file in the workspace"
        before = self.ws.snapshot()
        try:
            proc = subprocess.run([sys.executable, "-I", str(p)], cwd=self.ws.root, capture_output=True,
                                  text=True, timeout=self.run_timeout, stdin=subprocess.DEVNULL)
            out = f"exit code {proc.returncode}\n{proc.stdout}" + (f"\nSTDERR:\n{proc.stderr}" if proc.stderr else "")
        except subprocess.TimeoutExpired:
            out = f"ERROR: stopped after {self.run_timeout} seconds"
        for rel, sha in self.ws.snapshot().items():          # files the program produced are part of the action
            if before.get(rel) != sha and not RESERVED.match(rel.rsplit("/", 1)[-1]):
                self.written[rel] = sha
        return out[:self.max_out] + ("\n...[cut]" if len(out) > self.max_out else "")


def _result(res, box, log, turns):
    """Attach what the Auditor needs; only files that still match their hash stay declared."""
    declared = {}
    for rel, sha in box.written.items():
        try:
            if box.ws.file_hash(rel) == sha:
                declared[rel] = sha
        except OSError:
            pass
    return dict(res, declared=declared, harness_log=log, harness_turns=turns)


class ToolLoopHarness:
    def __init__(self, llm, workspace, max_turns=6, allow_run=False, keep_results=3, **toolbox_kw):
        self.llm, self.workspace = llm, workspace
        self.max_turns, self.allow_run, self.keep = max_turns, allow_run, keep_results
        self.toolbox_kw = toolbox_kw

    def run_episode(self, prompt, step, stats):
        box = ToolBox(self.workspace, self.allow_run, **self.toolbox_kw)
        files = sorted(self.workspace.snapshot())
        head = prompt + ("\n\nFiles already in the workspace: " + ", ".join(files[:20]) if files else "")
        help_ = TOOL_HELP.format(run=RUN_HELP if self.allow_run else "")
        seen, log = [], []                                   # seen = the last few tool results only
        for turn in range(1, self.max_turns + 2):
            last = turn > self.max_turns
            text = head + "\n\n" + (help_ if not last else
                                    "You are out of tool turns. Reply now with the finished deliverable as plain text.")
            if seen:
                text += "\n\nTool results so far (latest last):\n" + "\n".join(seen)
            res = _execute(self.llm, text, step, stats)
            out = res.get("output", "")
            if res.get("timeout") or not out.strip():
                return _result(res, box, log, turn)
            act = None if last else parse_action(out)
            if act is None or act[0] == "finish":
                if act:                                      # 'ACTION: finish' + the answer
                    res["output"] = act[2]
                    res["success"] = bool(act[2].strip())
                return _result(res, box, log, turn)
            name, path, content = act
            obs = box.call(name, path, content)
            log.append({"turn": turn, "action": name, "path": path, "ok": not obs.startswith("ERROR")})
            seen = (seen + [f"[{name} {path}]\n{obs}"])[-self.keep:]      # older results are dropped


class ExternalCLIHarness:
    """Runs YOUR command once per step, in the workspace folder, with the prompt on stdin; its stdout
    is the deliverable. Example: --harness-cmd "claude -p". Files the command creates or changes
    (other than step-N.md) are declared, so the Auditor accepts them but still verifies their hashes.
    It is a generic wrapper: check your tool's own flags and permissions, it runs with your rights."""

    def __init__(self, command, workspace, timeout=1800):
        self.argv, self.workspace, self.timeout = shlex.split(command), workspace, timeout

    def run_episode(self, prompt, step, stats):
        base = {"success": False, "message": "", "output": "", "evidence": {}, "timestamp": now(),
                "declared": {}, "harness_log": [], "harness_turns": 1}
        before = self.workspace.snapshot()
        try:
            proc = subprocess.run(self.argv, input=prompt, capture_output=True, text=True,
                                  cwd=self.workspace.root, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            return dict(base, timeout=True, harness_log=[{"action": "cli", "ok": False, "error": "timeout"}])
        except OSError as exc:
            return dict(base, harness_log=[{"action": "cli", "ok": False, "error": str(exc)}])
        out = proc.stdout.strip() if proc.returncode == 0 else ""
        stats.add({"input_tokens": estimate_tokens(prompt), "output_tokens": estimate_tokens(out)})
        declared = {k: v for k, v in self.workspace.snapshot().items()
                    if before.get(k) != v and not RESERVED.match(k.rsplit("/", 1)[-1])}
        return dict(base, success=bool(out), output=out, declared=declared,
                    harness_log=[{"action": "cli", "ok": proc.returncode == 0,
                                  "error": proc.stderr.strip()[-200:] if proc.returncode else ""}])
