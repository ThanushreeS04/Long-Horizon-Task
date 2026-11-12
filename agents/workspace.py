"""Task workspace: a real directory the Executor writes its deliverables into and the
Auditor inspects (idea from the v2 runtime). Every path is forced to stay inside the root,
so a step id can never write or read outside it."""
import hashlib
import re
from pathlib import Path


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


class Workspace:
    def __init__(self, path):
        self.root = Path(path).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, step_id):
        p = (self.root / (re.sub(r"[^a-zA-Z0-9_-]", "_", step_id) + ".md")).resolve()
        if self.root not in p.parents:
            raise ValueError("workspace path escaped its root")
        return p

    def snapshot(self):
        """{relative path: sha256} of every file now in the workspace."""
        return {p.relative_to(self.root).as_posix(): sha256_bytes(p.read_bytes())
                for p in sorted(self.root.rglob("*")) if p.is_file()}

    def write_step(self, step_id, content):
        """Write the deliverable and return a receipt (path, sha256, bytes)."""
        p = self.path_for(step_id)
        p.write_text(str(content), encoding="utf-8")
        data = p.read_bytes()
        return {"path": p.relative_to(self.root).as_posix(), "sha256": sha256_bytes(data),
                "bytes": len(data)}

    def read_artifact(self, relative):
        p = (self.root / relative).resolve()
        if self.root not in p.parents or not p.is_file():
            raise ValueError("artifact missing or outside the workspace")
        return p.read_text(encoding="utf-8")

    def file_hash(self, relative):
        return sha256_bytes((self.root / relative).read_bytes())
