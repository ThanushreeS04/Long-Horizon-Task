"""Deletes the .json and .csv files a previous run created.

Safe by design: it deletes ONLY files that a run registered itself, and only
inside the runs/ folder. Benchmark data, results, inputs and charts are never
touched. The runner calls purge_previous() at the start of every run, so each
run's files live until the next run begins (or until you run this module).

    python -m agents.cleanup          # delete the previous run's files now
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
DELETABLE = {".json", ".csv"}


class RunFiles:
    def __init__(self, runs_dir=RUNS):
        self.runs_dir = Path(runs_dir).resolve()
        self.manifest = self.runs_dir / ".last_run_files"

    def register(self, path):
        """Remember a file this run will create; refuses paths outside runs/."""
        p = Path(path).resolve()
        if self.runs_dir not in p.parents:
            raise ValueError(f"{p} is outside {self.runs_dir}; refusing to track it for deletion")
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        with open(self.manifest, "a", encoding="utf-8") as f:
            f.write(str(p) + "\n")
        return p

    def unregister(self, path):
        """Stop tracking a file so the next run does not delete it (e.g. a paused ASK run)."""
        p = str(Path(path).resolve())
        if self.manifest.exists():
            keep = [l for l in self.manifest.read_text(encoding="utf-8").splitlines() if l.strip() and l.strip() != p]
            self.manifest.write_text("".join(l + "\n" for l in keep), encoding="utf-8")

    def _delete_listed(self):
        deleted = []
        if not self.manifest.exists():
            return deleted
        for line in self.manifest.read_text(encoding="utf-8").splitlines():
            p = Path(line.strip()).resolve() if line.strip() else None
            if (p and p.suffix.lower() in DELETABLE and self.runs_dir in p.parents
                    and p.is_file()):
                try:
                    p.unlink()
                    deleted.append(str(p))
                except OSError:
                    pass                    # locked or already gone: leave it for next time
        try:
            self.manifest.unlink()
        except OSError:
            pass
        return deleted

    purge_previous = _delete_listed       # call at the start of a run
    purge_current = _delete_listed        # call at the end of a run (--clean-now)


if __name__ == "__main__":
    gone = RunFiles().purge_previous()
    print(f"Deleted {len(gone)} file(s) from the previous run." if gone
          else "Nothing to delete.")
    for g in gone:
        print("  ", g)
