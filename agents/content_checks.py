"""Content checks for the Auditor (fixes 'passes the format checks but is wrong').

  ContentAuditor  rejects output that (a) uses another language's syntax, or
                  (b) imports a module that does not exist in the standard library
  ContentManager  adds the task constraints to every contract (extra_acceptance)

What rules cannot catch: logic that is well-formed but wrong (e.g. a 'calculator' that
only doubles a number). Use a constraint for that, or a stronger model for the auditor role.
"""
import re
import sys

from .manager import EXECUTE, Manager
from .mea import Auditor

# Syntax that only appears in another language. Kept short on purpose (few false alarms).
FOREIGN_SYNTAX = {
    "python": [r"\bthrow new Error\b", r"\bisNaN\s*\(", r"\bconsole\.log\b", r"\bconst\s+\w+\s*=",
               r"\blet\s+\w+\s*=", r"\bfunction\s+\w+\s*\(", r"===", r"\bSystem\.out\.print",
               r"#include\s*<", r"\bpublic\s+static\s+void\b"],
}
# 'from X import' is matched anywhere (the model often quotes code inside a sentence);
# a bare 'import X' only at a line start or right after a quote or ';', so prose such as
# "to import functions" is not mistaken for code.
_FROM = re.compile(r"\bfrom\s+([A-Za-z_]\w*)[\w.]*\s+import\b")
_IMPORT = re.compile(r"(?:^|[\'\";])\s*import\s+([A-Za-z_]\w*)", re.M)
_NOT_MODULES = {"the", "a", "an", "this", "that", "it", "which", "each", "there", "here", "them"}
_STDLIB = set(getattr(sys, "stdlib_module_names", ()))


def foreign_syntax(text, language):
    return [p.strip(r"\b") for p in FOREIGN_SYNTAX.get(language, []) if re.search(p, text)]


def unknown_imports(text, allowed=()):
    """Python modules imported in `text` that are neither standard library nor allowed."""
    names = {m.group(1) for m in _FROM.finditer(text)} | {m.group(1) for m in _IMPORT.finditer(text)}
    names -= _NOT_MODULES
    ok = _STDLIB | set(allowed)
    return sorted(n for n in names if n not in ok) if _STDLIB else []


class ContentAuditor(Auditor):
    def __init__(self, language="python", allowed_modules=()):
        super().__init__()
        self.language, self.allowed = language, tuple(allowed_modules)

    def audit(self, contract, step, result, state):
        res = super().audit(contract, step, result, state)
        if not res["verified"] or step.get("final"):
            return res
        text, problems = result.get("output", ""), []
        if self.language == "python":
            bad = foreign_syntax(text, "python")
            if bad:
                problems.append(f"uses non-Python syntax ({bad[0]}); the task is {self.language}")
            unknown = unknown_imports(text, self.allowed)
            if unknown:
                problems.append(f"imports modules that are not in the standard library: {unknown}")
        if problems:
            res.update(verified=False, status="incomplete", reason="content check: " + "; ".join(problems),
                       gaps=list(res.get("gaps", [])) + problems)
            res["evidence"]["content_check"] = problems
        return res


class ContentManager(Manager):
    def __init__(self, constraints=(), **kw):
        super().__init__(**kw)
        self.constraints = list(constraints)

    def decide(self, state):
        d = super().decide(state)
        if d["decision"] == EXECUTE and self.constraints:
            d["contract"]["extra_acceptance"] = list(d["contract"].get("extra_acceptance", [])) + self.constraints
        return d


def default_constraints(language):
    out = []
    if language == "python":
        out += ["use Python only (no JavaScript or other languages)",
                "import only standard-library modules; do not invent libraries"]
    return out