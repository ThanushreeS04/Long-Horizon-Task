"""Task-state records (LongHorizon-Harness): requirements, facts and artifacts,
each with a status and a pointer to its audit evidence. Built from the saved
state, so it never changes what the agent trusts."""
import re

COMPLETED, PENDING, BLOCKED, UNTRUSTED = "completed", "pending", "blocked", "untrusted"


def _step_of(text):
    m = re.search(r"FACT-(\d+)\b", str(text))
    return f"step-{m.group(1)}" if m else None


def build_records(state):
    pending = set(state["pending_step_ids"])
    blocked = set(state.get("blocked_step_ids", []))
    replaced = set(state.get("replaced_step_ids", []))
    replaces = {r["alternative"]: r["failed"] for r in state.get("replans", [])}
    findings = [str(f) for f in state.get("findings", [])]
    finding_at = {}
    for i, f in enumerate(findings):
        if _step_of(f):
            finding_at[_step_of(f)] = i
    attempts = {}
    for h in state.get("step_history", []):
        attempts.setdefault(h["step_id"], []).append(h)
    failed_count = {}
    for e in state.get("errors", []):
        failed_count[e["step_id"]] = failed_count.get(e["step_id"], 0) + 1

    requirements, untrusted_facts = [], []
    for s in state["plan"]:
        sid = s["id"]
        if sid in replaced:
            continue
        hist = attempts.get(sid, [])
        last = hist[-1] if hist else None
        verified = next((h for h in reversed(hist) if h["status"] == "verified"), None)
        if sid in blocked:
            status = BLOCKED
        elif sid not in pending:
            status = COMPLETED
        elif last and last.get("evidence", {}).get("integrity", "clean") != "clean":
            status = UNTRUSTED
        else:
            status = PENDING
        evidence = None
        if status == COMPLETED:
            if verified:
                evidence = {"source": "step_history", "timestamp": verified["timestamp"],
                            "reason": verified["audit_reason"]}
            elif sid in finding_at:                 # history was compacted
                evidence = {"source": "findings", "index": finding_at[sid]}
        elif last:
            evidence = {"source": "step_history", "timestamp": last["timestamp"],
                        "reason": last["audit_reason"]}
            untrusted_facts.append({"id": f"claim-{sid}", "step": sid, "text": last["result"],
                                    "status": UNTRUSTED, "evidence": evidence})
        requirements.append({"id": sid, "text": s["description"], "status": status,
                             "failed_attempts": failed_count.get(sid, 0),
                             "replaces": replaces.get(sid), "evidence": evidence})
    facts = [{"id": f"fact-{i}", "step": _step_of(f), "text": f, "status": COMPLETED,
              "evidence": {"source": "findings", "index": i}} for i, f in enumerate(findings)]
    counts = {k: sum(1 for r in requirements if r["status"] == k)
              for k in (COMPLETED, PENDING, BLOCKED, UNTRUSTED)}
    return {"requirements": requirements, "facts": facts,
            "unverified_claims": untrusted_facts,
            "artifacts": list(state.get("artifacts", [])), "counts": counts}