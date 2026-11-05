"""Baseline and memory-enabled agents that work on any free-text task."""
import hashlib
import json
import re
import time
from datetime import datetime, timezone

from .memory_manager import MemoryManager

MAX_ATTEMPTS = 3


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_code(i):
    """Unique 'needle' code per step, used to test whether facts survive."""
    return f"K{i}-" + hashlib.md5(f"needle{i}".encode()).hexdigest()[:4]


def step_number(step):
    return int(step["id"].split("-")[1])


def step_rules(step):
    if step.get("final"):
        return ("Write a final report that lists every FACT-n line established "
                "in the earlier steps, with its exact code.")
    i = step_number(step)
    return (f"Reply with line 1 = 'FACT-{i}: <one short fact that includes the exact "
            f"code {step['code']}>', then 3-5 sentences of work for this step.")


def parse_steps(text, count):
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        items = json.loads(cleaned)
        if isinstance(items, list):
            return [str(x) for x in items][:count]
    except json.JSONDecodeError:
        pass
    lines = [re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", ln).strip() for ln in cleaned.splitlines()]
    return [ln for ln in lines if ln][:count]


def make_plan(llm, goal, n_steps):
    """Ask the LLM for n_steps-1 steps, then append a final report step."""
    count = n_steps - 1
    prompt = (f"Break this task into exactly {count} short, sequential steps. "
              f"Return ONLY a JSON array of strings.\nTask: {goal}")
    r = llm.complete(prompt, kind="plan", max_tokens=700, meta={"goal": goal, "n": count})
    descs = parse_steps(r["text"], count)
    while len(descs) < count:
        descs.append(f"Continue working on: {goal}")
    descs.append("Compile the final report")
    steps = []
    for i, d in enumerate(descs, start=1):
        steps.append({"id": f"step-{i}", "description": d, "operation": "llm_step",
                      "expected": f"FACT-{i}", "code": make_code(i), "final": i == len(descs),
                      "required_codes": ([make_code(j) for j in range(1, len(descs))]
                                         if i == len(descs) else [])})
    return steps, r


class OutputAuditor:
    """Checks the step output itself; does not trust the executor's claim."""

    def verify(self, step, result, environment=None):
        text = result.get("output", "")
        if step.get("final"):
            missing = [c for c in step.get("required_codes", []) if c not in text]
            ok = len(text.strip()) >= 40 and not missing
            return {"verified": ok, "evidence": {"length": len(text.strip()), "missing_codes": missing},
                    "reason": "Final report lists every code." if ok
                              else f"Final report missing {len(missing)} required code(s)."}
        i = step_number(step)
        head = text[:300]
        has_label = re.search(rf"FACT-{i}\b", head) is not None
        has_code = step["code"] in head
        long_enough = len(text.strip()) >= 80
        ok = has_label and has_code and long_enough
        return {"verified": ok,
                "evidence": {"has_fact_label": has_label, "has_code": has_code, "long_enough": long_enough},
                "reason": "Fact line and work present." if ok else
                          "Missing: " + ", ".join(
                              m for m, present in (
                                  (f"a line starting FACT-{i}", has_label),
                                  (f"the exact code {step['code']} in that line", has_code),
                                  ("at least 80 characters of work", long_enough))
                              if not present)}


class _Stats:
    def __init__(self):
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.prompt_series = []
        self.executed = []          # (step_id, verified)
        self.final_output = None
        self.final_verified = False

    def add(self, r):
        self.calls += 1
        self.input_tokens += r["input_tokens"]
        self.output_tokens += r["output_tokens"]
        self.prompt_series.append(r["input_tokens"])

def fact_text(text, step):
    """The fact memory keeps: from the FACT label through the step's code, so
    what the auditor verified is exactly what memory remembers."""
    flat = text.strip()
    m = re.search(rf"FACT-{step_number(step)}\b", flat)
    start = m.start() if m else 0
    code = step.get("code", "")
    pos = flat.find(code, start) if code else -1
    if pos != -1:
        seg = flat[start:pos + len(code)]
    else:
        seg = next((ln for ln in flat[start:].splitlines() if ln.strip()), flat[start:start + 200])
    return MemoryManager._shorten_middle(" ".join(seg.split()), 200)

def _execute(llm, prompt, step, stats):
    kind = "final" if step.get("final") else "step"
    r = llm.complete(prompt, kind=kind, max_tokens=700 if step.get("final") else 400,
                     meta={"index": step_number(step), "code": step["code"], "desc": step["description"]})
    stats.add(r)
    text = r["text"]
    
    return {"success": bool(text.strip()), "message": fact_text(text, step), "output": text,
            "evidence": {}, "timestamp": now()}


def _report(agent, stats, interrupted, elapsed, memory_seconds=0.0):
    return {"agent": agent, "interrupted": interrupted, "llm_calls": stats.calls,
            "input_tokens": stats.input_tokens, "output_tokens": stats.output_tokens,
            "prompt_series": stats.prompt_series, "executed": stats.executed,
            "final_output": stats.final_output, "final_verified": stats.final_verified,
            "elapsed_seconds": round(elapsed, 4),
            "memory_seconds": round(memory_seconds, 4)}


class LLMBaselineAgent:
    """No persistence. Full history of this session is placed in every prompt."""

    def __init__(self, llm):
        self.llm = llm

    def run(self, task, stop_after=None):
        start, stats, log = time.perf_counter(), _Stats(), []
        done, interrupted = 0, False
        for step in task["steps"]:
            if stop_after is not None and done >= stop_after:
                interrupted = True
                break
            history = "\n\n".join(log) if log else "(nothing yet)"
            base_prompt = (f"Task goal: {task['goal']}\n\nFull work log so far:\n{history}\n\n"
                           f"Now do {step['id']}: {step['description']}\n{step_rules(step)}")
            prompt, verified = base_prompt, False
            for _ in range(MAX_ATTEMPTS):
                result = _execute(self.llm, prompt, step, stats)
                audit = OutputAuditor().verify(step, result)
                verified = audit["verified"]
                stats.executed.append((step["id"], verified))
                if verified:
                    break
                # same information the memory agent gets from its recent_errors
                prompt = base_prompt + f"\n\nYour previous attempt was rejected. {audit['reason']}"
            if step.get("final"):
                stats.final_output, stats.final_verified = result["output"], verified
            if not verified:
                break
            log.append(f"[{step['id']}] {step['description']}\n{result['output']}")
            done += 1
        return _report("baseline", stats, interrupted, time.perf_counter() - start)


class LLMMemoryAgent:
    """Uses MemoryManager: saves state to JSON, builds compact context, resumes."""

    def __init__(self, llm, memory_path, ctx_budget=2000):
        self.llm = llm
        self.memory = MemoryManager(memory_path)
        self.ctx_budget = ctx_budget

    def run(self, task, stop_after=None):
        start, stats, mem_s = time.perf_counter(), _Stats(), 0.0
        t0 = time.perf_counter()
        state = self.memory.create_or_load(task)      # loads saved state if it exists
        mem_s += time.perf_counter() - t0
        attempts, done, interrupted = {}, 0, False
        auditor = OutputAuditor()
        while state["pending_step_ids"]:
            if stop_after is not None and done >= stop_after:
                interrupted = True
                break
            step = next(s for s in state["plan"] if s["id"] in state["pending_step_ids"])
            if attempts.get(step["id"], 0) >= MAX_ATTEMPTS:
                break
            t0 = time.perf_counter()
            ctx = self.memory.build_bounded_context(state, budget_chars=self.ctx_budget)
            mem_s += time.perf_counter() - t0
            prompt = (f"Task goal: {state['goal']}\n\nCompact task memory (JSON):\n{json.dumps(ctx)}\n\n"
                      f"Now do {step['id']}: {step['description']}\n{step_rules(step)}")
            result = _execute(self.llm, prompt, step, stats)
            audit = auditor.verify(step, result)
            stats.executed.append((step["id"], audit["verified"]))
            attempts[step["id"]] = attempts.get(step["id"], 0) + 1
            t0 = time.perf_counter()
            state = self.memory.record_attempt(state, step, result, audit)
            mem_s += time.perf_counter() - t0
            if step.get("final"):
                stats.final_output, stats.final_verified = result["output"], audit["verified"]
            if audit["verified"]:
                done += 1
        return _report("memory_enabled", stats, interrupted, time.perf_counter() - start, mem_s)