"""Memory size before/after compaction (mock model, no Ollama, a few seconds).

For each task size: run the memory agent to the end, then compare
  - saved state: size of the full JSON state on disk (grows with steps)
  - context: size of the bounded context sent to the model (stays under budget)
Sizes are characters, not bytes or Python RAM."""
import json
import tempfile
from pathlib import Path

from agents.llm import MockLLM
from agents.llm_agents import LLMMemoryAgent
from tests.test_llm_agents import make_task

BUDGET = 4000


def measure(n):
    p = Path(tempfile.mkdtemp()) / "m.json"
    task = make_task(n)
    agent = LLMMemoryAgent(MockLLM(), p, ctx_budget=BUDGET)
    rep = agent.run(task)
    state = agent.memory.create_or_load(task)
    full = len(json.dumps(state))
    ctx = agent.memory.build_bounded_context(state, budget_chars=BUDGET)
    ctx_chars = len(json.dumps(ctx))
    return {"steps": n, "verified": rep["final_verified"], "state_chars": full,
            "context_chars": ctx_chars, "ratio": round(full / max(1, ctx_chars), 1),
            "within_budget": ctx_chars <= BUDGET}


if __name__ == "__main__":
    print(f"{'steps':>6} {'done':>5} {'state chars':>12} {'context chars':>14} {'state/ctx':>10} {'<=budget':>9}")
    rows = [measure(n) for n in (10, 20, 40, 80)]
    for r in rows:
        print(f"{r['steps']:>6} {str(r['verified']):>5} {r['state_chars']:>12} "
              f"{r['context_chars']:>14} {r['ratio']:>10} {str(r['within_budget']):>9}")
    out = Path("data/results"); out.mkdir(parents=True, exist_ok=True)
    (out / "memory_size.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("saved data/results/memory_size.json")