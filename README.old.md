# Long-Horizon Agents: Memory Management Module

A runnable prototype that compares three agents on the same multi-step task,
including a simulated crash half-way through:

- **Baseline**: no persistent memory. Puts its whole work log in every prompt.
  After a crash it restarts from step 1.
- **Memory agent**: a JSON task-state store (`MemoryManager`), a bounded
  context in every prompt, and resume-after-crash.
- **MEA agent**: a Manager -> Executor -> Auditor loop on top of the same store,
  after the LongHorizon-Harness paper: the Manager writes a contract for each
  step, the Executor sees only that contract in a fresh context, and the
  Auditor checks the output independently. A Replanner replaces a step that
  keeps failing.

## Requirements
Python 3.11 or newer. No third-party packages.
Optional: [Ollama](https://ollama.com) for a real local model.

## Run the tests
```
python -m unittest discover -s tests
```

## Run on the mock model (offline, fast)
```
python solve.py "Plan a product launch" --steps 8 --only all
python bench.py --sizes 10 40 80 --trials 3
python mea_bench.py
python report_tables.py data\benchmark_<stamp>.json
```

## Run on a real local model
In a separate window, leave this running:
```
ollama serve
ollama pull qwen2.5:3b
```
Then:
```
python solve.py "Plan a simple calculator program" --steps 6 --only all --llm ollama --model qwen2.5:3b
python bench.py --sizes 6 --trials 3 --llm ollama --model qwen2.5:3b
```

## Scripts
| Script | Purpose |
|---|---|
| `solve.py` | Give one task to one or more agents; prints a comparison table and saves a how-to guide |
| `bench.py` | Repeated runs over several task lengths; saves JSON and CSV |
| `mea_bench.py` | Same, with injected failures (dead steps, transient faults, crashes) |
| `recovery_bench.py`, `workflow_bench.py` | Earlier failure and replanning benchmarks |
| `report_tables.py` | Turns a benchmark JSON into a markdown table |
| `main.py` | Original deterministic demo (`demo`, `memory-demo`, `evaluate`, `reset`) |

## Code layout (`agents/`)
| File | Role |
|---|---|
| `memory_manager.py` | JSON store: create/load, record attempts, bounded context, compaction |
| `manager.py` | Manager: reads state, returns execute / done / blocked / ask, writes the step contract |
| `mea.py` | Executor, Auditor (status + integrity) and the loop controller `MEAAgent` |
| `replanner.py` | Replaces a step after repeated rejections, using the saved rejection reason |
| `taskstate.py` | Requirement / fact / artifact records with status and evidence pointers |
| `llm.py` | Mock, Ollama and Anthropic backends behind one `complete()` |
| `llm_agents.py` | Plan generation, rule-based auditor, baseline and memory agents |
| `workflow.py` | Earlier agents with step dependencies and mechanical replanning |

## Memory record
The JSON state holds the goal, status, plan, pending step ids, step history,
errors, findings, summary, timestamps and verification evidence. Only audited
steps leave the pending list. The MEA agent also stores records and artifacts
(see `taskstate.py`).

## How to read the results
- On the mock model the numbers show the mechanism, not model accuracy.
- Input tokens are estimated (characters / 4). Output tokens and time from
  Ollama are measured.
- The real-model samples are small (a 3B model, few trials).
- The baseline has no truncation or summarisation, so its token numbers are an
  upper bound.
- Python RAM is not lower for the memory-based agents, and MEA uses more memory
  time than the memory agent.

## Known limitations
- The auditor checks format and traceability (a `FACT-n` label, the step's code,
  minimum length). It cannot tell whether a statement is true.
- Manager, Auditor and Replanner are rule-based, not model-based as in the paper.
  There is no per-role model routing and no per-role cost split.
- The memory agent's context budget must be tuned to task length (the default
  fails at 80 steps).
- Failures in the benchmarks are synthetic.
- The environment is simulated; nothing touches a real repository.
- LangGraph integration (listed as future work earlier) was not done; a custom
  loop is used instead.