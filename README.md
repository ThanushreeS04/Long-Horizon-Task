# Long-Horizon Agents: Memory Management Module

A runnable prototype that compares three agents on the same multi-step task,
including a simulated crash half-way through:

- **Baseline**: no persistent memory. Puts its whole work log in every prompt.
  After a crash it restarts from step 1.
- **Memory agent**: a JSON task-state store (`MemoryManager`), a bounded
  context in every prompt, and resume-after-crash. It is the ablation control.
- **MEA agent**: a Manager -> Executor -> Auditor loop on top of the same store,
  after the LongHorizon-Harness paper: the Manager writes a contract for each
  step, the Executor sees only that contract in a fresh context, and the
  Auditor checks the output independently. A Replanner replaces a step that
  keeps failing. MEA is the final system; the memory agent is kept separate so
  the two can be compared.

Controls built for fair comparison: memory+criteria, baseline+criteria and a
sliding-window baseline (`bench.py`), plus `--baseline-window`, `--mea-replan`,
`--llm-replan` and `--llm-audit` in `solve.py`.

## Requirements
Python 3.11 or newer. No third-party packages.
Optional: [Ollama](https://ollama.com) for a real local model (tested with `qwen2.5:3b` on CPU).

## Run the tests
python -m unittest discover -s tests
75 tests, about 12 seconds, no model and no network (a fake local server is
used for the timeout test). The suite is a reduced core set chosen by
per-test line coverage from an earlier 190 tests.

## Run on the mock model (offline, fast)
python solve.py “Plan a product launch” --steps 8 --only all
python bench.py --sizes 10 40 80 --trials 3 --ctx-budget 4000
python mea_bench.py
python memory_size_bench.py
python make_charts.py
python report_tables.py data\results\benchmark_<stamp>.json


## Run on a real local model
In a separate window, leave this running:

ollama serve
ollama pull qwen2.5:3b

Then:

python solve.py “Plan a simple calculator program” --steps 6 --only all --llm ollama --model qwen2.5:3b
python bench.py --sizes 6 --trials 3 --llm ollama --model qwen2.5:3b

Real runs are slow on a laptop CPU (a full benchmark can take an hour). Set
`LH_TIMEOUT` (seconds, default 900) to cap how long one model call may take.

## Dashboard

python frontend.py

Open http://127.0.0.1:8000. Pick the task, steps, backend (Mock by default) and
which agents to run; a Stop button ends a running job. The page shows the
comparison, a cost-by-role table for MEA, and the saved benchmark charts. It
listens on localhost only and refuses requests from other sites. It is a demo
front end over `solve.py` and produces no new results of its own.

## Scripts
| Script | Purpose |
|---|---|
| `solve.py` | Give one task to one or more agents; prints a comparison table and saves a how-to guide |
| `bench.py` | Repeated runs over several task lengths, with the control agents; saves JSON and CSV |
| `mea_bench.py` | Same, with injected failures (dead steps, transient faults, crashes) |
| `memory_size_bench.py` | Saved state size vs the context sent to the model (mock, seconds) |
| `make_charts.py` | Draws SVG charts from `data/results` into `data/results/charts` (no extra packages) |
| `report_tables.py` | Turns a benchmark JSON into a markdown table |
| `frontend.py` | Local dashboard |
| `recovery_bench.py`, `workflow_bench.py` | Earlier failure and replanning benchmarks |
| `main.py` | Original deterministic demo (`demo`, `memory-demo`, `evaluate`, `reset`) |

## Code layout (`agents/`)
| File | Role |
|---|---|
| `memory_manager.py` | JSON store: create/load, record attempts, bounded context, compaction |
| `manager.py` | Manager: reads state, returns execute / done / blocked / ask, writes the step contract |
| `mea.py` | Executor, Auditor (status + integrity) and the loop controller `MEAAgent` |
| `replanner.py` | Replaces a step after repeated rejections, using the saved rejection reason |
| `llm_replanner.py` | Optional model-based replanning and an advisory model audit check (off by default) |
| `adapter.py` | `AgentAdapter`, `ModelRouter` and the per-role cost ledger (calls, tokens, seconds) |
| `taskstate.py` | Requirement / fact / artifact records with status and evidence pointers |
| `llm.py` | Mock, Ollama and Anthropic backends behind one `complete()` |
| `timeouts.py` | `post_json` with a time limit; raises `LLMTimeout` when the model server is too slow |
| `llm_agents.py` | Plan generation, rule-based auditor, baseline and memory agents |
| `workflow.py` | Earlier agents with step dependencies and mechanical replanning |
| `guide.py` | Writes the how-to guide from the verified steps |

## Memory record
The JSON state holds the goal, status, plan, pending step ids, step history,
errors, findings, summary, timestamps and verification evidence. Only audited
steps leave the pending list. The MEA agent also stores records and artifacts
(see `taskstate.py`).

## Results (files in `data/results/`)
Input tokens are estimates (characters / 4). Mock results show the mechanism,
not model accuracy.

| Experiment | Finding |
|---|---|
| Mock, 10-80 steps, 3 trials, crash at 50% | Baseline input tokens grow roughly with the square of task length (16,224 at 10 steps, 1,088,285 at 80). Memory: 2,102 and 46,568. MEA: 1,185 and 9,873. The baseline repeats 50% of steps after the crash; memory and MEA repeat none. |
| Mock, 40 steps, 1-2 dead steps, 5 trials | Baseline, memory+replan and MEA+replan all finish; MEA without replanning does not. With 4 dead steps (more than the replan cap of 3) every agent fails and reports the blocked steps. |
| Mock, memory size | Saved state grows (4,017 to 21,203 characters for 10 to 80 steps) while the context sent to the model stays under its budget (657 to 3,598 characters). This is characters, not Python RAM. |
| Real qwen2.5:3b, 6 steps | Baseline, memory and MEA all complete. MEA uses about 981 input tokens against 2,248 (memory) and 4,594 (baseline). |
| Real, 6 steps, 5 trials (control) | Memory 80%, memory+criteria 100%, MEA 100%. Stating the acceptance criteria explains most of MEA's early advantage. |
| Real, 8 steps, 3 trials | Baseline 67%, sliding-window baseline 0% (it forgets early facts), memory+criteria 100%, MEA 100% with about half the tokens of memory+criteria. |

Not shown: MEA beating the memory agent on completion once the criteria are
stated, and long context as the cause of baseline failures (a baseline+criteria
run was not done).

## How to read the results
- Output tokens and time from Ollama are measured; input tokens are estimated.
- The real-model samples are small (a 3B model, 3 to 5 trials).
- The baseline has no truncation or summarisation, so its token numbers are an
  upper bound.
- Python RAM is not lower for the memory-based agents, and MEA uses more memory
  time than the memory agent.

## Known limitations
- The auditor checks format and traceability (a `FACT-n` label, the step's code,
  minimum length). It cannot tell whether a statement is true. The optional
  model audit is advisory and never blocks a step.
- Manager, Auditor and Replanner are rule-based by default, not model-based as
  in the paper. The model router and per-role ledger exist but were tested with
  mock models only.
- Time limits: an Ollama call that exceeds `LH_TIMEOUT` is interrupted and
  counted as a failed attempt that is retried. The limit applies to each socket
  read, not to total time, and not to the Anthropic backend. Overruns of the
  per-role budgets in `adapter.py` are recorded, not interrupted.
- The memory agent's context budget must be tuned to task length (the default
  of 2000 characters fails at 80 steps).
- Failures in the benchmarks are synthetic.
- The environment is simulated; nothing touches a real repository.
- Tests: the suite was reduced from 190 to 75 by line coverage. A mutation
  score of 55% was measured on a 248-mutant sample of an earlier 185-test
  suite; the reduced suite was not re-scored. Older modules (`baseline_agent.py`,
  `executor.py`, `memory_agent.py`, `auditor.py`) used by `main.py` are lightly tested.
- LangGraph integration (listed as future work earlier) was not done; a custom
  loop is used instead.