# Long-Horizon Agents — Core Implementation Checkpoint

This runnable prototype demonstrates two comparable agents:
- **BaselineAgent:** no persistent task-state storage.
- **MemoryEnabledAgent:** JSON task-state storage, context retrieval, step history, verification, and resumption.

## Scope note
The planner and environment are deterministic in this checkpoint version. This makes the memory workflow testable without an API key, but it is not yet an LLM-backed autonomous coding agent. A model adapter and real repository tools can be added next.

## Requirements
Python 3.11 or 3.12. No third-party packages.

## Run
```bash
python main.py demo
python main.py memory-demo
python -m unittest discover -s tests -v
```

`demo` runs both agents and writes `data/evaluation_results.json`.
`memory-demo` executes two steps, saves state, creates a new agent instance, and resumes from `data/task_memory.json`.
`reset` deletes generated JSON files.

## Memory record
The JSON state includes the task goal, status, plan, pending step IDs, step history, errors, findings, summary, timestamps, and verification evidence. Only audited steps are removed from the pending list.

## Metrics
The prototype records elapsed time, completion, execution steps, failures, compact-context size, and memory-processing time. Token usage is not reported because no LLM API is called yet.

## Next stages
1. Add an LLM-backed planner behind the same interface.
2. Replace the demo environment with safe repository tools.
3. Improve history compaction and add multiple tasks/trials.
