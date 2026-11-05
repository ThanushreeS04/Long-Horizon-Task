I'm Thanushree S (SRN R23EF282), doing a major project: "AI Agents for Long-Horizon Tasks – Memory Management Module" (CAIML_Nimrita_Koul_3_PS_212). I'm continuing from a long earlier chat. Please read this summary, then pick up at "CURRENT STATE".

HOW I WANT YOU TO HELP
- One step at a time. Give exact find/replace edits or full code IN THE CHAT. Do NOT create files, artifacts or documents for me.
- I use Windows PowerShell at D:\Major Project\long_horizon_agent. Give simple, single commands.
- I sometimes forget to save in Notepad (Ctrl+S). After edits, verify with Select-String before trusting results.
- Be honest about mock vs real-model results. Don't oversell.

BACKGROUND
- Papers: LongHorizon-Harness (Manage-Execute-Audit loop, task state outside context, independent auditor) and Plan-and-Act (planner/executor split, replanning).
- My Task 1-3 documents are done: literature survey, system design, and Memory Management Module design (MemoryManager + JSON store, Planner, Executor, Auditor).
- Goal: build a baseline agent (no persistent memory) and a memory agent (uses my MemoryManager), give both the same free-text task, and compare them fairly, including a simulated crash halfway.
- Environment: Ollama running locally with only qwen2.5:3b (CPU only, ~7.7 GB RAM, started with "ollama serve" in a separate window). I have no Anthropic key.

PROJECT FILES
- main.py: older commands (demo, memory-demo, evaluate, benchmark, failure-evaluate, reset).
- agents/memory_manager.py: MemoryManager (create_or_load, record_attempt, build_compact_context, compaction that keeps the NEWEST summary text). I added _shorten_middle and build_bounded_context(state, budget_chars=2000, ...): keeps goal, status, pending count + next 3 ids, recent steps/errors, key facts (newest in full, older shortened to 48 chars keeping start+end); drops OLDEST facts if over budget and reports older_facts_dropped.
- agents/llm.py: MockLLM (offline, only knows what is in the prompt; tokens = chars/4), AnthropicLLM, OllamaLLM (/api/generate, num_ctx 8192, input tokens estimated from prompt, output tokens real), make_llm. OllamaLLM prints a "." per call.
- agents/llm_agents.py: make_code(i) gives unique codes like K3-28f3; make_plan (n-1 LLM steps + final report step with required_codes); OutputAuditor (non-final: needs "FACT-i" label, the step's code in the first 300 chars, >=80 chars; final: must contain ALL earlier codes; rejection reason names what is missing); fact_text() (what memory remembers: from the FACT label through the code); LLMBaselineAgent (full session log in every prompt, no persistence, retry prompt includes the rejection reason); LLMMemoryAgent (MemoryManager, build_bounded_context with ctx_budget, resumes from JSON); MAX_ATTEMPTS=3.
- solve.py: python solve.py "task" --steps N --interrupt-at K --llm mock|ollama|anthropic --model X --ctx-budget B. Prints a comparison table, saves data/solve-*_comparison.json. Memory file should go in a temp dir (I was told to apply this edit; unsure if saved).
- bench.py: python bench.py --sizes 10 20 40 80 --trials 3 --llm ... Averages trials, saves benchmark_*.json/.csv, measures Python peak RAM with tracemalloc. Uses temp dirs for per-trial memory files.
- From a DIFFERENT earlier chat (code I have not shown you): agents/workflow.py (dependencies + replanning, WorkflowAuditor, WorkflowBaselineAgent, WorkflowMemoryAgent), recovery_bench.py (several crashes + faulty/missing-evidence steps), workflow_bench.py, tests/test_workflow.py. Total tests: 26.

EXPERIMENT DESIGN
Each step must output "FACT-i: ... <unique code Ki-xxxx>". The final step must list every code. "Recall" = share of codes in the final report. Baseline and memory agent get the SAME plan and the SAME crash point (baseline restarts from step 1; memory agent reloads JSON and continues). A step counts as repeated if it is re-run after already being verified.

KEY RESULTS SO FAR
Mock model, 3 trials, crash at 50%:
- Baseline input tokens grow ~quadratically (16k at 10 steps, 270k at 40, 1.09M at 80); peak prompt 2.5k, 10.7k, 21.7k tokens. Repeated steps after crash = half the steps.
- Memory agent: 2k, 14.6k, 46k input tokens (budget 4000 at 80 steps); peak prompt 241, 557, 979; 0 repeated steps; 100% completion and recall.
- At 80 steps with the default 2000-char budget, the memory agent FAILS (0% completion, 44% recall): oldest facts get dropped and the strict final auditor rejects the report. Budget must be tuned to task length. Keep this in the report.
- Python RAM is NOT lower for the memory agent except at 80 steps.
Real model qwen2.5:3b, 6 steps, crash after 3, 3 trials (BEFORE the fact_text fix):
- Baseline: 100% done, 11 calls, 3.67 repeats, 3829 input tokens (est.), peak prompt 943.
- Memory: 67% done (2/3), recall 80%, 9 calls, 0 repeats, 2636 input tokens (est.), peak prompt 389.
- So memory was more efficient but LESS reliable in this tiny sample. Cause unconfirmed.
Lessons found:
1. An earlier single real run showed baseline 0% because baseline retries were blind while the memory agent's context carried the rejection reason. Unfair; fixed (baseline retry now includes the reason).
2. Memory saved only the first line with "FACT-", so a code on line 2 was verified but forgotten. Demonstrated with a stub model (memory 0/5 recall); fixed with fact_text(). Applied and confirmed with Select-String. Not yet re-run on the real model.

CURRENT STATE (where I stopped)
- python -m unittest discover -s tests: 26 tests, 2 FAIL: test_baseline_replans_too and test_replan_on_completes_with_alternative_step (both "completed" is False after a replan).
- Likely cause: in agents/workflow.py, class WorkflowAuditor is defined at line 13, but the two places that create the auditor still use OutputAuditor(): `auditor = OutputAuditor()` in WorkflowBaselineAgent.run and `auditor, done, interrupted = OutputAuditor(), 0, False` in WorkflowMemoryAgent.run. They must become WorkflowAuditor(). My last check showed only 1 match for "WorkflowAuditor" (expected 3). I was about to run: Select-String -Path agents\workflow.py -Pattern "OutputAuditor"
NEXT STEPS
1. Fix the two lines above, rerun tests (expect 26 OK).
2. Rerun mock bench: python bench.py --sizes 10 40 --trials 1 (expect memory input tokens about 2032 at 10 steps and 14399 at 40).
3. Rerun real model: python bench.py --sizes 6 --trials 3 --llm ollama --model qwen2.5:3b (about 15 min) and read the result honestly.
4. Then decide: more trials / a longer real task / tune ctx-budget.
DEFERRED: Hatchable dashboard (separate hosted app with its own simulation, does not read my JSON), and the final write-up (architecture, memory format, compaction, tests, setup, results, limitations).

REPORT CAVEATS TO KEEP
Mock results show the mechanism, not LLM accuracy; mock and Ollama input tokens are estimated (chars/4), output tokens and time from Ollama are real; real-model sample is small (3B model, few trials); the baseline has no truncation or summarisation of its own, so its token numbers are an upper bound.
