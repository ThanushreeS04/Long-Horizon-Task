
import json
import sys
from pathlib import Path

from agents.baseline_agent import BaselineAgent
from agents.memory_agent import MemoryEnabledAgent
from agents.executor import (
    BenchmarkEnvironment,
    DemoEnvironment,
    FailOnceEnvironment,
)
from agents.planner import sample_task
from agents.auditor import Auditor


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MEMORY = DATA / "task_memory.json"
RESULTS = DATA / "evaluation_results.json"
RECOVERY_RESULTS = DATA / "recovery_evaluation.json"
EVAL_MEMORY = DATA / "evaluation_memory.json"
LONG_BENCHMARK_RESULTS = DATA / "long_task_benchmark.json"
FAILURE_RESULTS = DATA / "failure_injection_evaluation.json"

def show(title, result):
    print(f"\n=== {title} ===")
    for key, value in result.items():
        print(f"{key}: {value}")


def demo():
    DATA.mkdir(exist_ok=True)
    task = sample_task()
    baseline_result = BaselineAgent(
        DemoEnvironment()
    ).run_task(task)

    if MEMORY.exists():
        MEMORY.unlink()

    memory_result = MemoryEnabledAgent(
        DemoEnvironment(), MEMORY
    ).run_task(task)

    output = {
        "task_id": task["task_id"],
        "goal": task["goal"],
        "baseline": baseline_result,
        "memory_enabled": memory_result,
        "note": (
            "Deterministic prototype; use multiple trials "
            "before drawing conclusions."
        ),
    }

    RESULTS.write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    show("BASELINE AGENT", baseline_result)
    show("MEMORY-ENABLED AGENT", memory_result)
    print(f"\nComparison saved to {RESULTS}")


def memory_demo():
    DATA.mkdir(exist_ok=True)

    if MEMORY.exists():
        MEMORY.unlink()

    task = sample_task()
    env = DemoEnvironment()

    first = MemoryEnabledAgent(
        env, MEMORY
    ).run_task(task, max_steps=2)
    show("FIRST SESSION (SIMULATED INTERRUPTION)", first)

    resumed = MemoryEnabledAgent(
        env, MEMORY
    ).resume_task(task["task_id"])
    show("RESUMED SESSION", resumed)

    print(f"\nInspect persisted state: {MEMORY}")


def count_repeated_steps(actions):
    """Count execution attempts beyond the first attempt per step."""
    counts = {}

    for action in actions:
        step_id = action["step_id"]
        counts[step_id] = counts.get(step_id, 0) + 1

    return sum(count - 1 for count in counts.values())


def run_recovery_trial(trial_number):
    """Compare both agents under the same interruption conditions."""
    task = sample_task()
    task["task_id"] = f"recovery-trial-{trial_number:03d}"

    # Baseline: interrupt, discard agent state, then restart.
    baseline_env = DemoEnvironment()

    baseline_first = BaselineAgent(
        baseline_env
    ).run_task(task, max_steps=2)

    # A new agent has no saved progress. The environment remains
    # available, just as the external world would after a restart.
    baseline_second = BaselineAgent(
        baseline_env
    ).run_task(task)

    baseline_attempts = (
        baseline_first["execution_steps"]
        + baseline_second["execution_steps"]
    )

    baseline = {
        "completed": baseline_second["completed"],
        "execution_attempts": baseline_attempts,
        "repeated_steps": count_repeated_steps(
            baseline_env.actions
        ),
        "verified_steps": baseline_second["verified_steps"],
        "failures": (
            baseline_first["failures"]
            + baseline_second["failures"]
        ),
        "elapsed_seconds": round(
            baseline_first["elapsed_seconds"]
            + baseline_second["elapsed_seconds"],
            6,
        ),
        "history_size_chars": (
            baseline_first["history_size_chars"]
            + baseline_second["history_size_chars"]
        ),
    }

    # Proposed agent: interrupt and resume from persisted JSON.
    if EVAL_MEMORY.exists():
        EVAL_MEMORY.unlink()

    memory_env = DemoEnvironment()

    memory_first = MemoryEnabledAgent(
        memory_env, EVAL_MEMORY
    ).run_task(task, max_steps=2)

    memory_second = MemoryEnabledAgent(
        memory_env, EVAL_MEMORY
    ).resume_task(task["task_id"])

    memory_manager = MemoryEnabledAgent(
        memory_env, EVAL_MEMORY
    ).memory
    saved_state = memory_manager.get_task(task["task_id"])

    memory_attempts = (
        memory_first["execution_steps_this_session"]
        + memory_second["execution_steps_this_session"]
    )

    memory = {
        "completed": memory_second["completed"],
        "execution_attempts": memory_attempts,
        "repeated_steps": count_repeated_steps(
            memory_env.actions
        ),
        "verified_steps": (
            len(task["steps"])
            - len(saved_state["pending_step_ids"])
        ),
        "failures": memory_second["failures_total"],
        "elapsed_seconds": round(
            memory_first["elapsed_seconds"]
            + memory_second["elapsed_seconds"],
            6,
        ),
        "compact_context_size_chars": (
            memory_second["compact_context_size_chars"]
        ),
        "memory_processing_seconds": round(
            memory_first["memory_processing_seconds"]
            + memory_second["memory_processing_seconds"],
            6,
        ),
    }

    return {
        "trial": trial_number,
        "task_steps": len(task["steps"]),
        "interruption_after_steps": 2,
        "baseline": baseline,
        "memory_enabled": memory,
    }


def evaluate():
    """Run repeated recovery comparisons and save the results."""
    DATA.mkdir(exist_ok=True)

    trials = [
        run_recovery_trial(number)
        for number in range(1, 6)
    ]

    summary = {}

    for agent_name in ("baseline", "memory_enabled"):
        agent_results = [trial[agent_name] for trial in trials]

        attempts = [
            result["execution_attempts"]
            for result in agent_results
        ]
        repeats = [
            result["repeated_steps"]
            for result in agent_results
        ]
        completion_rate = (
            sum(result["completed"] for result in agent_results)
            / len(agent_results)
        )

        summary[agent_name] = {
            "trials": len(agent_results),
            "completion_rate": round(completion_rate * 100, 2),
            "average_execution_attempts": round(
                sum(attempts) / len(attempts), 2
            ),
            "average_repeated_steps": round(
                sum(repeats) / len(repeats), 2
            ),
            "total_failures": sum(
                result["failures"] for result in agent_results
            ),
            "average_elapsed_seconds": round(
                sum(
                    result["elapsed_seconds"]
                    for result in agent_results
                ) / len(agent_results),
                6,
            ),
        }

    output = {
        "experiment": "Controlled interruption and recovery",
        "environment": "Deterministic DemoEnvironment",
        "interruption_after_steps": 2,
        "task_steps": 4,
        "trials": trials,
        "summary": summary,
        "limitations": [
            "Synthetic deterministic environment; not a real LLM benchmark.",
            "Elapsed times are local prototype timings and may be noisy.",
            "Context characters are not token counts or RAM measurements.",
            "Only the tested task and interruption conditions are evaluated.",
        ],
    }

    RECOVERY_RESULTS.write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )

    print("\n=== RECOVERY EVALUATION SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nDetailed results saved to: {RECOVERY_RESULTS}")


def create_long_task(step_count, trial_number):
    """Create a deterministic task with a configurable number of steps."""
    steps = []

    for number in range(1, step_count + 1):
        steps.append({
            "id": f"step-{number}",
            "description": f"Benchmark task step {number}",
            "operation": "benchmark_step",
            "expected": f"fact-step-{number}",
        })

    return {
        "task_id": f"long-{step_count}-trial-{trial_number:03d}",
        "goal": f"Complete and verify a {step_count}-step task.",
        "success_criteria": [
            f"Step {number} verified"
            for number in range(1, step_count + 1)
        ],
        "steps": steps,
    }


def run_long_task_trial(step_count, trial_number):
    """Compare baseline and memory agents after a midpoint interruption."""
    task = create_long_task(step_count, trial_number)
    interruption_point = step_count // 2

    # Baseline: restart without persistent agent progress.
    baseline_env = BenchmarkEnvironment()

    baseline_first = BaselineAgent(
        baseline_env
    ).run_task(task, max_steps=interruption_point)

    baseline_second = BaselineAgent(
        baseline_env
    ).run_task(task)

    baseline = {
        "completed": baseline_second["completed"],
        "execution_attempts": (
            baseline_first["execution_steps"]
            + baseline_second["execution_steps"]
        ),
        "repeated_steps": count_repeated_steps(
            baseline_env.actions
        ),
        "failures": (
            baseline_first["failures"]
            + baseline_second["failures"]
        ),
        "elapsed_seconds": round(
            baseline_first["elapsed_seconds"]
            + baseline_second["elapsed_seconds"],
            6,
        ),
        "history_size_chars": (
            baseline_first["history_size_chars"]
            + baseline_second["history_size_chars"]
        ),
    }

    # Memory-enabled agent: persist progress and resume.
    memory_dir = DATA / "benchmark_memory"
    memory_dir.mkdir(parents=True, exist_ok=True)

    memory_path = memory_dir / (
        f"memory_{step_count}_{trial_number}.json"
    )

    if memory_path.exists():
        memory_path.unlink()

    memory_env = BenchmarkEnvironment()

    memory_first = MemoryEnabledAgent(
        memory_env, memory_path
    ).run_task(task, max_steps=interruption_point)

    memory_second = MemoryEnabledAgent(
        memory_env, memory_path
    ).resume_task(task["task_id"])

    memory = {
        "completed": memory_second["completed"],
        "execution_attempts": (
            memory_first["execution_steps_this_session"]
            + memory_second["execution_steps_this_session"]
        ),
        "repeated_steps": count_repeated_steps(
            memory_env.actions
        ),
        "failures": memory_second["failures_total"],
        "elapsed_seconds": round(
            memory_first["elapsed_seconds"]
            + memory_second["elapsed_seconds"],
            6,
        ),
        "compact_context_size_chars": (
            memory_second["compact_context_size_chars"]
        ),
        "memory_processing_seconds": round(
            memory_first["memory_processing_seconds"]
            + memory_second["memory_processing_seconds"],
            6,
        ),
    }

    return {
        "task_steps": step_count,
        "trial": trial_number,
        "interruption_after_steps": interruption_point,
        "baseline": baseline,
        "memory_enabled": memory,
    }


def evaluate_long_tasks():
    """Evaluate recovery for 5-step, 10-step, and 20-step tasks."""
    DATA.mkdir(exist_ok=True)

    task_sizes = [5, 10, 20]
    trials_per_size = 5
    all_results = []
    summary = {}

    for step_count in task_sizes:
        size_results = [
            run_long_task_trial(step_count, trial_number)
            for trial_number in range(1, trials_per_size + 1)
        ]

        all_results.extend(size_results)
        summary[str(step_count)] = {}

        for agent_name in ("baseline", "memory_enabled"):
            results = [
                trial[agent_name]
                for trial in size_results
            ]

            count = len(results)
            summary[str(step_count)][agent_name] = {
                "trials": count,
                "completion_rate_percent": round(
                    100 * sum(
                        result["completed"] for result in results
                    ) / count,
                    2,
                ),
                "average_execution_attempts": round(
                    sum(
                        result["execution_attempts"]
                        for result in results
                    ) / count,
                    2,
                ),
                "average_repeated_steps": round(
                    sum(
                        result["repeated_steps"]
                        for result in results
                    ) / count,
                    2,
                ),
                "total_failures": sum(
                    result["failures"] for result in results
                ),
                "average_elapsed_seconds": round(
                    sum(
                        result["elapsed_seconds"]
                        for result in results
                    ) / count,
                    6,
                ),
            }

            if agent_name == "memory_enabled":
                summary[str(step_count)][agent_name][
                    "average_compact_context_chars"
                ] = round(
                    sum(
                        result["compact_context_size_chars"]
                        for result in results
                    ) / count,
                    2,
                )
                summary[str(step_count)][agent_name][
                    "average_memory_processing_seconds"
                ] = round(
                    sum(
                        result["memory_processing_seconds"]
                        for result in results
                    ) / count,
                    6,
                )

    output = {
        "experiment": "Long-task interruption and recovery",
        "environment": "Synthetic BenchmarkEnvironment",
        "task_sizes": task_sizes,
        "trials_per_size": trials_per_size,
        "results": all_results,
        "summary": summary,
        "limitations": [
            "The environment is synthetic and deterministic.",
            "The agents do not use an LLM in this benchmark.",
            "Character counts are not token counts or RAM measurements.",
            "Execution times depend on the local machine and file I/O.",
            "Results apply only to the tested task sizes and interruptions.",
        ],
    }

    LONG_BENCHMARK_RESULTS.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print("\n=== LONG-TASK BENCHMARK SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(
        "\nDetailed results saved to: "
        f"{LONG_BENCHMARK_RESULTS}"
    )


def reset():
    for path in (MEMORY, RESULTS, RECOVERY_RESULTS, EVAL_MEMORY):
        if path.exists():
            path.unlink()

    print("Generated JSON files removed.")



def run_failure_trial(agent_name, trial_number):
    """Run one controlled failure-recovery trial."""
    task = create_long_task(10, trial_number)
    target_step_id = "step-3"
    environment = FailOnceEnvironment(target_step_id)

    if agent_name == "baseline":
        first = BaselineAgent(environment).run_task(
            task,
            max_steps=3,
        )
        second = BaselineAgent(environment).run_task(task)

        attempts = (
            first["execution_steps"]
            + second["execution_steps"]
        )
        completed = second["completed"]
        failures = first["failures"] + second["failures"]

    else:
        memory_path = (
            DATA / "failure_memory" /
            f"trial_{trial_number}.json"
        )
        memory_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if memory_path.exists():
            memory_path.unlink()

        first_agent = MemoryEnabledAgent(
            environment,
            memory_path,
        )
        first = first_agent.run_task(task, max_steps=3)

        second_agent = MemoryEnabledAgent(
            environment,
            memory_path,
        )
        second = second_agent.resume_task(task["task_id"])

        attempts = (
            first["execution_steps_this_session"]
            + second["execution_steps_this_session"]
        )
        completed = second["completed"]
        failures = second["failures_total"]

    
    target_step_attempts = sum(
        1
        for action in environment.actions
        if action["step_id"] == target_step_id
    )

    target_fact = f"fact-{target_step_id}"

    return {
        "agent": agent_name,
        "trial": trial_number,
        "target_step": target_step_id,
        "completed": completed,
        "execution_attempts": attempts,
        "repeated_steps": count_repeated_steps(
            environment.actions
        ),
        "failures": failures,
        "target_step_attempts": target_step_attempts,
        "failure_recovered": (
            completed
            and target_step_attempts >= 2
            and environment.facts.get(target_fact, False)
        ),
    }



def evaluate_failure_injection():
    """Compare both agents across repeated injected-failure trials."""
    DATA.mkdir(exist_ok=True)
    trials = []

    for trial_number in range(1, 6):
        trials.append({
            "trial": trial_number,
            "baseline": run_failure_trial(
                "baseline",
                trial_number,
            ),
            "memory_enabled": run_failure_trial(
                "memory_enabled",
                trial_number,
            ),
        })

    summary = {}

    for agent_name in ("baseline", "memory_enabled"):
        results = [
            trial[agent_name] for trial in trials
        ]
        total = len(results)

        summary[agent_name] = {
            "trials": total,
            "completion_rate_percent": round(
                100 * sum(
                    result["completed"] for result in results
                ) / total,
                2,
            ),
            "average_execution_attempts": round(
                sum(
                    result["execution_attempts"]
                    for result in results
                ) / total,
                2,
            ),
            "average_repeated_steps": round(
                sum(
                    result["repeated_steps"]
                    for result in results
                ) / total,
                2,
            ),
            "total_failures": sum(
                result["failures"] for result in results
            ),

            
            "failure_recovery_rate_percent": round(
                100 * sum(
                    result["failure_recovered"]
                    for result in results
                ) / total,
                2,
            ),
            "average_target_step_attempts": round(
                sum(
                    result["target_step_attempts"]
                    for result in results
                ) / total,
                2,
            ),

        }

    output = {
        "experiment": "Injected execution failure and recovery",
        "environment": "Synthetic FailOnceEnvironment",
        "task_steps": 10,
        "failure_step": "step-3",
        "trials": trials,
        "summary": summary,
        "limitations": [
            "Synthetic deterministic environment; no LLM is used.",
            "Only one selected step is made to fail once.",
            "Results do not establish real-world reliability.",
        ],
    }

    FAILURE_RESULTS.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print("\n=== FAILURE-INJECTION SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nDetailed results saved to: {FAILURE_RESULTS}")




if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "demo"

    if command == "demo":
        demo()
    elif command == "memory-demo":
        memory_demo()
    elif command == "evaluate":
        evaluate()
    elif command == "benchmark":
        evaluate_long_tasks()
    elif command == "failure-evaluate":
        evaluate_failure_injection()
    elif command == "reset":
        reset()
    else:
        raise SystemExit(
            "Usage: python main.py [demo|memory-demo|evaluate|reset]"
        )
