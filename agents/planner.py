from copy import deepcopy

def sample_task():
    return {
        "task_id": "repo-docs-001",
        "goal": "Prepare and verify a repository documentation update.",
        "success_criteria": ["repository inspected", "outline created",
                            "setup instructions drafted", "documentation checks passed"],
        "steps": [
            {"id": "step-1", "description": "Inspect repository structure",
             "operation": "inspect_repository", "expected": "repository_inspected"},
            {"id": "step-2", "description": "Create documentation outline",
             "operation": "create_outline", "expected": "outline_created"},
            {"id": "step-3", "description": "Draft setup instructions",
             "operation": "draft_setup", "expected": "setup_drafted"},
            {"id": "step-4", "description": "Run documentation checks",
             "operation": "check_docs", "expected": "docs_checked"},
        ],
    }

class Planner:
    # Deterministic checkpoint planner; replaceable by an LLM-backed planner later.
    def next_step(self, task, verified_ids):
        for step in task["steps"]:
            if step["id"] not in verified_ids:
                return deepcopy(step)
        return None
