from datetime import datetime, timezone

def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

class DemoEnvironment:
    # Safe stand-in for repository tools; no real files are modified.
    def __init__(self):
        self.facts = {}
        self.actions = []

    def execute(self, step):
        outputs = {
            "inspect_repository": ("repository_inspected", "Repository structure inspected."),
            "create_outline": ("outline_created", "Documentation outline created."),
            "draft_setup": ("setup_drafted", "Setup instructions drafted."),
            "check_docs": ("docs_checked", "Documentation checks passed."),
        }
        if step["operation"] not in outputs:
            return {"success": False, "fact": None, "message": "Unsupported operation",
                    "evidence": {}, "timestamp": timestamp()}
        fact, message = outputs[step["operation"]]
        self.facts[fact] = True
        action = {"step_id": step["id"], "fact": fact, "timestamp": timestamp()}
        self.actions.append(action)
        return {"success": True, "fact": fact, "message": message,
                "evidence": {"observed": self.facts.get(fact, False),
                             "action_recorded": action in self.actions},
                "timestamp": action["timestamp"]}


    
class BenchmarkEnvironment:
    """Synthetic environment for evaluating tasks of different lengths."""

    def __init__(self):
        self.facts = {}
        self.actions = []

    def execute(self, step):
        expected_fact = step["expected"]

        self.facts[expected_fact] = True

        action = {
            "step_id": step["id"],
            "fact": expected_fact,
            "timestamp": timestamp(),
        }
        self.actions.append(action)

        return {
            "success": True,
            "fact": expected_fact,
            "message": f"Completed {step['description']}.",
            "evidence": {
                "observed": self.facts.get(expected_fact, False),
                "action_recorded": action in self.actions,
            },
            "timestamp": action["timestamp"],
        }


class FailOnceEnvironment(BenchmarkEnvironment):
    """Fail the first attempt for one selected step."""

    def __init__(self, target_step_id):
        super().__init__()
        self.target_step_id = target_step_id
        self.failed = False

    def execute(self, step):
        if (
            step["id"] == self.target_step_id
            and not self.failed
        ):
            self.failed = True
            self.actions.append({"step_id": step["id"]})

            return {
                "success": False,
                "message": "Injected temporary execution failure",
                "fact": None,
                "evidence": {"observed": False},
                "timestamp": "injected-failure",
            }

        return super().execute(step)
