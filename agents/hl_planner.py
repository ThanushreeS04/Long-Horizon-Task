"""Planner: turns the user's query into high-level steps (Plan-and-Act, Sec. 3.1),
with an optional chain-of-thought (Sec. 3.4). Falls back to the plain one-shot
planner if the model's answer cannot be parsed."""
from .cot import extract_json, split_think
from .llm_agents import make_code, make_plan, parse_steps

PROMPT = """You are the PLANNER. Turn the user's goal into exactly {count} HIGH-LEVEL steps.
A high-level step says WHAT to achieve (for example "Identify the target audience"),
not low-level actions. Steps must be in a sensible order and each should build on the earlier ones.

Goal: {goal}

{cot}Return ONLY a JSON array of {count} short strings."""

COT = "First think briefly inside <think>...</think> about what the goal needs and in what order. Then answer.\n"


def steps_from_descriptions(descs):
    """Same step structure make_plan builds; the last step is the final report."""
    descs = list(descs) + ["Compile the final report"]
    return [{"id": f"step-{i}", "description": d, "operation": "llm_step",
             "expected": f"FACT-{i}", "code": make_code(i), "final": i == len(descs),
             "required_codes": ([make_code(j) for j in range(1, len(descs))]
                                if i == len(descs) else [])}
            for i, d in enumerate(descs, start=1)]


def _as_texts(obj):
    if isinstance(obj, dict):
        obj = obj.get("steps")
    if not isinstance(obj, list):
        return []
    out = []
    for x in obj:
        if isinstance(x, dict):
            x = x.get("step") or x.get("description") or x.get("title") or ""
        x = " ".join(str(x).split())
        if x:
            out.append(x[:200])
    return out


class HighLevelPlanner:
    def __init__(self, adapter, cot=True, trained_prompt=False):
        self.adapter, self.cot, self.trained = adapter, cot, trained_prompt   # trained = LoRA planner

    def plan(self, goal, n_steps):
        count = max(2, n_steps - 1)
        prompt = ("Create a high-level plan for: " + goal if self.trained       # same wording as the training data
                  else PROMPT.format(goal=goal, count=count, cot=COT if self.cot else ""))
        res = self.adapter.run_episode("planner", prompt, {"max_tokens": 900},
                                       kind="cot_plan" if self.cot else "plan",
                                       meta={"goal": goal, "n": count})
        reasoning, answer = split_think(res["text"])
        descs = _as_texts(extract_json(res["text"])) or parse_steps(answer, count)
        source = "model"
        if not descs:                                  # unparseable: use the one-shot planner
            steps, _ = make_plan(self.adapter.for_role("planner"), goal, count + 1)
            return {"steps": steps, "reasoning": reasoning, "source": "fallback"}
        descs = descs[:12 if self.trained else count]
        while not self.trained and len(descs) < count:
            descs.append(f"Continue working on: {goal}")
        return {"steps": steps_from_descriptions(descs), "reasoning": reasoning, "source": source}
