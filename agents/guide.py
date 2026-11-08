"""Turns the agents' verified step outputs into a readable how-to guide file."""
import re
from datetime import datetime

from .llm_agents import split_fact


def slugify(text, limit=50):
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug[:limit].strip("-")) or "task"


def build_guide(goal, steps, outputs, agent_label, model, completed, stats_line):
    """Markdown text: one section per step, key points, and a one-line run summary."""
    when = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# How to: {goal}", "",
             f"*Written by the {agent_label} ({model}) on {when}.*", ""]
    key_points, number = [], 0
    for step in steps:
        if step.get("final"):
            continue
        number += 1
        lines += [f"## Step {number}: {step['description']}", ""]
        out = outputs.get(step["id"])
        if out is None:
            lines += ["*This step was not completed.*", ""]
            continue
        fact, body = split_fact(out, step)
        if body:
            lines += [body, ""]
        if fact:
            lines += [f"**Key point:** {fact}", ""]
            key_points.append(f"- {fact}")
    if key_points:
        lines += ["## Summary of key points", ""] + key_points + [""]
    if not completed:
        lines += ["> Note: the run did not finish, so this guide may be incomplete.", ""]
    lines += ["---", stats_line]
    return "\n".join(lines) + "\n"


def write_guide(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path