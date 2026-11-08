"""Turn a benchmark JSON (bench.py or mea_bench.py output) into a markdown table.

Usage:
  python report_tables.py data\\results\\benchmark_20261107-141021.json
  python report_tables.py data\\meabench_20261107-141608.json --out tables.md
"""
import argparse
import json
from pathlib import Path

COLUMNS = [("steps", "Steps"), ("agent", "Agent"), ("trials", "Trials"),
           ("completion_pct", "Done %"), ("unique_verified", "Verified"),
           ("llm_calls", "LLM calls"), ("repeated_steps", "Repeated"),
           ("rejected_attempts", "Rejected"), ("replans", "Replans"),
           ("replan_events", "Replans"), ("blocked_steps", "Blocked"),
           ("input_tokens", "Input tokens"), ("output_tokens", "Output tokens"),
           ("peak_prompt_tokens", "Peak prompt"), ("recall_pct", "Recall %"),
           ("memory_seconds", "Memory time (s)"), ("elapsed_seconds", "Total time (s)"),
           ("peak_python_kb", "Peak Python KB")]


def fmt(v):
    if isinstance(v, float):
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:g}"
    return f"{v:,}" if isinstance(v, int) else str(v)


def table(summary):
    cols = [(k, label) for k, label in COLUMNS if any(k in row for row in summary)]
    lines = ["| " + " | ".join(label for _, label in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for row in summary:
        lines.append("| " + " | ".join(fmt(row.get(k, "")) for k, _ in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    data = json.loads(Path(args.file).read_text(encoding="utf-8"))
    s = data.get("settings", {})
    header = (f"Backend: {data.get('backend')}"
              + (f" / {data['model']}" if data.get("model") else "")
              + f". Trials per row: {s.get('trials', '?')}. "
              + "Input tokens are estimated (characters / 4); time is measured.")
    text = header + "\n\n" + table(data["summary"]) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Saved: {args.out}")
    print(text)


if __name__ == "__main__":
    main()