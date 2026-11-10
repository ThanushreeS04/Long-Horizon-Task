"""Draw report charts as SVG files from the saved results. Pure Python, no extra packages,
no model calls. Output: data/results/charts/*.svg (open in a browser, or insert into Word/PPT)."""
import json
import math
from pathlib import Path

RES = Path("data/results")
OUT = RES / "charts"
FILES = {
    "scaling": "benchmark_20261108-162916.json",       # mock, 10-80 steps
    "real8": "benchmark_20261108-163554.json",         # qwen2.5:3b, 8 steps
    "fault1": "meabench_20261108-162945.json",         # mock, 1 dead step
    "fault2": "meabench_20261108-163014.json",         # mock, 2 dead steps
    "memsize": "memory_size.json",
}
COLORS = ["#c0392b", "#2471a3", "#1e8449", "#b9770e", "#7d3c98"]
W, H, L, R, T, B = 760, 440, 80, 30, 50, 90


def load(key):
    path = RES / FILES[key]
    assert path.exists(), f"missing results file: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;")


def num(v):
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:g}"


def frame(title, ylabel, body, legend):
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Arial, sans-serif">',
             f'<rect width="{W}" height="{H}" fill="white"/>',
             f'<text x="{W/2}" y="26" text-anchor="middle" font-size="17" font-weight="bold">{esc(title)}</text>',
             f'<text x="18" y="{(T + H - B)/2}" font-size="12" text-anchor="middle" '
             f'transform="rotate(-90 18 {(T + H - B)/2})">{esc(ylabel)}</text>']
    parts += body
    x = L
    for i, name in enumerate(legend):
        parts.append(f'<rect x="{x}" y="{H-24}" width="12" height="12" fill="{COLORS[i % 5]}"/>')
        parts.append(f'<text x="{x+17}" y="{H-14}" font-size="12">{esc(name)}</text>')
        x += 30 + 7 * len(name)
    parts.append("</svg>")
    return "\n".join(parts)


def line_chart(title, ylabel, xs, series, log=False):
    """series: {name: [y per x]}; categorical x positions."""
    vals = [v for ys in series.values() for v in ys]
    lo, hi = (min(vals), max(vals)) if log else (0, max(vals))
    f = (lambda v: math.log10(max(v, 1))) if log else (lambda v: v)
    flo, fhi = (math.floor(f(lo)), math.ceil(f(hi))) if log else (0, hi * 1.1)
    px = lambda i: L + (W - L - R) * (i + 0.5) / len(xs)
    py = lambda v: T + (H - B - T) * (1 - (f(v) - flo) / (fhi - flo))
    body = [f'<line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" stroke="#333"/>',
            f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H-B}" stroke="#333"/>']
    ticks = [10 ** k for k in range(int(flo), int(fhi) + 1)] if log else [hi * 1.1 * k / 5 for k in range(6)]
    for tv in ticks:
        y = py(tv)
        body.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="#ddd"/>')
        body.append(f'<text x="{L-6}" y="{y+4:.1f}" font-size="11" text-anchor="end">{num(tv)}</text>')
    for i, x in enumerate(xs):
        body.append(f'<text x="{px(i):.1f}" y="{H-B+18}" font-size="12" text-anchor="middle">{esc(x)}</text>')
    body.append(f'<text x="{(L+W-R)/2}" y="{H-B+38}" font-size="12" text-anchor="middle">task steps</text>')
    for k, (name, ys) in enumerate(series.items()):
        pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(ys))
        c = COLORS[k % 5]
        body.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2.5"/>')
        for i, v in enumerate(ys):
            body.append(f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="4" fill="{c}"/>')
            body.append(f'<text x="{px(i):.1f}" y="{py(v)-8:.1f}" font-size="10" text-anchor="middle" fill="{c}">{num(v)}</text>')
    return frame(title, ylabel, body, list(series))


def bar_chart(title, ylabel, groups, series, ymax=None):
    """groups: x labels; series: {name: [value per group]} drawn as grouped bars."""
    top = ymax or max(v for ys in series.values() for v in ys) * 1.15 or 1
    gw = (W - L - R) / len(groups)
    bw = gw * 0.8 / len(series)
    py = lambda v: T + (H - B - T) * (1 - v / top)
    body = [f'<line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" stroke="#333"/>',
            f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H-B}" stroke="#333"/>']
    for k in range(6):
        tv = top * k / 5
        body.append(f'<line x1="{L}" y1="{py(tv):.1f}" x2="{W-R}" y2="{py(tv):.1f}" stroke="#ddd"/>')
        body.append(f'<text x="{L-6}" y="{py(tv)+4:.1f}" font-size="11" text-anchor="end">{num(tv)}</text>')
    for g, label in enumerate(groups):
        gx = L + g * gw + gw * 0.1
        body.append(f'<text x="{L+g*gw+gw/2:.1f}" y="{H-B+18}" font-size="12" text-anchor="middle">{esc(label)}</text>')
        for k, (name, ys) in enumerate(series.items()):
            v, x = ys[g], gx + k * bw
            body.append(f'<rect x="{x:.1f}" y="{py(v):.1f}" width="{bw-3:.1f}" height="{H-B-py(v):.1f}" fill="{COLORS[k % 5]}"/>')
            body.append(f'<text x="{x+(bw-3)/2:.1f}" y="{py(v)-5:.1f}" font-size="10" text-anchor="middle">{num(v)}</text>')
    return frame(title, ylabel, body, list(series))


def by_agent(summary, agents, field, steps=None):
    rows = {r["agent"]: r for r in summary if steps is None or r["steps"] == steps}
    return [rows[a][field] for a in agents]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    charts = {}

    d = load("scaling")["summary"]
    sizes = sorted({r["steps"] for r in d})
    charts["1_input_tokens_vs_steps.svg"] = line_chart(
        "Input tokens vs task length (mock model, crash at 50%, 3 trials, estimated chars/4)",
        "input tokens (log scale)", sizes,
        {a: [next(r["input_tokens"] for r in d if r["steps"] == n and r["agent"] == a) for n in sizes]
         for a in ("baseline", "memory", "mea")}, log=True)

    d = load("real8")["summary"]
    agents = [r["agent"] for r in d]
    charts["2_real_model_completion.svg"] = bar_chart(
        "Real qwen2.5:3b, 8 steps, 3 trials: completion %", "tasks completed (%)",
        agents, {"completion %": [r["completion_pct"] for r in d]}, ymax=125)
    charts["3_real_model_tokens.svg"] = bar_chart(
        "Real qwen2.5:3b, 8 steps, 3 trials: input tokens per run (estimated)", "input tokens",
        agents, {"input tokens": [r["input_tokens"] for r in d]})

    f1, f2 = load("fault1")["summary"], load("fault2")["summary"]
    ag = ["baseline", "memory+replan", "mea+replan"]
    charts["4_fault_injection_tokens.svg"] = bar_chart(
        "Fault injection, 40 steps (mock, 5 trials): input tokens, all runs completed",
        "input tokens", ["1 dead step", "2 dead steps"],
        {a: [by_agent(f1, [a], "input_tokens", 40)[0], by_agent(f2, [a], "input_tokens", 40)[0]] for a in ag})

    m = load("memsize")
    charts["5_memory_size.svg"] = line_chart(
        "Saved state vs context sent to the model (memory agent, mock, budget 4000 chars)",
        "characters", [r["steps"] for r in m],
        {"saved state": [r["state_chars"] for r in m], "context sent": [r["context_chars"] for r in m]})

    for name, svg in charts.items():
        (OUT / name).write_text(svg, encoding="utf-8")
        print("wrote", OUT / name)
    print(f"Done: {len(charts)} charts in {OUT}")


if __name__ == "__main__":
    main()