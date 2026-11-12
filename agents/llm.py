"""LLM backends used by both agents.

MockLLM      - offline and deterministic. It only "knows" what is inside the
               prompt it receives, so it shows what each memory strategy
               makes available to a model. Tokens are estimated (chars / 4).
AnthropicLLM - real model through the Anthropic Messages API. Token counts
               come from the API's usage field. Needs ANTHROPIC_API_KEY.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request

from agents.timeouts import post_json


def estimate_tokens(text):
    return max(1, len(str(text)) // 4)


class MockLLM:
    name = "mock"

    FILLER = (
        "Reviewed the available inputs, compared alternatives, noted the "
        "constraints that apply, and recorded the intermediate result so "
        "that later steps can build on it. Verified the work against the "
        "stated goal and listed the remaining risks for follow-up. "
    )

    def complete(self, prompt, kind="step", max_tokens=500, meta=None):
        meta = meta or {}
        if kind == "plan":
            clauses = [c.strip() for c in re.split(r"[,;.]| and | then ", meta["goal"]) if c.strip()]
            clauses = clauses or [meta["goal"]]
            n = meta["n"]
            text = json.dumps([f"Phase {i + 1}: work on '{clauses[i % len(clauses)]}'" for i in range(n)])
        elif kind == "replan":
            text = json.dumps([d if d.endswith("(revised after audit)") else d + " (revised after audit)"
                               for d in meta["descriptions"]])
        elif kind == "grounded_plan":            # synthetic_pipeline.py (Plan-and-Act Sec. 4.2)
            n = max(1, int(meta.get("action_count", 1)))
            text = json.dumps({"rationale_summary": "Grouped recorded actions without inventing actions.",
                               "steps": [{"title": f"Recorded action group {i + 1}", "action_indices": [i]}
                                         for i in range(n)]})
        elif kind == "plan_expand":              # Sec. 4.3 plan expansion
            text = json.dumps({"query": meta.get("query", "Equivalent task phrasing"),
                               "steps": meta.get("steps", [])})
        elif kind == "targeted_augment":         # targeted plan augmentation
            idx = int(meta.get("target_index", 0))
            text = json.dumps({"query": meta.get("query", "Task") + f" with special attention to action {idx}",
                               "steps": [{"title": f"Focused action {idx}", "action_indices": [idx]}]})
        elif kind == "cot_plan":
            clauses = [c.strip() for c in re.split(r"[,;.]| and | then ", meta["goal"]) if c.strip()]
            clauses = clauses or [meta["goal"]]
            text = ("<think>Work out what the goal needs, then order the steps.</think>\n" +
                    json.dumps([f"Phase {i + 1}: work on '{clauses[i % len(clauses)]}'"
                                for i in range(meta["n"])]))
        elif kind == "manager":
            text = ('<think>The next step is ready and its dependencies are verified.</think>\n'
                    + json.dumps({"decision": "execute", "contract_goal": "", "extra_acceptance": [],
                                  "question": ""}))
        elif kind == "model_audit":
            text = ('<think>Compared the output with the goal.</think>\n'
                    '{"verdict": "pass", "reason": "The output carries out the step."}')
        elif kind == "state_replan":
            text = ("<think>The new facts do not change the remaining steps.</think>\n"
                    + json.dumps(meta["descriptions"]))
        elif kind == "audit":
            text = "YES. The output carries out the step."
        elif kind == "final":
            seen = {}
            for num, val in re.findall(r'FACT-(\d+): ([^"\\\n]*)', prompt):
                seen.setdefault(int(num), val.strip())
            lines = [f"FACT-{k}: {v}" for k, v in sorted(seen.items())]
            tail = "\nAll visible facts are listed above."
            if "lists every FACT-n line" not in prompt:    # natural mode: combine the earlier work shown as [step-n] text
                work = re.findall(r"^\[(step-\d+)\] (.+)$", prompt, re.M)
                lines = [f"{sid}: {' '.join(body.split())[:160]}" for sid, body in work]
                tail = "\nThe earlier verified work is combined above. " + self.FILLER
            text = (("<think>List every fact I was given.</think>\n" if "<think>" in prompt else "")
                    + "FINAL REPORT\n" + "\n".join(lines) + tail)
        else:
            i, code = meta["index"], meta["code"]
            think = "<think>Restate the contract, do the work, include the code.</think>\n" \
                if "<think>" in prompt else ""
            text = think + f"FACT-{i}: Step {i} established {code}.\n" + self.FILLER * 4
        return {"text": text,
                "input_tokens": estimate_tokens(prompt),
                "output_tokens": estimate_tokens(text)}


class AnthropicLLM:
    name = "anthropic"
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, model=None, api_key=None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set the ANTHROPIC_API_KEY environment variable first.")
        self.model = model or os.environ.get("LH_MODEL", "claude-haiku-5-5")

    def complete(self, prompt, kind="step", max_tokens=500, meta=None):
        body = json.dumps({"model": self.model, "max_tokens": max_tokens,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        headers = {"content-type": "application/json", "x-api-key": self.api_key,
                   "anthropic-version": "2023-06-01"}
        for attempt in range(4):
            try:
                req = urllib.request.Request(self.URL, data=body, headers=headers)
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read())
                break
            except urllib.error.HTTPError as exc:
                with exc:                                   # close the response so it isn't left open
                    detail = exc.read().decode(errors="replace")
                if exc.code in (429, 500, 529) and attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"API error {exc.code}: {detail}") from exc
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage", {})
        return {"text": text, "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0)}

class OllamaLLM:
    """Local model served by Ollama. No API key needed."""
    name = "ollama"

    def __init__(self, model=None, host=None):
        self.model = model or os.environ.get("LH_MODEL")
        if not self.model:
            raise RuntimeError("Give a model name with --model (see: ollama list).")
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.num_ctx = int(os.environ.get("LH_NUM_CTX", "8192"))
        self.timeout = float(os.environ.get("LH_TIMEOUT", "900"))

    def complete(self, prompt, kind="step", max_tokens=500, meta=None):
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False,
                           "options": {"num_predict": max_tokens, "num_ctx": self.num_ctx,
                                       "temperature": 0.2}}).encode()
        try:
            data = post_json(self.host + "/api/generate", body, self.timeout)
        except urllib.error.HTTPError as exc:
            with exc:                                       # close the response so it isn't left open
                detail = exc.read().decode(errors="replace")
            raise RuntimeError(f"Ollama error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cannot reach Ollama at {self.host}. Is it running? ({exc.reason})") from exc
        text = data.get("response", "")
        print(".", end="", flush=True)
        # Input tokens are estimated from the full prompt: Ollama reuses cached
        # prompt prefixes, which can make its own prompt count unreliable here.
        return {"text": text, "input_tokens": estimate_tokens(prompt),
                "output_tokens": data.get("eval_count") or estimate_tokens(text)}


def make_llm(name, model=None):
    if name == "mock":
        return MockLLM()
    if name == "ollama":
        return OllamaLLM(model=model)
    if name == "anthropic":
        return AnthropicLLM(model=model)
    raise ValueError(f"Unknown LLM backend: {name}")