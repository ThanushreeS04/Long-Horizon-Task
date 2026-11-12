"""Ask a question, get a finished answer. No codes, no typing of extra input, just the answer.

    python chat.py                                        (offline mock model: for testing only)
    python chat.py --llm anthropic --model claude-sonnet-5-5
    python chat.py --llm ollama --model qwen2.5:3b --steps 5

Every line you type runs the full Plan -> Manage -> Execute -> Audit pipeline and prints only the
final answer. Extra options (anything agentic_run.py accepts) are passed through."""
import sys

import agentic_run


def main():
    extra = sys.argv[1:]
    print("Ask me anything (blank line to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        agentic_run.main([question, "--quiet", *extra])


if __name__ == "__main__":
    main()
