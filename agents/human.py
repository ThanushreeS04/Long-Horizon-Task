"""Human-in-the-loop ASK route: how a question from the Manager reaches a person."""


def cli_ask(question, input_fn=input, out=print):
    """Ask on the terminal. A blank answer (or Ctrl-D) returns None, which stops the run
    with status 'ask' and the state saved, so it can be resumed later."""
    out(f"\n[ASK] The manager needs your input:\n  {question}")
    try:
        answer = input_fn("Your answer (blank = stop here): ")
    except (EOFError, KeyboardInterrupt):
        return None
    return answer.strip() or None


def auto_ask(question):
    """Non-interactive stand-in for a person (demos, tests, unattended runs)."""
    return "No preference. Choose a sensible default and state the assumption you made."


def make_ask_fn(mode):
    """'cli' = terminal prompt, 'auto' = canned answer, 'off' = ASK stops the run."""
    return {"cli": cli_ask, "auto": auto_ask, "off": None}[mode]
