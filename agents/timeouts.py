"""Interruptible HTTP JSON POST for LLM calls."""
import json
import socket
import urllib.error
import urllib.request


class LLMTimeout(RuntimeError):
    """Raised when the model server does not answer within the timeout."""


def _is_timeout(exc):
    return isinstance(exc, (socket.timeout, TimeoutError))


def post_json(url, payload, timeout):
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.URLError as exc:
        if _is_timeout(exc.reason):
            raise LLMTimeout("no response within %ss" % timeout) from exc
        raise
    except (socket.timeout, TimeoutError) as exc:
        raise LLMTimeout("no response within %ss" % timeout) from exc
    return json.loads(body.decode("utf-8"))