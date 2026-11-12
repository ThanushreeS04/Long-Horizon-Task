"""Skip tests that need a loopback socket when the environment forbids it.

Some sandboxes (antivirus, locked-down Windows, containers) refuse bind() on
127.0.0.1 with e.g. WinError 10013. Those tests exercise HTTP plumbing only,
not the agent logic, so they skip with a clear reason instead of erroring.
"""
import socket
import unittest


def _loopback_works():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            s.listen(1)
        return True
    except OSError:
        return False


LOOPBACK_OK = _loopback_works()
needs_loopback = unittest.skipUnless(
    LOOPBACK_OK, "loopback sockets are blocked in this environment (e.g. WinError 10013)")
