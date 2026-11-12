import json
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from net_guard import needs_loopback

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.timeouts import LLMTimeout, post_json


def make_handler(delay):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            time.sleep(delay)
            try:
                body = json.dumps({"response": "ok"}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                pass

        def log_message(self, *a):
            pass
    return H


class Server:
    def __init__(self, delay):
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(delay))
        self.srv.daemon_threads = True
        self.url = "http://127.0.0.1:%d/api/generate" % self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


@needs_loopback
class TimeoutTests(unittest.TestCase):
    def test_slow_server_raises_llmtimeout_quickly(self):
        s = Server(delay=2.0)
        self.addCleanup(s.close)
        t0 = time.monotonic()
        with self.assertRaises(LLMTimeout):
            post_json(s.url, {"prompt": "x"}, timeout=0.3)
        self.assertLess(time.monotonic() - t0, 1.5)


if __name__ == "__main__":
    unittest.main()
