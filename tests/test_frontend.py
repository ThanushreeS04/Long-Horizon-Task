import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

import frontend

ROOT = Path(frontend.__file__).resolve().parent
GOOD = {"task": "Plan a product launch", "steps": 3, "backend": "mock"}


class FrontendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frontend.Handler.log_message = lambda *a, **k: None      # keep test output quiet
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), frontend.Handler)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def post(self, body, headers=None, path="/run"):
        h = {"Content-Type": "application/json"}
        h.update(headers or {})
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=120)
        conn.request("POST", path, body=json.dumps(body) if not isinstance(body, str) else body,
                     headers=h)
        r = conn.getresponse()
        data = json.loads(r.read() or b"{}")
        conn.close()
        return r.status, data

    def test_page_loads(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", "/")
        r = conn.getresponse()
        self.assertEqual(r.status, 200)
        self.assertIn(b"Long-Horizon Agent", r.read())

    def test_request_from_another_origin_is_refused(self):
        self.assertEqual(self.post(GOOD, {"Origin": "http://evil.example"})[0], 403)

    def test_request_with_a_foreign_host_name_is_refused(self):
        self.assertEqual(self.post(GOOD, {"Host": "evil.example:80"})[0], 403)

    def test_plain_text_body_is_refused(self):
        self.assertEqual(self.post(json.dumps(GOOD), {"Content-Type": "text/plain"})[0], 415)

    def test_task_that_looks_like_an_option_is_refused(self):
        self.assertEqual(self.post(dict(GOOD, task="--help"))[0], 400)

    def test_bad_steps_and_backend_are_refused(self):
        self.assertEqual(self.post(dict(GOOD, steps=2))[0], 400)
        self.assertEqual(self.post(dict(GOOD, steps=31))[0], 400)
        self.assertEqual(self.post(dict(GOOD, backend="anthropic"))[0], 400)

    def test_mock_run_returns_all_three_agents_with_the_fields_the_page_reads(self):
        before_data = set((ROOT / "data").glob("*_comparison.json"))
        before_out = set((ROOT / "outputs").glob("*")) if (ROOT / "outputs").exists() else set()
        try:
            status, data = self.post(GOOD, {"Origin": f"http://127.0.0.1:{self.port}"})
            self.assertEqual(status, 200)
            self.assertEqual(list(data["results"]), ["baseline", "memory", "mea"])
            for agent in data["results"].values():
                for key in ("completed", "input_tokens", "verified_steps", "repeated_steps",
                            "elapsed_seconds", "final_output"):
                    self.assertIn(key, agent)
            self.assertEqual(data["backend"], "mock")
        finally:                                  # leave the project folders as they were
            for p in set((ROOT / "data").glob("*_comparison.json")) - before_data:
                p.unlink()
            if (ROOT / "outputs").exists():
                for p in set((ROOT / "outputs").glob("*")) - before_out:
                    p.unlink()


if __name__ == "__main__":
    unittest.main()
