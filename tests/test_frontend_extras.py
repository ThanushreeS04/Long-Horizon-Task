import http.client
import json
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import frontend

ROOT = Path(frontend.__file__).resolve().parent


class FrontendExtraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frontend.Handler.log_message = lambda *a, **k: None
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), frontend.Handler)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def call(self, method, path, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=120)
        conn.request(method, path, body=json.dumps(body) if body is not None else None,
                     headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        data = r.read()
        conn.close()
        return r.status, data

    def test_charts_are_read_only_and_cannot_escape_the_folder(self):
        status, data = self.call("GET", "/charts-list")
        self.assertEqual(status, 200)
        self.assertIsInstance(json.loads(data), list)
        for bad in ("/charts/../frontend.py", "/charts/..%2Ffrontend.py", "/charts/missing.svg"):
            self.assertEqual(self.call("GET", bad)[0], 404)

    def test_stop_with_nothing_running_is_harmless_and_bad_agent_is_refused(self):
        status, data = self.call("POST", "/stop", {})
        self.assertEqual((status, json.loads(data)["stopped"]), (200, False))
        status, _ = self.call("POST", "/run", {"task": "Plan a launch", "steps": 3,
                                                "backend": "mock", "agents": "everything"})
        self.assertEqual(status, 400)

    def test_agent_picker_runs_only_the_chosen_agent(self):
        before = set((ROOT / "data").glob("*_comparison.json"))
        out_dir = ROOT / "outputs"
        before_out = set(out_dir.glob("*")) if out_dir.exists() else set()
        try:
            status, data = self.call("POST", "/run", {"task": "Plan a product launch", "steps": 3,
                                                       "backend": "mock", "agents": "memory"})
            self.assertEqual(status, 200)
            self.assertEqual(list(json.loads(data)["results"]), ["memory"])
        finally:
            for p in set((ROOT / "data").glob("*_comparison.json")) - before:
                p.unlink()
            if out_dir.exists():
                for p in set(out_dir.glob("*")) - before_out:
                    p.unlink()


if __name__ == "__main__":
    unittest.main()