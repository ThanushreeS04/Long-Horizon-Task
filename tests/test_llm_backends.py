import contextlib
import io
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

from agents.llm import AnthropicLLM, MockLLM, OllamaLLM, estimate_tokens, make_llm


class FakeOllama(BaseHTTPRequestHandler):
    seen = []
    status = 200

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.seen.append((self.path, body))
        payload = json.dumps({"response": "FACT-1: hello", "eval_count": 7}).encode()
        if FakeOllama.status != 200:
            payload = b"model not found"
        self.send_response(FakeOllama.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


class OllamaTests(unittest.TestCase):
    def setUp(self):
        FakeOllama.seen, FakeOllama.status = [], 200
        self.server = HTTPServer(("127.0.0.1", 0), FakeOllama)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.host = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def call(self, llm, prompt="hello there", **kw):
        with contextlib.redirect_stdout(io.StringIO()):
            return llm.complete(prompt, **kw)

    def test_returns_text_and_real_output_tokens(self):
        r = self.call(OllamaLLM("m", host=self.host))
        self.assertEqual(r["text"], "FACT-1: hello")
        self.assertEqual(r["output_tokens"], 7)

    def test_input_tokens_are_estimated_from_the_prompt(self):
        r = self.call(OllamaLLM("m", host=self.host), prompt="x" * 400)
        self.assertEqual(r["input_tokens"], estimate_tokens("x" * 400))

    def test_request_carries_model_and_token_limit(self):
        self.call(OllamaLLM("qwen-test", host=self.host), max_tokens=123)
        path, body = FakeOllama.seen[0]
        self.assertEqual(path, "/api/generate")
        self.assertEqual((body["model"], body["options"]["num_predict"]), ("qwen-test", 123))
        self.assertFalse(body["stream"])

    def test_http_error_becomes_runtime_error(self):
        FakeOllama.status = 404
        with self.assertRaises(RuntimeError) as cm:
            self.call(OllamaLLM("m", host=self.host))
        self.assertIn("404", str(cm.exception))

    def test_unreachable_server_gives_a_helpful_error(self):
        with self.assertRaises(RuntimeError) as cm:
            self.call(OllamaLLM("m", host="http://127.0.0.1:1"))
        self.assertIn("Cannot reach Ollama", str(cm.exception))

    def test_host_without_scheme_gets_http(self):
        self.assertEqual(OllamaLLM("m", host="localhost:11434").host, "http://localhost:11434")

    def test_missing_model_name_fails_clearly(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                OllamaLLM()

    def test_model_can_come_from_environment(self):
        with mock.patch.dict(os.environ, {"LH_MODEL": "from-env"}, clear=True):
            self.assertEqual(OllamaLLM().model, "from-env")


class OtherBackendTests(unittest.TestCase):
    def test_anthropic_without_key_fails_clearly(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                AnthropicLLM()

    def test_make_llm_unknown_backend(self):
        with self.assertRaises(ValueError):
            make_llm("nonsense")

    def test_make_llm_mock(self):
        self.assertIsInstance(make_llm("mock"), MockLLM)

    def test_mock_final_report_lists_facts_seen_in_prompt(self):
        prompt = 'memory: "FACT-1: alpha K1-aaaa" "FACT-2: beta K2-bbbb"'
        text = MockLLM().complete(prompt, kind="final", meta={})["text"]
        self.assertIn("K1-aaaa", text)
        self.assertIn("K2-bbbb", text)

    def test_mock_is_deterministic(self):
        meta = {"index": 3, "code": "K3-x"}
        a = MockLLM().complete("p", meta=meta)
        b = MockLLM().complete("p", meta=meta)
        self.assertEqual(a, b)

    def test_estimate_tokens_is_chars_over_four_with_floor(self):
        self.assertEqual((estimate_tokens("a" * 40), estimate_tokens("")), (10, 1))


if __name__ == "__main__":
    unittest.main()