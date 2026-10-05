"""Offline stand-ins for OpenRouter (agent model + judge) and the corpus retriever.

Used only by test_failure_modes.py to prove the harness's exit codes without network or keys.
One HTTP server, three roles:
  /api/v1/chat/completions   scripted agent model; behaviour picked by a MODE=<name> marker in the task
  /judge/chat/completions    scripted judge; unparseable verdict when the question contains JUDGE=garbage
  /retriever/{health,search,get_document}   canned corpus; a query containing FAIL returns HTTP 500
"""
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STEPS = {}
LOCK = threading.Lock()


def fill(schema, defs=None):
    """Minimal instance of a JSON schema, for the agent's non-ThinkOutput structured calls (memory)."""
    defs = defs or schema.get("$defs", {})
    if "$ref" in schema:
        return fill(defs[schema["$ref"].split("/")[-1]], defs)
    if "anyOf" in schema:
        return fill(schema["anyOf"][0], defs)
    t = schema.get("type")
    if t == "object":
        return {k: fill(v, defs) for k, v in schema.get("properties", {}).items()}
    if t == "array":
        return []
    if t in ("integer", "number"):
        return 0
    if t == "boolean":
        return False
    return "x"


def think(actions, note="scripted"):
    return {"thinking": note, "evaluation_previous_goal": note, "memory": note, "next_goal": note,
            "actions": [{"type": "tool", "name": n, "args": json.dumps(a)} for n, a in actions]}


def agent_reply(mode, step):
    if mode == "answer":
        return think([("search", {"query": "capital"})]) if step == 0 else \
            think([("done", {"result": "Explanation: found [d1].\nExact Answer: 42\nConfidence: 90%", "reasoning": "doc d1"})])
    if mode == "never_done":
        return think([("bash", {"command": "echo still-working"})])
    if mode == "empty_answer":
        return think([("done", {"result": "", "reasoning": ""})])
    if mode == "tool_error":
        return think([("search", {"query": "FAIL please"})]) if step == 0 else \
            think([("done", {"result": "Exact Answer: 42", "reasoning": "r"})])
    raise ValueError(mode)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def completion(self, content, model):
        return {"id": "fake", "object": "chat.completion", "created": 0, "model": model,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110, "cost": 0.001}}

    def do_GET(self):
        if self.path == "/retriever/health":
            return self.send(200, {"ok": True, "k": 5, "fake": True})
        self.send(404, {})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/retriever/search":
            if "FAIL" in req.get("query", ""):
                return self.send(500, {"error": "injected retriever failure"})
            return self.send(200, {"results": [{"docid": "d1", "score": 1.0, "snippet": "The answer is 42."}]})
        if self.path == "/retriever/get_document":
            return self.send(200, {"document": {"docid": req["docid"], "text": "The answer is 42."}})
        if self.path == "/judge/chat/completions":
            if "JUDGE=garbage" in json.dumps(req.get("messages", [])):
                return self.send(200, self.completion("I cannot decide.", req["model"]))
            return self.send(200, self.completion("extracted_final_answer: 42\nreasoning: match\ncorrect: yes\nconfidence: 90", req["model"]))
        if self.path.endswith("/chat/completions"):
            text = json.dumps(req.get("messages", []))
            m = re.search(r"MODE=(\w+)#(\w+)", text)
            mode, key = (m.group(1), m.group(0)) if m else ("none", "none")
            if mode == "fallback" and req["model"] == "google/gemini-3-flash-preview":
                return self.send(500, {"error": {"message": "injected primary model failure"}})
            schema = (req.get("response_format") or {}).get("json_schema", {}).get("schema", {})
            props = schema.get("properties", {})
            if "actions" in props and "next_goal" in props:
                with LOCK:
                    step = STEPS.get(key, 0)
                    STEPS[key] = step + 1
                body = agent_reply("answer" if mode == "fallback" else mode, step)
            else:
                body = fill(schema) if schema else "ok"
            return self.send(200, self.completion(json.dumps(body) if not isinstance(body, str) else body, req["model"]))
        self.send(404, {})


if __name__ == "__main__":
    port = int(sys.argv[1])
    print(f"fakes on {port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
