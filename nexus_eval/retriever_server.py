"""BrowseComp-Plus retriever served over local HTTP for the Nexus Research v0 harness.

Runs in its own Python environment (pyserini + Java 21), separate from the agent's,
so the agent environment stays exactly as baselined. Search behaviour reproduces
BrowseComp-Plus searcher/tools.py at the pinned commit: BM25 top-k, snippets cut to
`snippet_max_tokens` tokens of the Qwen3-0.6B tokenizer, plus get_document.

Usage:
  python retriever_server.py --bcp-dir <BrowseComp-Plus checkout> --index-path <indexes/bm25> \
      --port 8765 --log <requests.jsonl>
"""
import argparse
import importlib.util
import json
import os
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def load_bm25_searcher_class(bcp_dir: str):
    """Load BM25Searcher from the pinned checkout without importing the package
    __init__, which pulls in the GPU/faiss searchers."""
    pkg_dir = os.path.join(bcp_dir, "searcher", "searchers")
    pkg = types.ModuleType("bcp_searchers")
    pkg.__path__ = [pkg_dir]
    sys.modules["bcp_searchers"] = pkg
    for name in ("base", "bm25_searcher"):
        spec = importlib.util.spec_from_file_location(
            f"bcp_searchers.{name}", os.path.join(pkg_dir, f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    return sys.modules["bcp_searchers.bm25_searcher"].BM25Searcher


class Retriever:
    def __init__(self, bcp_dir, index_path, k, snippet_max_tokens, tokenizer_name, log_path):
        BM25Searcher = load_bm25_searcher_class(bcp_dir)
        self.searcher = BM25Searcher(argparse.Namespace(index_path=index_path))
        self.k = k
        self.snippet_max_tokens = snippet_max_tokens
        self.tokenizer = None
        if snippet_max_tokens and snippet_max_tokens > 0:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.log_path = log_path
        self._lock = threading.Lock()

    def _log(self, record):
        if not self.log_path:
            return
        with self._lock, open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def search(self, query: str):
        candidates = self.searcher.search(query, self.k)
        results = []
        for cand in candidates:
            text = cand["text"]
            if self.tokenizer is not None:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > self.snippet_max_tokens:
                    text = self.tokenizer.decode(tokens[: self.snippet_max_tokens], skip_special_tokens=True)
            item = {"docid": cand["docid"], "snippet": text}
            if cand.get("score") is not None:
                item["score"] = cand["score"]
            results.append(item)
        return results

    def get_document(self, docid: str):
        return self.searcher.get_document(docid)


def make_handler(retriever: Retriever):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "k": retriever.k, "snippet_max_tokens": retriever.snippet_max_tokens})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            started = time.time()
            try:
                length = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/search":
                    out = {"results": retriever.search(str(req["query"]))}
                elif self.path == "/get_document":
                    out = {"document": retriever.get_document(str(req["docid"]))}
                else:
                    self._send(404, {"error": "not found"})
                    return
                retriever._log({"path": self.path, "request": req, "ok": True,
                                "docids": [r["docid"] for r in out.get("results", [])] or
                                          ([out["document"]["docid"]] if out.get("document") else []),
                                "seconds": round(time.time() - started, 3)})
                self._send(200, out)
            except Exception as exc:  # report, never hide
                retriever._log({"path": self.path, "ok": False, "error": repr(exc)})
                self._send(500, {"error": repr(exc)})

    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bcp-dir", required=True)
    p.add_argument("--index-path", required=True)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--snippet-max-tokens", type=int, default=512)
    p.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    p.add_argument("--log", default=None)
    args = p.parse_args()
    retriever = Retriever(args.bcp_dir, args.index_path, args.k, args.snippet_max_tokens, args.tokenizer, args.log)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(retriever))
    print(f"retriever ready on 127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
