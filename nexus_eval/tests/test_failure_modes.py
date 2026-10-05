"""Prove the harness never reports success for a failed run. Offline: scripted model, judge and
retriever from fakes.py; the real agent code runs unchanged.

  python nexus_eval/tests/test_failure_modes.py --bcp-dir <BrowseComp-Plus checkout> --standin <stand-in dir>

Exits non-zero if any case's exit code or task statuses differ from what is expected.
"""
import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bcp-dir", required=True)
    p.add_argument("--standin", required=True)
    a = p.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="nexus-failmodes-"))
    port = free_port()
    fakes = subprocess.Popen([PY, str(ROOT / "nexus_eval/tests/fakes.py"), str(port)], stdout=subprocess.PIPE, text=True)
    fakes.stdout.readline()

    modes = ["answer", "never_done", "empty_answer", "tool_error", "fallback", "answer2", "answer3"]
    rows = [{"query_id": f"t-{m}", "query": f"What is the number? MODE={m.rstrip('23') if m.startswith('answer') else m}#{m}",
             "answer": "42"} for m in modes]
    rows[modes.index("answer3")]["query"] += " JUDGE=garbage"
    data = tmp / "data.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    src_sha = hashlib.sha256(data.read_bytes()).hexdigest()
    by_id = {r["query_id"]: r for r in rows}

    def subset(name, ids):
        f = tmp / f"{name}.json"
        f.write_text(json.dumps({"name": name, "source_sha256": src_sha, "tasks": [
            {"query_id": q, "query_sha256": hashlib.sha256(by_id[q]["query"].encode()).hexdigest() if q in by_id else "",
             "answer_sha256": ""} for q in ids]}))
        return f

    base_env = {k: os.environ[k] for k in ("PATH", "HOME") if k in os.environ}
    base_env.update(OPENROUTER_API_KEY="offline-fake-key", OPENROUTER_API_BASE=f"http://127.0.0.1:{port}/api/v1",
                    NEXUS_RETRIEVER_URL=f"http://127.0.0.1:{port}/retriever",
                    NEXUS_JUDGE_ENDPOINT=f"http://127.0.0.1:{port}/judge/chat/completions",
                    NEXUS_BCP_DIR=a.bcp_dir, NEXUS_STANDIN_DIR=a.standin, NO_PROXY="127.0.0.1,localhost",
                    no_proxy="127.0.0.1,localhost", PYTHONUNBUFFERED="1")

    tampered = tmp / "tampered-standin"
    shutil.copytree(a.standin, tampered)
    (tampered / "libs/BinanceDatabase/src/core/time_utils.py").write_text("def utc_ms(*a, **k):\n    return 0\n")

    cases = [
        ("empty subset", subset("empty", []), {}, [], 2, None),
        ("unknown task id", subset("unknown", ["t-missing"]), {}, [], 2, None),
        ("retriever unreachable", subset("r", ["t-answer"]), {"NEXUS_RETRIEVER_URL": "http://127.0.0.1:9"}, [], 2, None),
        ("stand-in hash mismatch", subset("h", ["t-answer"]), {"NEXUS_STANDIN_DIR": str(tampered)}, [], 2, None),
        # t-fallback: the pinned model returns HTTP 500 on every call. The inherited OpenRouter client turns API
        # errors into failed responses instead of raising, so the manager's fallback never fires; it is a step error.
        ("mixed task failures", subset("mixed", ["t-answer", "t-never_done", "t-empty_answer", "t-tool_error", "t-fallback"]), {}, [], 3,
         {"t-answer": "completed", "t-never_done": "max_steps", "t-empty_answer": "no_answer",
          "t-tool_error": "tool_error", "t-fallback": "agent_step_error"}),
        ("task timeout", subset("timeout", ["t-answer2"]), {}, ["--task-timeout", "5"], 3, {"t-answer2": "timeout"}),
        ("judge unparseable", subset("judge", ["t-answer3"]), {}, [], 4, {"t-answer3": "completed"}),
        ("all good", subset("good", ["t-answer"]), {}, [], 0, {"t-answer": "completed"}),
    ]
    report, ok_all = [], True
    for name, sub, env_extra, extra_args, want_rc, want_status in cases:
        out = tmp / "runs" / name.replace(" ", "_")
        env = dict(base_env, **env_extra)
        started = time.time()
        r = subprocess.run([PY, "nexus_eval/run_bcp.py", "--subset", str(sub), "--data", str(data), "--out", str(out)] + extra_args,
                           cwd=ROOT, env=env, capture_output=True, text=True)
        got_status = {}
        if (out / "results").exists():
            for f in (out / "results").glob("*.json"):
                got_status[f.stem] = json.loads(f.read_text())["status"]
        ok = r.returncode == want_rc and (want_status is None or got_status == want_status)
        ok_all &= ok
        summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
        report.append({"case": name, "expected_exit": want_rc, "exit": r.returncode, "statuses": got_status,
                       "accuracy": summary.get("accuracy"), "ok": ok, "seconds": round(time.time() - started, 1),
                       "stderr_tail": r.stderr.strip().splitlines()[-1:] if r.stderr.strip() else []})
        print(f"{'PASS' if ok else 'FAIL'}  {name}: exit {r.returncode} (want {want_rc}) {got_status}", flush=True)
    fakes.terminate()
    (tmp / "report.json").write_text(json.dumps(report, indent=2))
    print(f"report: {tmp / 'report.json'}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
