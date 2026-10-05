"""Run the inherited agent on a frozen BrowseComp-Plus subset, then score it.

  python nexus_eval/run_bcp.py --subset nexus_eval/subsets/smoke-v0.json \
      --data <browsecomp_plus_decrypted.jsonl> --out <new run dir> [--task-timeout 3600]

Each task runs in its own process (nexus_eval/run_one.py) with a fresh workdir, so no tracer,
memory or version state carries over between tasks. Tasks run one at a time.
Requires a running retriever (NEXUS_RETRIEVER_URL) and the variables run_one.py needs.

Exit codes: 0 = all tasks completed and judged; 3 = some tasks failed (scored incorrect);
4 = judging failed; 2 = setup error. Never 0 unless every task produced a judged answer.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def fail(msg: str):
    print(f"SETUP ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--task-timeout", type=int, default=3600)
    a = p.parse_args()

    subset = json.loads(Path(a.subset).read_text())
    if not subset.get("tasks"):
        fail("subset has no tasks")
    data = {}
    for line in Path(a.data).read_text(encoding="utf-8").splitlines():
        if line.strip():
            o = json.loads(line)
            data[str(o["query_id"])] = o
    if not data:
        fail("dataset is empty")
    if hashlib.sha256(Path(a.data).read_bytes()).hexdigest() != subset["source_sha256"]:
        fail("dataset file does not match the subset's source_sha256")
    for t in subset["tasks"]:
        if t["query_id"] not in data:
            fail(f"query_id {t['query_id']} missing from dataset")
    for var in ("OPENROUTER_API_KEY", "NEXUS_BCP_DIR", "NEXUS_RETRIEVER_URL", "NEXUS_STANDIN_DIR"):
        if not os.environ.get(var):
            fail(f"{var} is not set")
    try:
        health = httpx.get(os.environ["NEXUS_RETRIEVER_URL"].rstrip("/") + "/health", timeout=10).json()
    except Exception as exc:
        fail(f"retriever not reachable: {exc!r}")
    out = Path(a.out)
    if out.exists():
        fail(f"{out} already exists; every run gets a new directory")
    (out / "results").mkdir(parents=True)
    (out / "tasks").mkdir()
    (out / "logs").mkdir()

    pinned = json.loads((ROOT / "nexus_eval" / "pins.json").read_text())["agent"]["commit"]
    manifest = {
        "run_name": out.name,
        "started_utc": dt.datetime.utcnow().isoformat() + "Z",
        "command": " ".join([sys.executable] + sys.argv),
        "repo_commit": sh(["git", "rev-parse", "HEAD"]),
        "repo_dirty_outside_nexus_eval": sh(["git", "status", "--porcelain", "--", ".", ":(exclude)nexus_eval"]),
        "agent_source_commit_pinned": pinned,
        "agent_src_matches_pin": subprocess.run(["git", "diff", "--quiet", pinned, "--", "src", "configs", "examples", "libs"],
                                                cwd=ROOT).returncode == 0,
        "bcp_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=os.environ["NEXUS_BCP_DIR"]).stdout.strip(),
        "pins_sha256": hashlib.sha256((ROOT / "nexus_eval" / "pins.json").read_bytes()).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "agent_env_freeze_sha256": hashlib.sha256(subprocess.run(["uv", "pip", "freeze", "-p", sys.executable], capture_output=True, text=True).stdout.encode()).hexdigest(),
        "retriever_health": health,
        "subset": subset,
        "task_timeout_seconds": a.task_timeout,
    }
    if not manifest["agent_src_matches_pin"]:
        fail("agent source (src/, configs/, examples/, libs/) differs from the pinned commit")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    for i, t in enumerate(subset["tasks"], 1):
        qid = t["query_id"]
        task_file = out / "tasks" / f"{qid}.json"
        task_file.write_text(json.dumps({"query_id": qid, "query": data[qid]["query"]}))
        res_file = out / "results" / f"{qid}.json"
        wd = out / "workdirs" / qid
        started = time.time()
        with open(out / "logs" / f"{qid}.log", "w") as log:
            try:
                rc = subprocess.run([sys.executable, "nexus_eval/run_one.py", "--task", str(task_file),
                                     "--workdir", str(wd), "--out", str(res_file)],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=a.task_timeout).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        if rc == 2:
            fail(f"run_one setup error on {qid}; see {out / 'logs' / (qid + '.log')}")
        if rc == "timeout" or not res_file.exists():
            res_file.write_text(json.dumps({"query_id": qid, "status": "timeout" if rc == "timeout" else "crash",
                                            "tool_call_counts": {}, "retrieved_docids": [],
                                            "result": [{"type": "output_text", "output": ""}],
                                            "nexus": {"runtime_seconds": round(time.time() - started, 1)}}))
        status = json.loads(res_file.read_text())["status"]
        print(f"[{i}/{len(subset['tasks'])}] {qid}: {status} ({time.time() - started:.0f}s)", flush=True)

    manifest["finished_utc"] = dt.datetime.utcnow().isoformat() + "Z"
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    rc = subprocess.run([sys.executable, "nexus_eval/score.py", "--run", str(out), "--data", a.data], cwd=ROOT).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()
