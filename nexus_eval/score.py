"""Judge and summarise one harness run directory.

  python nexus_eval/score.py --run <run dir> --data <browsecomp_plus_decrypted.jsonl>

Uses BrowseComp-Plus's own GRADER_TEMPLATE and parse_judge_response (pinned checkout in
NEXUS_BCP_DIR), with the judge model and settings in pins.json, called through OpenRouter.
Tasks whose status is not "completed" are scored incorrect without a judge call.

Exit codes: 0 = every task completed and judged; 3 = score produced but some tasks failed;
4 = judging failed (no valid score); 2 = setup error (missing results, empty subset, unknown IDs).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
PINS = json.loads((ROOT / "nexus_eval" / "pins.json").read_text())
# Override exists only for the offline failure-mode tests; it is recorded in summary.json.
JUDGE_ENDPOINT = os.environ.get("NEXUS_JUDGE_ENDPOINT") or PINS["judge"]["endpoint"]


def fail(code: int, msg: str):
    print(f"SCORE FAILURE ({code}): {msg}", file=sys.stderr)
    sys.exit(code)


def load_bcp_eval(bcp_dir: str):
    sys.path.insert(0, bcp_dir)
    spec = importlib.util.spec_from_file_location("bcp_eval", os.path.join(bcp_dir, "scripts_evaluation", "evaluate_with_openai.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def judge(prompt: str, key: str) -> dict:
    j = PINS["judge"]
    body = {"model": j["model"], "temperature": j["temperature"], "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}]}
    last = None
    for attempt in range(3):  # transport retries only; the verdict is never re-sampled after a parse
        try:
            r = httpx.post(JUDGE_ENDPOINT, json=body, timeout=180,
                           headers={"Authorization": f"Bearer {key}"})
            data = r.json()
            if r.status_code != 200 or "choices" not in data:
                raise RuntimeError(f"HTTP {r.status_code}: {str(data)[:300]}")
            return {"text": data["choices"][0]["message"]["content"] or "", "model": data.get("model"),
                    "usage": data.get("usage", {})}
        except Exception as exc:
            last = exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"judge call failed after 3 attempts: {last!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--data", required=True)
    a = p.parse_args()
    run = Path(a.run)
    manifest = json.loads((run / "manifest.json").read_text())
    subset = manifest["subset"]
    ids = [t["query_id"] for t in subset["tasks"]]
    if not ids:
        fail(2, "subset is empty")
    gt = {}
    for line in Path(a.data).read_text(encoding="utf-8").splitlines():
        if line.strip():
            o = json.loads(line)
            gt[str(o["query_id"])] = o
    for t in subset["tasks"]:
        q = gt.get(t["query_id"])
        if q is None:
            fail(2, f"query_id {t['query_id']} not in dataset")
        if hashlib.sha256(q["query"].encode()).hexdigest() != t["query_sha256"]:
            fail(2, f"query text for {t['query_id']} does not match the frozen subset hash")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        fail(2, "OPENROUTER_API_KEY is not set")

    bcp = load_bcp_eval(os.environ["NEXUS_BCP_DIR"])
    qrels = bcp.load_qrel_data(Path(os.environ["NEXUS_BCP_DIR"]) / "topics-qrels" / "qrel_evidence.txt")
    per_task, judge_failures = [], []
    for qid in ids:
        f = run / "results" / f"{qid}.json"
        if not f.exists():
            fail(2, f"missing result file for {qid}")
        r = json.loads(f.read_text())
        response = r["result"][-1]["output"] if r.get("result") else ""
        rel = set(qrels.get(qid, []))
        recall = len(rel & set(r.get("retrieved_docids", []))) / len(rel) if rel else None
        row = {"query_id": qid, "status": r["status"], "correct": False, "retrieval_recall": recall,
               "tool_call_counts": r.get("tool_call_counts", {}), "nexus": {k: r.get("nexus", {}).get(k) for k in
               ("steps", "runtime_seconds", "tokens", "cost_usd", "models_used")}}
        if r["status"] == "completed" and response.strip():
            prompt = bcp.create_judge_prompt(gt[qid]["query"], response, gt[qid]["answer"])
            try:
                jr = judge(prompt, key)
                parsed = bcp.parse_judge_response(jr["text"])
            except Exception as exc:
                judge_failures.append({"query_id": qid, "error": repr(exc)})
                row["judge_error"] = repr(exc)
            else:
                if parsed.get("correct") is None:
                    judge_failures.append({"query_id": qid, "error": "judge verdict unparseable"})
                row.update(correct=bool(parsed.get("correct")), judge=parsed, judge_model=jr["model"],
                           judge_usage=jr["usage"])
        per_task.append(row)

    n = len(ids)
    completed = sum(1 for r in per_task if r["status"] == "completed")
    correct = sum(1 for r in per_task if r["correct"])
    recalls = [r["retrieval_recall"] for r in per_task if r["retrieval_recall"] is not None]
    num = lambda k: sum((r["nexus"].get(k) or 0) for r in per_task)
    summary = {
        "run": manifest["run_name"],
        "subset": subset["name"],
        "n_tasks": n,
        "completed": completed,
        "failed": n - completed,
        "failure_statuses": {s: sum(1 for r in per_task if r["status"] == s) for s in sorted({r["status"] for r in per_task}) if s != "completed"},
        "correct": correct,
        "accuracy": round(correct / n, 4),
        "evidence_recall_mean": round(sum(recalls) / len(recalls), 4) if recalls else None,
        "search_calls_mean": round(sum(r["tool_call_counts"].get("search", 0) for r in per_task) / n, 2),
        "get_document_calls_mean": round(sum(r["tool_call_counts"].get("get_document", 0) for r in per_task) / n, 2),
        "agent_cost_usd": round(num("cost_usd"), 4),
        "agent_tokens_total": sum((r["nexus"].get("tokens") or {}).get("total", 0) for r in per_task),
        "agent_runtime_seconds": round(num("runtime_seconds"), 1),
        "judge": PINS["judge"]["model"],
        "judge_endpoint": JUDGE_ENDPOINT,
        "judge_failures": judge_failures,
        "per_task": per_task,
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_task"}, indent=2))
    if judge_failures:
        fail(4, f"{len(judge_failures)} judge failures; accuracy is not valid")
    if completed < n:
        fail(3, f"{n - completed} of {n} tasks failed (counted as incorrect); accuracy {summary['accuracy']}")
    sys.exit(0)


if __name__ == "__main__":
    main()
