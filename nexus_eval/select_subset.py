"""Freeze a BrowseComp-Plus task subset: deterministic sample of query IDs.

  python nexus_eval/select_subset.py --data <browsecomp_plus_decrypted.jsonl> --n 10 --seed 20261005 \
      --name smoke-v0

Writes nexus_eval/subsets/<name>.json with the IDs plus SHA-256 of each question and answer
(never their text, which the benchmark keeps encrypted). Refuses to overwrite an existing subset.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--exclude", nargs="*", default=[], help="subset files whose IDs must not be reused")
    a = p.parse_args()

    out = Path(__file__).parent / "subsets" / f"{a.name}.json"
    if out.exists():
        sys.exit(f"{out} already exists; frozen subsets are never overwritten")
    data_bytes = Path(a.data).read_bytes()
    rows = [json.loads(l) for l in data_bytes.decode("utf-8").splitlines() if l.strip()]
    if not rows:
        sys.exit("dataset is empty")
    excluded = set()
    for f in a.exclude:
        excluded |= {t["query_id"] for t in json.loads(Path(f).read_text())["tasks"]}
    by_id = {str(r["query_id"]): r for r in rows}
    pool = sorted((q for q in by_id if q not in excluded), key=lambda q: (len(q), q))
    if a.n > len(pool):
        sys.exit(f"asked for {a.n} tasks but only {len(pool)} are available")
    chosen = sorted(random.Random(a.seed).sample(pool, a.n), key=lambda q: (len(q), q))
    subset = {
        "name": a.name,
        "benchmark": "BrowseComp-Plus (Tevatron/browsecomp-plus, split test)",
        "source_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "source_rows": len(rows),
        "seed": a.seed,
        "excluded_subsets": a.exclude,
        "method": "random.Random(seed).sample over query_ids sorted by (len, id), after exclusions",
        "tasks": [{"query_id": q, "query_sha256": sha(by_id[q]["query"]), "answer_sha256": sha(str(by_id[q]["answer"]))}
                  for q in chosen],
    }
    out.write_text(json.dumps(subset, indent=2) + "\n")
    print(f"wrote {out}: {[t['query_id'] for t in subset['tasks']]}")


if __name__ == "__main__":
    main()
