"""Download and pin everything BrowseComp-Plus needs, into a data directory outside the repo.

Run with the retriever environment's Python (needs huggingface.co access):
  python nexus_eval/prepare_data.py --bcp-dir <BrowseComp-Plus checkout> --data-dir <dir>

Produces:
  <dir>/browsecomp_plus_decrypted.jsonl   queries + answers (BrowseComp-Plus's own decrypt script)
  <dir>/indexes/bm25/                     pre-built Lucene BM25 index
  <dir>/tokenizer/                        Qwen/Qwen3-0.6B tokenizer (snippet truncation)
  <dir>/data_manifest.json                Hugging Face revisions and SHA-256 of every file
Re-running with --check verifies an existing directory against its manifest instead.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_table(root: Path):
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*"))
            if p.is_file() and ".cache" not in p.parts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bcp-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    d = Path(a.data_dir)
    if a.check:
        m = json.loads((d / "data_manifest.json").read_text())
        now = file_table(d)
        now.pop("data_manifest.json", None)
        bad = sorted(k for k in set(m["files"]) | set(now) if m["files"].get(k) != now.get(k))
        if bad:
            sys.exit(f"data directory differs from manifest: {bad[:10]}")
        print(f"ok: {len(now)} files match data_manifest.json")
        return
    from huggingface_hub import HfApi, snapshot_download

    d.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    revs = {
        "Tevatron/browsecomp-plus": api.dataset_info("Tevatron/browsecomp-plus").sha,
        "Tevatron/browsecomp-plus-indexes": api.dataset_info("Tevatron/browsecomp-plus-indexes").sha,
        "Qwen/Qwen3-0.6B": api.model_info("Qwen/Qwen3-0.6B").sha,
    }
    # BrowseComp-Plus's own decryption functions, applied to the dataset at the pinned revision.
    sys.path.insert(0, str(Path(a.bcp_dir) / "scripts_build_index"))
    from decrypt_dataset import DEFAULT_CANARY, transform_decrypt
    from datasets import load_dataset
    ds = load_dataset("Tevatron/browsecomp-plus", split="test", revision=revs["Tevatron/browsecomp-plus"])
    with open(d / "browsecomp_plus_decrypted.jsonl", "w", encoding="utf-8") as out:
        for record in ds:
            row = transform_decrypt(json.loads(json.dumps(record, ensure_ascii=False)), DEFAULT_CANARY, {"query_id"})
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    snapshot_download("Tevatron/browsecomp-plus-indexes", repo_type="dataset", allow_patterns=["bm25/*"],
                      revision=revs["Tevatron/browsecomp-plus-indexes"], local_dir=str(d / "indexes"))
    snapshot_download("Qwen/Qwen3-0.6B", allow_patterns=["tokenizer*", "vocab.json", "merges.txt", "*.json"],
                      ignore_patterns=["*.safetensors"], revision=revs["Qwen/Qwen3-0.6B"], local_dir=str(d / "tokenizer"))
    rows = sum(1 for l in open(d / "browsecomp_plus_decrypted.jsonl", encoding="utf-8") if l.strip())
    manifest = {"hf_revisions": revs, "queries": rows,
                
                "files": file_table(d)}
    (d / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
