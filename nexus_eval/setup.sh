#!/bin/bash
# Build the two pinned environments and the pinned BrowseComp-Plus checkout for the Nexus Research v0 harness.
#   bash nexus_eval/setup.sh <out dir>
# Linux x86_64. Needs uv, git, Java 21 (pyserini), and network access to pypi.org and github.com.
# The agent environment is the baseline's "Env P"; it is installed from its lock with --no-deps so it is
# byte-identical to the one measured, including the inherited dependency conflicts.
set -euo pipefail
OUT=$(realpath -m "${1:?usage: setup.sh <out dir>}")
HERE=$(cd "$(dirname "$0")" && pwd)
BCP_COMMIT=$(python3 -c "import json;print(json.load(open('$HERE/pins.json'))['benchmark']['commit'])")

java -version 2>&1 | grep -q 'version "21' || { echo "Java 21 is required by pyserini" >&2; exit 2; }
uv python install 3.11.13
PY=$(uv python find --managed-python 3.11.13)   # python-build-standalone ships tkinter, which src/ imports

uv venv -q -p "$PY" "$OUT/agent"
uv pip install -p "$OUT/agent/bin/python" --no-deps -r "$HERE/env/agent.lock.txt"
uv venv -q -p "$PY" "$OUT/retriever"
uv pip install -p "$OUT/retriever/bin/python" --no-deps -r "$HERE/env/retriever.lock.txt"

if [ ! -d "$OUT/BrowseComp-Plus" ]; then git clone -q https://github.com/texttron/BrowseComp-Plus "$OUT/BrowseComp-Plus"; fi
git -C "$OUT/BrowseComp-Plus" checkout -q "$BCP_COMMIT"

for env in agent retriever; do
  diff <(uv pip freeze -p "$OUT/$env/bin/python" 2>/dev/null) "$HERE/env/$env.lock.txt" >/dev/null \
    || { echo "$env environment does not match its lock" >&2; exit 2; }
done
echo "ok: $OUT/agent, $OUT/retriever, $OUT/BrowseComp-Plus @ $BCP_COMMIT"
