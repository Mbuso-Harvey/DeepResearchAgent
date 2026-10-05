# Nexus Research v0 evaluation harness

An external BrowseComp-Plus harness for measuring the inherited agent. It lives outside `src/`, `configs/`,
`examples/` and `libs/`, and `run_bcp.py` refuses to run if any of those differ from the pinned commit
in `pins.json`. Nothing here changes the agent's prompts, model, settings, memory or tools. The only
additions are BrowseComp-Plus's own `search` and `get_document` corpus tools (`bcp_tools.py`), registered
from outside the source tree.

## What is pinned (`pins.json`)
- Agent: repo commit, config, agent name, model `openrouter/gemini-3-flash-preview`, and every inherited
  inference setting.
- Benchmark: BrowseComp-Plus commit, BM25 index, top-5 search, 512-token snippets (Qwen3-0.6B tokenizer),
  and the official `QUERY_TEMPLATE`.
- Judge: `openai/gpt-4.1` at temperature 0 with BrowseComp-Plus's `GRADER_TEMPLATE`.
- Environments: `env/agent.lock.txt` and `env/retriever.lock.txt`, installed with `--no-deps`.

## Not in git
- `OPENROUTER_API_KEY`: comes from the environment only, and is never written anywhere.
- The BinanceDatabase import stand-in, passed as `NEXUS_STANDIN_DIR`. `run_one.py` checks its tree hash
  against the pin. Every entry point raises `StandInInvoked`.
- Data: the decrypted queries, the index and the tokenizer, built by `prepare_data.py` with a manifest of
  Hugging Face revisions and file hashes.

## Run
```bash
bash nexus_eval/setup.sh $OUT                      # agent + retriever envs, BrowseComp-Plus checkout
$OUT/retriever/bin/python nexus_eval/prepare_data.py --bcp-dir $OUT/BrowseComp-Plus --data-dir $DATA
OPENAI_API_KEY=unused-placeholder $OUT/retriever/bin/python nexus_eval/retriever_server.py \
    --bcp-dir $OUT/BrowseComp-Plus --index-path $DATA/indexes/bm25 --tokenizer $DATA/tokenizer --log $RUN.retriever.jsonl &
python3 nexus_eval/select_subset.py --data $DATA/browsecomp_plus_decrypted.jsonl --name smoke-v0 --n 10 --seed 20261005
NEXUS_BCP_DIR=$OUT/BrowseComp-Plus NEXUS_RETRIEVER_URL=http://127.0.0.1:8765 NEXUS_STANDIN_DIR=$STANDIN \
PATH=$OUT/agent/bin:$PATH python nexus_eval/run_bcp.py --subset nexus_eval/subsets/smoke-v0.json \
    --data $DATA/browsecomp_plus_decrypted.jsonl --out $RUN
```
(pyserini needs `OPENAI_API_KEY` to be set to import. The placeholder is never used.)

## Scoring and exit codes
Accuracy is correct answers divided by the number of selected tasks. Any task whose status is not
`completed` counts as incorrect. Possible statuses are `crash`, `timeout`, `model_fallback`,
`agent_step_error`, `tool_error`, `max_steps`, `not_done` and `no_answer`.

| Exit code | Meaning |
| --- | --- |
| 0 | Every task completed and was judged |
| 3 | A score was produced, but some tasks failed |
| 4 | Judging failed, so there is no valid score |
| 2 | Setup error: empty or unknown subset, data hash mismatch, retriever down, stand-in hash mismatch, or agent source differs from the pin |

`tests/test_failure_modes.py` proves these exit codes offline, against a scripted model, judge and retriever.

Two runs are consistent when their correct counts differ by at most max(1, ceil(0.10·n)), and either both
runs have the same number of failed tasks or each has fewer than 10% failed.
