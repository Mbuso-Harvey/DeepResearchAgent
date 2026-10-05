"""Run the inherited tool-calling agent on ONE BrowseComp-Plus query, in a fresh workdir.

Run from the repository root with the agent environment's Python:
  python nexus_eval/run_one.py --task <task.json> --workdir <fresh dir> --out <result.json>

<task.json> = {"query_id": ..., "query": ...}. Requires NEXUS_BCP_DIR (pinned BrowseComp-Plus
checkout), NEXUS_RETRIEVER_URL, NEXUS_STANDIN_DIR (the approved BinanceDatabase stand-in, verified
against its recorded hash) and OPENROUTER_API_KEY.

Exit codes: 0 = task completed with an answer; 1 = task failed (status says why);
2 = setup error (bad inputs, stand-in hash mismatch, missing env).
Nothing in src/ is modified; the agent is initialised exactly as examples/run_tool_calling_agent.py
does, plus the two corpus tools from nexus_eval/bcp_tools.py.
"""
import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
import traceback
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINS = json.loads((ROOT / "nexus_eval" / "pins.json").read_text())
TOOL_NAMES = PINS["agent"]["inherited_tools"] + PINS["agent"]["added_tools"]
NO_RESULT = {"", "No result provided", "The task has not been completed."}


def setup_error(msg: str):
    print(f"SETUP ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


def standin_tree_sha256(standin_dir: str) -> str:
    base = Path(standin_dir)
    files = sorted(str(p.relative_to(base)) for p in (base / "libs").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    h = hashlib.sha256()
    for rel in files:
        h.update((base / rel).read_bytes())
    return h.hexdigest()


class Capture(logging.Handler):
    """Collects the agent's own log records that signal failures, steps and usage."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.step_errors, self.fallbacks, self.tool_infra_errors, self.usage, self.errors = [], [], [], [], []
        self.steps_reported = None

    def emit(self, record):
        msg = record.getMessage()
        if "Error in thinking and tool step" in msg:
            self.step_errors.append(msg[:500])
        if "falling back to" in msg or "Fallback model" in msg:
            self.fallbacks.append(msg[:500])
        if re.search(r"Tool '.*' (is not registered|execution timed out)", msg):
            self.tool_infra_errors.append(msg[:500])
        m = re.search(r"Agent completed after (\d+)/(\d+) steps", msg)
        if m:
            self.steps_reported = [int(m.group(1)), int(m.group(2))]
        if "💰 Usage:" in msg:
            u = {}
            for part in msg.split("Usage:", 1)[1].split(","):
                if "=" in part:
                    k, v = part.strip().split("=", 1)
                    u[k] = v
            self.usage.append(u)
        if record.levelno >= logging.ERROR:
            self.errors.append(msg[:500])


async def run(task: dict, workdir: str) -> dict:
    sys.path.insert(0, str(ROOT))
    from src.config import config
    from src.logger import logger
    from src.model import model_manager
    from src.version import version_manager
    from src.prompt import prompt_manager
    from src.memory import memory_manager
    from src.tool import tcp
    from src.skill import scp
    from src.environment import ecp
    from src.agent import acp
    from src.session.types import SessionContext
    import nexus_eval.bcp_tools as bcp_tools  # registers search / get_document

    sys.path.insert(0, os.path.join(os.environ["NEXUS_BCP_DIR"], "search_agent"))
    from prompts import QUERY_TEMPLATE  # BrowseComp-Plus task prompt, verbatim

    cfg = {"workdir": workdir, "tool_calling_agent.workdir": workdir, "tool_names": TOOL_NAMES}
    config.initialize(config_path=PINS["agent"]["config"], args=Namespace(cfg_options=cfg))
    if config.model_name != PINS["agent"]["model"]:
        raise RuntimeError(f"config model {config.model_name} != pinned {PINS['agent']['model']}")
    logger.initialize(config=config)
    cap = Capture()
    logger.addHandler(cap)

    await model_manager.initialize()
    await prompt_manager.initialize()
    await memory_manager.initialize(memory_names=config.memory_names)
    await tcp.initialize(tool_names=config.tool_names)
    missing = [t for t in TOOL_NAMES if t not in await tcp.list()]
    if missing:
        raise RuntimeError(f"tools failed to initialise: {missing}")
    await scp.initialize(skill_names=getattr(config, "skill_names", None))
    await ecp.initialize(config.env_names)
    await acp.initialize(agent_names=config.agent_names)
    await version_manager.initialize()

    prompt = QUERY_TEMPLATE.format(Question=task["query"])
    started = time.time()
    response = await acp(name=PINS["agent"]["agent_name"], input={"task": prompt, "files": []}, ctx=SessionContext())
    runtime = time.time() - started

    data = (response.extra.data if response and response.extra and response.extra.data else {}) or {}
    done = bool(data.get("done"))
    result = "" if data.get("result") is None else str(data.get("result"))
    calls = bcp_tools.CALL_LOG
    tool_errors = [c for c in calls if not c["ok"]] + [{"log": m} for m in cap.tool_infra_errors]
    steps = cap.steps_reported

    off_pin = sorted({u.get("model", "?") for u in cap.usage} - {PINS["agent"]["model"]})
    if cap.fallbacks or off_pin:
        status = "model_fallback"
    elif cap.step_errors:
        status = "agent_step_error"
    elif tool_errors:
        status = "tool_error"
    elif not done and steps and steps[0] >= steps[1]:
        status = "max_steps"
    elif not done:
        status = "not_done"
    elif result.strip() in NO_RESULT:
        status = "no_answer"
    else:
        status = "completed"

    counts = {}
    for c in calls:
        counts[c["tool"]] = counts.get(c["tool"], 0) + 1
    docids = sorted({d for c in calls if c["ok"] for d in c.get("docids", [])})
    tokens = {k: sum(int(u.get(k, 0)) for u in cap.usage) for k in ("input", "output", "total")}
    cost = sum(float(u["cost"].lstrip("$")) for u in cap.usage if "cost" in u)
    return {
        "query_id": task["query_id"],
        "status": status,
        "tool_call_counts": counts,
        "retrieved_docids": docids,
        "result": [{"type": "output_text", "output": result}],
        "nexus": {
            "done": done,
            "reasoning": data.get("reasoning"),
            "steps": steps,
            "runtime_seconds": round(runtime, 2),
            "model_calls": len(cap.usage),
            "models_used": sorted({u.get("model", "?") for u in cap.usage}),
            "tokens": tokens,
            "cost_usd": round(cost, 6),
            "cost_reported_calls": sum(1 for u in cap.usage if "cost" in u),
            "step_errors": cap.step_errors,
            "fallbacks": cap.fallbacks,
            "tool_errors": tool_errors,
            "error_log_lines": cap.errors[:50],
            "tool_calls": calls,
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--workdir", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    for var in ("NEXUS_BCP_DIR", "NEXUS_RETRIEVER_URL", "NEXUS_STANDIN_DIR"):
        if not os.environ.get(var):
            setup_error(f"{var} is not set")
    got = standin_tree_sha256(os.environ["NEXUS_STANDIN_DIR"])
    if got != PINS["agent"]["binance_db_standin_tree_sha256"]:
        setup_error(f"BinanceDatabase stand-in hash {got} != pinned {PINS['agent']['binance_db_standin_tree_sha256']}")
    sys.path.insert(0, os.environ["NEXUS_STANDIN_DIR"])
    try:
        task = json.loads(Path(a.task).read_text())
        assert task.get("query_id") and task.get("query"), "task needs query_id and query"
    except Exception as exc:
        setup_error(f"bad task file: {exc!r}")
    if os.path.exists(a.workdir) and os.listdir(a.workdir):
        setup_error(f"workdir {a.workdir} is not empty; every task needs a fresh one")
    os.makedirs(a.workdir, exist_ok=True)

    try:
        out = asyncio.run(run(task, os.path.abspath(a.workdir)))
    except Exception as exc:
        out = {"query_id": task["query_id"], "status": "crash", "tool_call_counts": {}, "retrieved_docids": [],
               "result": [{"type": "output_text", "output": ""}],
               "nexus": {"exception": repr(exc), "traceback": traceback.format_exc()[-4000:]}}
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"{out['query_id']}: {out['status']}")
    sys.exit(0 if out["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
