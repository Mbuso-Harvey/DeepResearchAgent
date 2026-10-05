"""BrowseComp-Plus corpus tools, registered into the agent from outside src/.

Importing this module registers two tools in the inherited TOOL registry:
  search(query)        -> top-5 BM25 hits from the fixed corpus (docid, score, snippet)
  get_document(docid)  -> full document text
Names and descriptions are BrowseComp-Plus's own (searcher/searchers/base.py at the
pinned commit). Both call the local retriever_server.py over HTTP
(NEXUS_RETRIEVER_URL). Every call and every failure is recorded in CALL_LOG so the
harness can fail a task on any tool error instead of letting it pass silently.
"""
import json
import os
import time
from typing import Any, Dict, List

import httpx
from pydantic import Field

from src.registry import TOOL
from src.tool.types import Tool, ToolResponse, ToolExtra

CALL_LOG: List[Dict[str, Any]] = []

_K = 5
_SEARCH_DESCRIPTION = (f"Perform a search on a knowledge source. Returns top-{_K} hits with docid, score, "
                       "and snippet. The snippet contains the document's contents (may be truncated based on token limits).")
_GET_DOCUMENT_DESCRIPTION = "Retrieve a full document by its docid."


def _url(path: str) -> str:
    base = os.environ.get("NEXUS_RETRIEVER_URL")
    if not base:
        raise RuntimeError("NEXUS_RETRIEVER_URL is not set")
    return base.rstrip("/") + path


async def _post(tool: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    started = time.time()
    entry = {"tool": tool, "args": payload, "ok": False}
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(_url(path), json=payload)
        data = resp.json()
        if resp.status_code != 200:
            raise RuntimeError(f"retriever HTTP {resp.status_code}: {data.get('error')}")
        entry["ok"] = True
        return data
    except Exception as exc:
        entry["error"] = repr(exc)
        raise
    finally:
        entry["seconds"] = round(time.time() - started, 3)
        CALL_LOG.append(entry)


@TOOL.register_module(force=True)
class BcpSearchTool(Tool):
    name: str = "search"
    description: str = _SEARCH_DESCRIPTION
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool")
    require_grad: bool = Field(default=False, description="Whether the tool requires gradients")

    def __init__(self, require_grad: bool = False, **kwargs):
        super().__init__(require_grad=require_grad, **kwargs)

    async def __call__(self, query: str, **kwargs) -> ToolResponse:
        """
        Search the index and return top-k hits.

        Args:
            query (str): Search query string.
        """
        try:
            data = await _post("search", "/search", {"query": str(query)})
        except Exception as exc:
            return ToolResponse(success=False, message=f"search failed: {exc!r}")
        results = data["results"]
        CALL_LOG[-1]["docids"] = [r["docid"] for r in results]
        return ToolResponse(success=True, message=json.dumps(results, ensure_ascii=False),
                            extra=ToolExtra(data={"docids": CALL_LOG[-1]["docids"]}))


@TOOL.register_module(force=True)
class BcpGetDocumentTool(Tool):
    name: str = "get_document"
    description: str = _GET_DOCUMENT_DESCRIPTION
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool")
    require_grad: bool = Field(default=False, description="Whether the tool requires gradients")

    def __init__(self, require_grad: bool = False, **kwargs):
        super().__init__(require_grad=require_grad, **kwargs)

    async def __call__(self, docid: str, **kwargs) -> ToolResponse:
        """
        Retrieve the full text of a document by its ID from the search index.

        Args:
            docid (str): Document ID to retrieve.
        """
        try:
            data = await _post("get_document", "/get_document", {"docid": str(docid)})
        except Exception as exc:
            return ToolResponse(success=False, message=f"get_document failed: {exc!r}")
        doc = data.get("document")
        if doc is None:
            CALL_LOG[-1]["docids"] = []
            return ToolResponse(success=True, message="null")
        CALL_LOG[-1]["docids"] = [doc["docid"]]
        return ToolResponse(success=True, message=json.dumps(doc, ensure_ascii=False))
