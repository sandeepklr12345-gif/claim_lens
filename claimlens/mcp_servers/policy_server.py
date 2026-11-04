"""MCP server `policy-search` (stdio). Run: python -m claimlens.mcp_servers.policy_server

Thin wrapper over claimlens.core. Tools return one JSON object each.
"""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from claimlens import core

mcp = FastMCP("policy-search")


@mcp.tool()
def search_clauses(query: str, doc_id: str, top_k: int = 5) -> dict[str, Any]:
    """Hybrid (BM25 + dense) search of one policy; returns the top_k clauses with section paths."""
    hits = core.search_clauses(query, doc_id, top_k)
    return {"results": [h.model_dump(mode="json") for h in hits]}


@mcp.tool()
def get_clause(doc_id: str, clause_id: str) -> dict[str, Any]:
    """Return one clause (text, section path, page) by id."""
    return core.get_clause(doc_id, clause_id).model_dump(mode="json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)  # stderr; stdout carries the MCP protocol
    # Warm up in the main thread BEFORE the event loop starts: importing torch / loading the
    # embedding model inside a tool call hangs on Windows.
    from claimlens.rag.parse import list_policies

    for _doc in list_policies():
        core.get_retriever(_doc)
    mcp.run()
