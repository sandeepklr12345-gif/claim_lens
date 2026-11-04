"""MCP server `explainer` (stdio). Run: python -m claimlens.mcp_servers.explainer_server

Thin wrapper over claimlens.core: matcher score, Integrated Gradients, alignment score.
"""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from claimlens import core
from claimlens.schemas import Attribution, CitedSpan

mcp = FastMCP("explainer")


@mcp.tool()
def score_pair(reason: str, clause: str, clause_id: str = "") -> dict[str, Any]:
    """Cross-encoder support score for (rejection reason, clause text): raw logit and sigmoid."""
    return core.score_pair(reason, clause, clause_id).model_dump(mode="json")


@mcp.tool()
def attribute_ig(reason: str, clause: str, clause_id: str = "") -> dict[str, Any]:
    """Integrated Gradients word attributions over the clause for the matcher's score."""
    return core.attribute_ig(reason, clause, clause_id).model_dump(mode="json")


@mcp.tool()
def alignment_score(
    reason: str,
    clause: str,
    cited_spans: list[dict[str, Any]],
    attribution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attribution Alignment Score of cited spans (share of positive IG mass they cover).

    Pass `attribution` (from attribute_ig) to avoid recomputing IG.
    """
    spans = [CitedSpan.model_validate(s) for s in cited_spans]
    attr = Attribution.model_validate(attribution) if attribution else None
    return core.alignment_score(reason, clause, spans, attr).model_dump(mode="json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)  # stderr; stdout carries the MCP protocol
    import claimlens.explain.ig  # noqa: F401 - import torch/captum in the main thread

    core.get_matcher()  # warm up before the event loop starts (see policy_server.py)
    mcp.run()
