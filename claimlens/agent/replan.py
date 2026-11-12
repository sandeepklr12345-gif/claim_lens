"""R3 re-plan helpers (SPEC.md 14.5): decision rule, query rewriting and rank fusion.

Plain tested functions; the graph nodes `check_retrieval` and `replan` are thin wrappers.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from claimlens.agent.tools import Tools
from claimlens.llm.client import LLM
from claimlens.schemas import RetrievedClause

log = logging.getLogger(__name__)
ABSTAIN_REASON = "no clause above confidence threshold"
RRF_K = 60
REPLAN_SYSTEM = "You help search health-insurance policy wording. Reply with JSON only."
REPLAN_PROMPT = """A claim was rejected for this reason:
\"\"\"{reason}\"\"\"

The first search of the policy wording found no clause that clearly supports it. Write TWO
alternative search queries (each under 20 words) that use different words a policy might use
for the same idea. Reply as JSON: {{"queries": ["...", "..."]}}"""


def top_prob(clauses: list[RetrievedClause]) -> float | None:
    """Matcher probability of the best re-ranked clause (`score` is the matcher logit)."""
    if not clauses:
        return None
    logit = max(c.score for c in clauses)
    return 1.0 / (1.0 + math.exp(-logit))  # sigmoid, no torch import (CI has no torch)


def retrieval_decision(top: float | None, tau_retr: float | None, replanned: bool) -> str:
    """"ok" (continue), "replan" (rewrite and search again) or "abstain".

    tau_retr None disables the check. A re-plan happens at most once per case.
    """
    if tau_retr is None:
        return "ok"
    if top is not None and top >= tau_retr:
        return "ok"
    return "abstain" if replanned else "replan"


def parse_queries(raw: Any, reason: str) -> list[str]:
    """At most two distinct, non-empty alternative queries (never the reason itself)."""
    items = raw.get("queries", []) if isinstance(raw, dict) else []
    out: list[str] = []
    for q in items if isinstance(items, list) else []:
        q = " ".join(str(q).split())
        if q and q != reason and q not in out:
            out.append(q)
    return out[:2]


def rrf_merge(rankings: list[list[RetrievedClause]], k: int = RRF_K) -> list[RetrievedClause]:
    """Reciprocal rank fusion over clause lists, keyed by chunk id."""
    scores: dict[str, float] = {}
    first: dict[str, RetrievedClause] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            cid = hit.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            first.setdefault(cid, hit)
    return [first[c].model_copy(update={"score": s})
            for c, s in sorted(scores.items(), key=lambda kv: -kv[1])]


def replan_retrieve(tools: Tools, llm: LLM, reason: str, policy_id: str, top_k_fused: int,
                    top_k_return: int) -> tuple[list[RetrievedClause], list[str]]:
    """Rewrite the reason into two queries, search with the reason and each query, merge by RRF,
    then re-rank the merged top clauses against the ORIGINAL reason with the matcher.

    Returns (clauses, queries). If the LLM gives no usable queries the original search is simply
    repeated (and the caller will abstain if it is still below threshold).
    """
    try:
        queries = parse_queries(llm.generate_json(REPLAN_PROMPT.format(reason=reason),
                                                  REPLAN_SYSTEM), reason)
    except Exception as e:  # noqa: BLE001 - a failed rewrite must not crash the case
        log.warning("replan query rewrite failed: %s", e)
        queries = []
    rankings = [tools.search_clauses(q, policy_id, top_k_fused) for q in [reason, *queries]]
    merged = rrf_merge(rankings)[:top_k_fused]
    scored = []
    for h in merged:
        m = tools.score_pair(reason, h.chunk.text, h.chunk.chunk_id)
        scored.append(h.model_copy(update={"score": m.score}))
    scored.sort(key=lambda h: -h.score)
    return scored[:top_k_return], queries
