"""Core tool functions shared by the MCP servers and the direct (non-MCP) fallback.

These are the plain, tested functions; the MCP servers and LangGraph nodes are thin wrappers.
Heavy objects (matcher, retrievers) are created lazily and cached per process.
"""

from __future__ import annotations

from functools import lru_cache

from claimlens.explain.aas import alignment_score as _aas
from claimlens.rag.parse import load_chunks
from claimlens.schemas import (
    AlignmentResult,
    Attribution,
    Chunk,
    CitedSpan,
    MatchResult,
    RetrievedClause,
)


@lru_cache(maxsize=1)
def get_matcher():  # noqa: ANN201 - Matcher (imported lazily: needs torch)
    from claimlens.matcher.scorer import Matcher

    return Matcher()


@lru_cache(maxsize=8)
def get_retriever(doc_id: str):  # noqa: ANN201 - HybridRetriever
    from claimlens.rag.retriever import build_retriever

    return build_retriever(doc_id, dense=True)


@lru_cache(maxsize=8)
def _chunks_by_id(doc_id: str) -> dict[str, Chunk]:
    return {c.chunk_id: c for c in load_chunks(doc_id)}


def search_clauses(query: str, doc_id: str, top_k: int = 5) -> list[RetrievedClause]:
    return get_retriever(doc_id).retrieve(query, k=top_k)


def get_clause(doc_id: str, clause_id: str) -> Chunk:
    try:
        return _chunks_by_id(doc_id)[clause_id]
    except KeyError as e:
        raise KeyError(f"no clause {clause_id!r} in {doc_id!r}") from e


def score_pair(reason: str, clause: str, clause_id: str = "") -> MatchResult:
    from claimlens.matcher.scorer import sigmoid

    logit = float(get_matcher().logits([(reason, clause)])[0])
    return MatchResult(clause_id=clause_id, score=logit, prob=float(sigmoid(logit)))


def attribute_ig(reason: str, clause: str, clause_id: str = "") -> Attribution:
    from claimlens.explain.ig import attribute

    return attribute(get_matcher(), reason, clause, clause_id, method="integrated_gradients")


def alignment_score(
    reason: str,
    clause: str,
    cited_spans: list[CitedSpan],
    attribution: Attribution | None = None,
) -> AlignmentResult:
    """AAS of `cited_spans` against IG on (reason, clause). Pass `attribution` to skip IG."""
    attr = attribution or attribute_ig(reason, clause)
    return _aas(attr, cited_spans, clause)
