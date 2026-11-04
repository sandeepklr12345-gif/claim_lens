"""Pydantic models shared by every ClaimLens component (SPEC.md 7.1)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Verdict(StrEnum):
    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class Chunk(BaseModel):
    """One clause-sized piece of a policy document."""

    chunk_id: str
    doc_id: str
    text: str
    section_path: str
    page: int = Field(ge=1)


class RetrievedClause(BaseModel):
    chunk: Chunk
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None


class MatchResult(BaseModel):
    """Cross-encoder score for a (rejection reason, clause) pair."""

    clause_id: str
    score: float  # raw logit
    prob: float = Field(ge=0.0, le=1.0)  # sigmoid(score)


class Attribution(BaseModel):
    """Token-level attribution over the clause text."""

    clause_id: str
    method: str = "integrated_gradients"
    tokens: list[str]
    scores: list[float]


class CitedSpan(BaseModel):
    text: str
    source: str = "clause"  # "clause" or "reason"; only clause spans can earn attribution mass
    clause_id: str | None = None


class Explanation(BaseModel):
    text: str
    cited_spans: list[CitedSpan] = Field(default_factory=list)


class AlignmentResult(BaseModel):
    """Attribution Alignment Score and which attribution words the cited spans covered."""

    aas: float = Field(ge=0.0, le=1.0)
    cited_idx: list[int] = Field(default_factory=list)
    unmatched: list[str] = Field(default_factory=list)


class Case(BaseModel):
    case_id: str
    policy_id: str
    letter_text: str
    reason: str
    gold_clause_ids: list[str]
    gold_verdict: Verdict
    source: str = "synthetic"


class Pair(BaseModel):
    pair_id: str
    policy_id: str
    reason: str
    clause_id: str
    clause_text: str
    label: int = Field(ge=0, le=1)
    source: str = "synthetic"


class CaseResult(BaseModel):
    case_id: str
    reason: str
    top_clauses: list[RetrievedClause]
    match: MatchResult | None = None
    attribution: Attribution | None = None
    explanation_initial: Explanation | None = None
    explanation_final: Explanation | None = None
    aas_initial: float | None = None
    aas_final: float | None = None
    aas_history: list[tuple[str, float]] = Field(default_factory=list)  # (round, AAS)
    verdict: Verdict
    appeal_draft: str | None = None
