"""Pydantic models shared by every ClaimLens component (SPEC.md 7.1)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

DISCLAIMER = "Decision support, not legal advice."


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
    verified: bool = False  # True only after the project owner checked it (data/QA.md)
    original_label: int | None = None  # label before any audit/review change (R1b)
    hard_negative: bool = False  # R1b: mined negative, judge-confirmed (data/hard_negatives.jsonl)


class CitedClause(BaseModel):
    """A policy quote in an appeal; `quote` is verified verbatim against the clause text."""

    clause_id: str
    quote: str


class AppealDraft(BaseModel):
    """Structured appeal (SPEC.md 14.4). `kind` is "appeal" (NOT_SUPPORTED) or "clarification"
    (INSUFFICIENT_EVIDENCE). `source` is "llm" or "template" (the grounded fallback)."""

    kind: str
    source: str
    attempts: int = 1  # LLM tries used; a first-try pass has attempts == 1 and source "llm"
    fallback_reason: str | None = None
    details: dict[str, str] = Field(default_factory=dict)  # letter-sourced or [placeholder]
    facts: list[str]
    cited_clauses: list[CitedClause]
    reasoning: str
    relief_sought: str
    document_checklist: list[str]
    deadline_note: str
    deadline_verified: bool = False  # Rule 14(3) checked vs the official text till 09.11.2023
    disclaimer: str = DISCLAIMER
    placeholders: list[str] = Field(default_factory=list)


class DeadlineInfo(BaseModel):
    """Output of compute_deadlines. `verified` means Rule 14(3)(a)-(b) were checked against the
    official text as amended till 09.11.2023 (see `source`); later amendments were not checked."""

    rejection_date: str
    grievance_filed_date: str | None = None
    insurer_reply_date: str | None = None
    ombudsman_earliest_date: str | None = None
    ombudsman_latest_date: str | None = None
    basis: str
    conservative: bool = False
    rules: list[str]
    source: str = ""
    verified: bool = False
    verify_note: str = "Rule texts are not verified. Verify with the Ombudsman office."
    disclaimer: str = DISCLAIMER


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
    revisions: list[Explanation] = Field(default_factory=list)  # lines up with aas_history[1:]
    verdict: Verdict
    appeal_draft: str | None = None  # plain-text rendering of `appeal`
    appeal: AppealDraft | None = None
    plan_trace: list[dict[str, Any]] = Field(default_factory=list)  # R3: one entry per decision
    abstain_reason: str | None = None  # set when the agent stopped before matching (R3)


class CaseStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class CaseCreate(BaseModel):
    """Body of POST /cases."""

    letter_text: str = Field(min_length=1, max_length=20000)
    policy_id: str = Field(min_length=1)


class CaseRecord(BaseModel):
    """A case as the API stores and returns it."""

    case_id: str
    policy_id: str
    letter_text: str
    status: CaseStatus = CaseStatus.CREATED
    result: CaseResult | None = None
    error: str | None = None
    disclaimer: str = DISCLAIMER
