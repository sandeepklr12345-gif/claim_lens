"""Linear LangGraph agent: parse -> retrieve -> match -> explain_reflect -> verdict -> draft.

Nodes are thin wrappers over tested functions; tools come through the `Tools` interface
(MCP or direct). Named nodes keep the graph extensible for Phase 2 (SPEC.md 12.1).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from claimlens.agent.tools import Tools
from claimlens.agent.verdict import appeal_template, decide, stance_for
from claimlens.appeal.draft import appeal_text, draft_appeal
from claimlens.explain.explain import TAU_AAS, reflect
from claimlens.llm.client import LLM
from claimlens.schemas import (
    AppealDraft,
    Attribution,
    Case,
    CaseResult,
    Explanation,
    MatchResult,
    RetrievedClause,
    Verdict,
)

log = logging.getLogger(__name__)
# Called with (event, data): a node name and its state update once that node finishes, plus
# "attribution" {"words": n} and "aas" {"label", "aas"} from inside explain_reflect.
Progress = Callable[[str, dict[str, Any]], None]
SYSTEM = "You are a careful assistant for health-insurance claim audits. Reply with JSON only."
PARSE_PROMPT = """Claim rejection letter:
\"\"\"{letter}\"\"\"

Extract the insurer's stated reason for rejecting the claim as ONE short sentence (max 30
words), in the letter's own terms. Reply as JSON: {{"reason": "..."}}"""


class AgentState(TypedDict, total=False):
    case_id: str
    policy_id: str
    letter_text: str
    reason: str
    top_clauses: list[RetrievedClause]
    match: MatchResult
    attribution: Attribution
    explanation_initial: Explanation
    explanation_final: Explanation
    aas_initial: float
    aas_final: float
    aas_history: list[tuple[str, float]]
    revisions: list[Explanation]
    verdict: Verdict
    appeal_draft: str | None
    appeal: AppealDraft | None


def extract_reason(llm: LLM, letter_text: str) -> str:
    """Ask the LLM for the stated rejection reason; fall back to the start of the letter."""
    try:
        out = llm.generate_json(PARSE_PROMPT.format(letter=letter_text), SYSTEM)
        reason = str(out.get("reason", "")).strip()
    except Exception as e:  # noqa: BLE001
        log.warning("reason extraction failed (%s); using the letter text", e)
        reason = ""
    return reason or letter_text[:300]  # FALLBACK (marked): the letter's first 300 chars


def retrieve_rerank(
    tools: Tools, reason: str, policy_id: str, top_k_fused: int, top_k_return: int
) -> list[RetrievedClause]:
    """Hybrid top-k, re-ranked by the cross-encoder; `score` is the matcher logit."""
    hits = tools.search_clauses(reason, policy_id, top_k_fused)
    scored = []
    for h in hits:
        m = tools.score_pair(reason, h.chunk.text, h.chunk.chunk_id)
        scored.append(h.model_copy(update={"score": m.score}))
    scored.sort(key=lambda h: -h.score)
    return scored[:top_k_return]


def _no_progress(_event: str, _data: dict[str, Any]) -> None:
    return None


def build_graph(
    tools: Tools, llm: LLM, cfg: dict[str, Any], emit: Progress = _no_progress
) -> Any:
    rag = cfg["rag"]
    agent = cfg["agent"]
    tau_high, tau_low = agent["tau_high"], agent["tau_low"]

    def parse(state: AgentState) -> dict[str, Any]:
        return {"reason": extract_reason(llm, state["letter_text"])}

    def retrieve(state: AgentState) -> dict[str, Any]:
        return {"top_clauses": retrieve_rerank(
            tools, state["reason"], state["policy_id"], rag["top_k_fused"], rag["top_k_return"])}

    def match(state: AgentState) -> dict[str, Any]:
        if not state["top_clauses"]:  # nothing retrieved: abstain downstream
            return {}
        best = state["top_clauses"][0]
        return {"match": tools.score_pair(state["reason"], best.chunk.text, best.chunk.chunk_id)}

    def explain_reflect(state: AgentState) -> dict[str, Any]:
        if "match" not in state:
            return {}
        best = state["top_clauses"][0].chunk
        reason, prob = state["reason"], state["match"].prob
        attr = tools.attribute_ig(reason, best.text, best.chunk_id)
        emit("attribution", {"words": len(attr.tokens)})
        stance = stance_for(decide(prob, tau_high, tau_low))
        res = reflect(
            llm, attr, reason, best.chunk_id, best.text, prob, stance,
            tau=agent.get("tau_aas", TAU_AAS), max_rounds=agent.get("max_rounds", 2),
            aas_fn=lambda spans: tools.alignment_score(reason, best.text, spans, attr),
            on_aas=lambda label, aas: emit("aas", {"label": label, "aas": aas}))
        return {"attribution": attr, "explanation_initial": res.initial,
                "explanation_final": res.final, "aas_initial": res.aas_initial,
                "aas_final": res.aas_final, "aas_history": res.history,
                "revisions": res.revisions}

    def verdict(state: AgentState) -> dict[str, Any]:
        if "match" not in state:
            return {"verdict": Verdict.INSUFFICIENT_EVIDENCE}
        return {"verdict": decide(state["match"].prob, tau_high, tau_low)}

    def draft(state: AgentState) -> dict[str, Any]:
        """Grounded appeal (NOT_SUPPORTED) or clarification request (INSUFFICIENT_EVIDENCE)."""
        if state["verdict"] == Verdict.SUPPORTED or not state.get("top_clauses"):
            return {"appeal_draft": None, "appeal": None}
        partial = CaseResult(
            case_id=state["case_id"], reason=state["reason"], top_clauses=state["top_clauses"],
            explanation_final=state.get("explanation_final"), verdict=state["verdict"])
        try:
            appeal = draft_appeal(partial, state["letter_text"], llm)
        except Exception as e:  # noqa: BLE001
            log.warning("appeal drafter failed (%s); using the Phase 1 template", e)
            appeal = None
        if appeal is not None:
            return {"appeal": appeal, "appeal_draft": appeal_text(appeal)}
        if state["verdict"] == Verdict.NOT_SUPPORTED and "explanation_final" in state:
            best = state["top_clauses"][0].chunk  # FALLBACK (marked): Phase 1 template, no LLM
            return {"appeal": None, "appeal_draft": appeal_template(
                state["reason"], best.chunk_id, best.text, state["explanation_final"])}
        return {"appeal": None, "appeal_draft": None}

    g = StateGraph(AgentState)
    for name, fn in [("parse", parse), ("retrieve", retrieve), ("match", match),
                     ("explain_reflect", explain_reflect), ("verdict", verdict),
                     ("draft", draft)]:
        g.add_node(name, fn)
    g.add_edge(START, "parse")
    for a, b in [("parse", "retrieve"), ("retrieve", "match"), ("match", "explain_reflect"),
                 ("explain_reflect", "verdict"), ("verdict", "draft")]:
        g.add_edge(a, b)
    g.add_edge("draft", END)
    return g.compile()


def run_case(
    case: Case, tools: Tools, llm: LLM, cfg: dict[str, Any], on_progress: Progress | None = None
) -> CaseResult:
    emit = on_progress or _no_progress
    graph = build_graph(tools, llm, cfg, emit)
    out: AgentState = {
        "case_id": case.case_id, "policy_id": case.policy_id, "letter_text": case.letter_text}
    for step in graph.stream(dict(out), stream_mode="updates"):  # same order as invoke()
        for node, update in step.items():
            out.update(update or {})
            emit(node, update or {})
    return CaseResult(
        case_id=case.case_id, reason=out["reason"], top_clauses=out["top_clauses"],
        match=out.get("match"), attribution=out.get("attribution"),
        explanation_initial=out.get("explanation_initial"),
        explanation_final=out.get("explanation_final"), aas_initial=out.get("aas_initial"),
        aas_final=out.get("aas_final"), aas_history=out.get("aas_history", []),
        revisions=out.get("revisions", []), verdict=out["verdict"],
        appeal_draft=out.get("appeal_draft"), appeal=out.get("appeal"))
