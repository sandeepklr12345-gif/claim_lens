"""LLM explanation with cited spans, and the reflection loop driven by AAS (SPEC.md 7.5).

Torch-free: attribution and the LLM are injected, so tests run with FakeLLM and a hand-made
Attribution.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from claimlens.explain.aas import alignment_score, feedback_words
from claimlens.llm.client import LLM
from claimlens.schemas import AlignmentResult, Attribution, CitedSpan, Explanation

log = logging.getLogger(__name__)
SYSTEM = "You explain health-insurance claim audits. Reply with JSON only."
TAU_AAS = 0.6  # reflect when AAS is below this (SPEC.md 7.5)
MAX_ROUNDS = 2

_FORMAT = (
    '{"explanation": "<2-4 sentences>", '
    '"cited_spans": [{"source": "clause", "text": "<exact words copied from the clause>"}]}'
)

EXPLAIN_PROMPT = """Rejection reason given by the insurer: {reason}

Policy clause {clause_id}:
\"\"\"{clause}\"\"\"

An automatic matcher scored how well this clause supports the rejection: {prob:.2f}
(0 = does not support, 1 = fully supports). Conclusion to explain: the clause {stance}.

Explain in 2-4 sentences why the clause {stance}. Support your claims by quoting words from
the clause: give 2-5 cited_spans, each 3-15 words copied EXACTLY from the clause (no changes,
no paraphrase). Cite only the clause.
Reply as JSON: {fmt}"""

REVISE_PROMPT = """Rejection reason given by the insurer: {reason}

Policy clause {clause_id}:
\"\"\"{clause}\"\"\"

Your previous explanation: {explanation}
Your previous cited spans: {spans}

An attribution analysis shows which clause words most influenced the matcher:
- Influential words you did NOT cite: {missing}
- Words you cited that had little influence: {weak}

Revise the explanation. Keep the SAME conclusion (the clause {stance}). Cite spans, copied
EXACTLY from the clause, that include the influential words, and drop spans built on the
low-influence words. 2-4 sentences, 2-5 cited_spans of 3-15 words each.
Reply as JSON: {fmt}"""


def parse_explanation(raw: dict[str, Any]) -> Explanation:
    spans = []
    for s in raw.get("cited_spans") or []:
        if isinstance(s, dict) and isinstance(s.get("text"), str):
            # LLMs write things like "Policy clause care_supreme-0051" here; only an explicit
            # mention of the reason makes it a reason span, everything else is a clause span.
            src = "reason" if "reason" in str(s.get("source", "")).lower() else "clause"
            spans.append(CitedSpan(text=s["text"], source=src))
        elif isinstance(s, str):
            spans.append(CitedSpan(text=s))
    return Explanation(text=str(raw.get("explanation", "")).strip(), cited_spans=spans)


def explain(llm: LLM, reason: str, clause_id: str, clause: str, prob: float, stance: str
            ) -> Explanation:
    prompt = EXPLAIN_PROMPT.format(reason=reason, clause_id=clause_id, clause=clause, prob=prob,
                                   stance=stance, fmt=_FORMAT)
    return parse_explanation(llm.generate_json(prompt, SYSTEM))


def revise(llm: LLM, reason: str, clause_id: str, clause: str, prev: Explanation, stance: str,
           missing: list[str], weak: list[str]) -> Explanation:
    prompt = REVISE_PROMPT.format(
        reason=reason, clause_id=clause_id, clause=clause, explanation=prev.text,
        spans=[s.text for s in prev.cited_spans], missing=missing or "(none)",
        weak=weak or "(none)", stance=stance, fmt=_FORMAT)
    return parse_explanation(llm.generate_json(prompt, SYSTEM))


@dataclass
class ReflectionResult:
    initial: Explanation
    final: Explanation
    aas_initial: float
    aas_final: float
    history: list[tuple[str, float]] = field(default_factory=list)  # (label, AAS)


def reflect(llm: LLM, attr: Attribution, reason: str, clause_id: str, clause: str, prob: float,
            stance: str, tau: float = TAU_AAS, max_rounds: int = MAX_ROUNDS,
            aas_fn: Callable[[list[CitedSpan]], AlignmentResult] | None = None,
            on_aas: Callable[[str, float], None] | None = None,
            ) -> ReflectionResult:
    """Explain, then revise up to `max_rounds` times while AAS < tau; keep the best.

    `aas_fn` lets the agent score spans through the explainer MCP tool; the default is the
    local function. `on_aas(label, aas)` is told each score as it arrives (live UI progress).
    """
    score = aas_fn or (lambda spans: alignment_score(attr, spans, clause))
    notify = on_aas or (lambda _label, _aas: None)
    initial = explain(llm, reason, clause_id, clause, prob, stance)
    res = score(initial.cited_spans)
    notify("initial", res.aas)
    best, best_aas = initial, res.aas
    history = [("initial", res.aas)]
    cur, cur_idx, cur_aas = initial, set(res.cited_idx), res.aas
    for r in range(1, max_rounds + 1):
        if cur_aas >= tau:
            break
        missing, weak = feedback_words(attr, cur_idx)
        try:
            cur = revise(llm, reason, clause_id, clause, cur, stance, missing, weak)
        except Exception as e:  # noqa: BLE001 - keep the best so far if the LLM fails
            log.warning("revision %d failed: %s", r, e)
            break
        rr = score(cur.cited_spans)
        cur_idx, cur_aas = set(rr.cited_idx), rr.aas
        history.append((f"revision_{r}", rr.aas))
        notify(f"revision_{r}", rr.aas)
        if rr.aas > best_aas:
            best, best_aas = cur, rr.aas
    return ReflectionResult(initial, best, res.aas, best_aas, history)
