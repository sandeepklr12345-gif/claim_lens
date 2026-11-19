"""Convert a CaseResult into the dict `render_result` expects (keys: see app/result_view.py).

Pure and free of torch and Streamlit, so CI can test it.

What the schema does not hold, and how each gap is handled (no values are invented):
- `Attribution` covers clause words only, so reason words carry score 0.0 and stay unshaded.
- Attribution tokens are lower-cased and split on punctuation, while `cited_indices` matches
  whitespace words. `clause_words` rebuilds the clause's own words and adds up their tokens'
  positive attribution, the same mass AAS counts.
- There are no per-clause summaries, so "Other clauses" shows shortened clause text.
- The reflection note is built from `aas_history` and `feedback_words`; the agent stores no prose.
"""

from __future__ import annotations

import math
from difflib import SequenceMatcher
from typing import Any

from claimlens.explain.aas import feedback_words, span_indices, words
from claimlens.schemas import Attribution, CaseResult, Explanation, RetrievedClause

SUMMARY_CHARS = 160


def sigmoid(logit: float) -> float:
    """Logit to probability. Same formula as the matcher, which needs torch to import."""
    return 1.0 / (1.0 + math.exp(-logit))


def clause_words(attr: Attribution, clause_text: str) -> list[dict[str, Any]]:
    """The clause's whitespace words, each scored by the attribution tokens inside it.

    Tokens are aligned rather than counted because the tokenizer can truncate a long clause and
    can drop or merge characters. A word with no aligned token keeps score 0.0.
    """
    raw = clause_text.split()
    owners: list[int] = []  # word index for each flat token
    flat: list[str] = []
    for i, word in enumerate(raw):
        pieces = words(word)
        flat.extend(pieces)
        owners.extend([i] * len(pieces))
    totals = [0.0] * len(raw)
    matcher = SequenceMatcher(None, flat, attr.tokens, autojunk=False)
    for a, b, size in matcher.get_matching_blocks():
        for k in range(size):
            totals[owners[a + k]] += max(0.0, attr.scores[b + k])
    return [{"text": w, "score": s} for w, s in zip(raw, totals, strict=True)]


def summary(text: str, limit: int = SUMMARY_CHARS) -> str:
    """Shorten clause text at a word boundary. It is an excerpt, not a generated summary."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:") + "..."


def other_clause(hit: RetrievedClause) -> dict[str, Any]:
    return {
        "section_path": hit.chunk.section_path or hit.chunk.chunk_id,
        "summary": summary(hit.chunk.text),
        "score": sigmoid(hit.score),  # agent output: `score` is the re-rank logit
    }


def explanation_view(
    expl: Explanation, aas: float, attr: Attribution, clause_text: str
) -> dict[str, Any]:
    """Spans AAS could not place are tagged "unmatched": still listed, never underlined.

    AAS needs an exact substring, while `cited_indices` ignores case and edge punctuation. Without
    the tag, the page could underline words that scored zero, so it would disagree with AAS.
    """
    _, unmatched = span_indices(attr, expl.cited_spans, clause_text)
    spans = [
        {"source": "unmatched" if s.source == "clause" and s.text in unmatched else s.source,
         "text": s.text}
        for s in expl.cited_spans
    ]
    return {"text": expl.text, "spans": spans, "aas": aas}


def reflection_note(res: CaseResult, target: float) -> str:
    """What the check did, stated only from the recorded scores."""
    rounds = len(res.aas_history) - 1
    if rounds < 1:
        if res.aas_initial >= target:
            return "The first draft met the alignment target, so no revision was requested."
        return "No revision was completed, so the first draft stands."
    path = " to ".join(f"{aas:.2f}" for _, aas in res.aas_history)
    plural = "round" if rounds == 1 else "rounds"
    if res.aas_final > res.aas_initial:
        out = f"{rounds} revision {plural} ran (alignment {path}). The best version is shown."
    else:
        out = f"{rounds} revision {plural} ran (alignment {path}). None beat the first draft."
    clause = res.top_clauses[0].chunk.text
    cited, _ = span_indices(res.attribution, res.explanation_final.cited_spans, clause)
    missing, _ = feedback_words(res.attribution, cited)
    if missing:
        out += f" Influential words still not cited: {', '.join(missing)}."
    return out + " The verdict comes from the match score, so revising does not change it."


def has_evidence(res: CaseResult) -> bool:
    """False when nothing was retrieved, so there is no clause or explanation to show."""
    return bool(res.top_clauses) and res.attribution is not None and res.match is not None


def to_view(res: CaseResult, policy_id: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Build the `render_result` dict. Call only when `has_evidence(res)` is True."""
    if not has_evidence(res):
        raise ValueError(f"{res.case_id}: no clause was retrieved, nothing to render")
    agent = cfg["agent"]
    target = float(agent["tau_aas"])
    best = res.top_clauses[0].chunk
    return {
        "case_id": res.case_id,
        "policy_id": policy_id,
        "sample": False,  # real pipeline output, never the preview_result.py wireframe values
        "reason": [{"text": w, "score": 0.0} for w in res.reason.split()],  # no reason attribution
        "clause": {
            "section_path": best.section_path or best.chunk_id,
            "page": best.page,
            "words": clause_words(res.attribution, best.text),
        },
        "other_clauses": [other_clause(h) for h in res.top_clauses[1:]],
        "support_score": res.match.prob,
        "verdict": res.verdict.value,
        "explanations": {
            "initial": explanation_view(
                res.explanation_initial, res.aas_initial, res.attribution, best.text),
            "final": explanation_view(
                res.explanation_final, res.aas_final, res.attribution, best.text),
        },
        "reflection_note": reflection_note(res, target),
        "appeal_draft": res.appeal_draft,
        "appeal": res.appeal.model_dump(mode="json") if res.appeal else None,
        "tau_low": float(agent["tau_low"]),
        "tau_high": float(agent["tau_high"]),
        "aas_target": target,
    }
