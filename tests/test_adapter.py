"""app/adapter.py on a hand-made CaseResult (no torch, Streamlit or Ollama)."""

import math

import pytest

from app.adapter import clause_words, has_evidence, reflection_note, summary, to_view
from app.result_view import cited_indices
from claimlens.explain.aas import alignment_score
from claimlens.schemas import (
    Attribution,
    CaseResult,
    Chunk,
    CitedSpan,
    Explanation,
    MatchResult,
    RetrievedClause,
    Verdict,
)

CFG = {"agent": {"tau_high": 0.8, "tau_low": 0.4, "tau_aas": 0.6, "max_rounds": 2}}
CLAUSE = "Pre-existing diseases are excluded, until 36 months."
TOKENS = ["pre", "-", "existing", "diseases", "are", "excluded", ",", "until", "36", "months", "."]
SCORES = [0.3, 0.0, 0.2, 0.4, -0.1, 0.5, 0.0, 0.1, 0.6, 0.7, 0.0]
ATTR = Attribution(clause_id="c1", method="integrated_gradients", tokens=TOKENS, scores=SCORES)


def hit(chunk_id: str, text: str, logit: float, section: str = "4.2 Exclusions") -> RetrievedClause:
    chunk = Chunk(chunk_id=chunk_id, doc_id="p", text=text, section_path=section, page=12)
    return RetrievedClause(chunk=chunk, score=logit)


def make_result(**overrides) -> CaseResult:
    first = Explanation(text="First <draft>.", cited_spans=[CitedSpan(text="are excluded")])
    final = Explanation(text="Final.", cited_spans=[CitedSpan(text="Pre-existing diseases")])
    values = dict(
        case_id="case_x", reason="Claim rejected: pre-existing disease.",
        top_clauses=[hit("c1", CLAUSE, 2.0), hit("c2", "Other clause text.", 0.0, "2.1 Terms")],
        match=MatchResult(clause_id="c1", score=2.0, prob=0.88), attribution=ATTR,
        explanation_initial=first, explanation_final=final, aas_initial=0.2, aas_final=0.4,
        aas_history=[("initial", 0.2), ("revision_1", 0.4)], verdict=Verdict.SUPPORTED,
        appeal_draft=None)
    return CaseResult(**{**values, **overrides})


def test_maps_fields_and_thresholds():
    view = to_view(make_result(), "care_supreme", CFG)
    assert (view["case_id"], view["policy_id"], view["verdict"]) == (
        "case_x", "care_supreme", "SUPPORTED")
    assert view["sample"] is False
    assert view["support_score"] == 0.88
    assert (view["tau_low"], view["tau_high"], view["aas_target"]) == (0.4, 0.8, 0.6)
    assert view["clause"]["section_path"] == "4.2 Exclusions" and view["clause"]["page"] == 12
    assert view["explanations"]["initial"]["aas"] == 0.2
    assert view["explanations"]["final"]["aas"] == 0.4
    assert view["explanations"]["final"]["spans"] == [
        {"source": "clause", "text": "Pre-existing diseases"}]
    assert view["appeal_draft"] is None


def test_other_clauses_skip_the_best_and_turn_logits_into_probabilities():
    view = to_view(make_result(), "p", CFG)
    assert [o["section_path"] for o in view["other_clauses"]] == ["2.1 Terms"]
    assert math.isclose(view["other_clauses"][0]["score"], 0.5)


def test_clause_words_keep_original_text_and_sum_positive_mass():
    out = clause_words(ATTR, CLAUSE)
    assert " ".join(w["text"] for w in out) == CLAUSE
    assert math.isclose(out[0]["score"], 0.5)  # pre + - + existing
    assert out[2]["score"] == 0.0  # negative attribution counts for nothing, as in AAS
    total = sum(w["score"] for w in out)
    assert math.isclose(total, sum(s for s in SCORES if s > 0))


def test_clause_words_survive_a_truncated_attribution():
    short = Attribution(clause_id="c1", method="integrated_gradients", tokens=TOKENS[:4],
                        scores=SCORES[:4])
    out = clause_words(short, CLAUSE)
    assert len(out) == len(CLAUSE.split())
    assert out[-1]["score"] == 0.0


def test_spans_the_agent_cites_underline_the_right_words():
    view = to_view(make_result(), "p", CFG)
    spans = view["explanations"]["final"]["spans"]
    hits, missed = cited_indices(view["clause"]["words"], spans, "clause")
    assert hits == {0, 1} and missed == []


def test_view_words_give_the_same_mass_as_aas():
    res = make_result()
    words = to_view(res, "p", CFG)["clause"]["words"]
    spans = res.explanation_final.cited_spans
    hits, _ = cited_indices(words, [s.model_dump() for s in spans], "clause")
    share = sum(words[i]["score"] for i in hits) / sum(w["score"] for w in words)
    assert math.isclose(share, alignment_score(ATTR, spans, CLAUSE).aas)


def test_span_aas_rejected_is_listed_but_never_underlined():
    # not an exact substring (case differs), so AAS scores it zero; the view must agree
    loose = Explanation(text="t", cited_spans=[CitedSpan(text="ARE EXCLUDED")])
    view = to_view(make_result(explanation_final=loose), "p", CFG)
    spans = view["explanations"]["final"]["spans"]
    assert spans == [{"source": "unmatched", "text": "ARE EXCLUDED"}]
    assert cited_indices(view["clause"]["words"], spans, "clause") == (set(), [])


def test_reason_has_no_attribution_so_scores_are_zero():
    view = to_view(make_result(), "p", CFG)
    assert [w["text"] for w in view["reason"]][:2] == ["Claim", "rejected:"]
    assert all(w["score"] == 0.0 for w in view["reason"])


def test_reflection_note_reports_recorded_scores_only():
    note = reflection_note(make_result(), 0.6)
    assert "1 revision round ran" in note and "0.20 to 0.40" in note and "best version" in note
    assert "does not change" in note


def test_reflection_note_when_no_revision_won_or_ran():
    no_gain = make_result(aas_final=0.2, explanation_final=make_result().explanation_initial,
                          aas_history=[("initial", 0.2), ("revision_1", 0.1)])
    assert "None beat the first draft" in reflection_note(no_gain, 0.6)
    met = make_result(aas_initial=0.7, aas_final=0.7, aas_history=[("initial", 0.7)])
    assert "met the alignment target" in reflection_note(met, 0.6)
    failed = make_result(aas_initial=0.2, aas_final=0.2, aas_history=[("initial", 0.2)])
    assert "No revision was completed" in reflection_note(failed, 0.6)


def test_nothing_retrieved_has_no_view():
    empty = CaseResult(case_id="x", reason="r", top_clauses=[],
                       verdict=Verdict.INSUFFICIENT_EVIDENCE)
    assert not has_evidence(empty)
    with pytest.raises(ValueError):
        to_view(empty, "p", CFG)


def test_summary_cuts_at_a_word_boundary():
    assert summary("short text") == "short text"
    out = summary("word " * 100, limit=20)
    assert out.endswith("...") and len(out) <= 23 and "wor..." not in out
