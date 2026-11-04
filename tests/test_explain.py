import pytest

from claimlens.explain.aas import alignment_score, feedback_words, words
from claimlens.explain.explain import parse_explanation, reflect
from claimlens.llm.client import FakeLLM
from claimlens.schemas import Attribution, CitedSpan

CLAUSE = "Pre-existing diseases are covered only after 36 months of continuous coverage."


def make_attr() -> Attribution:
    toks = words(CLAUSE)
    # hand-made attributions: mass sits on "36" and "months"; "covered" is negative
    scores = [0.0] * len(toks)
    scores[toks.index("36")] = 0.6
    scores[toks.index("months")] = 0.3
    scores[toks.index("only")] = 0.1
    scores[toks.index("covered")] = -0.4
    return Attribution(clause_id="c", tokens=toks, scores=scores)


def test_aas_hand_example():
    attr = make_attr()
    full = alignment_score(attr, [CitedSpan(text="after 36 months")], CLAUSE)
    assert full.aas == pytest.approx(0.9)  # (0.6 + 0.3) / 1.0 positive mass
    none = alignment_score(attr, [CitedSpan(text="Pre-existing diseases")], CLAUSE)
    assert none.aas == 0.0 and not none.unmatched
    # a span that is not an exact substring of the clause counts as zero and is reported
    bad = alignment_score(attr, [CitedSpan(text="after 12 months")], CLAUSE)
    assert bad.aas == 0.0 and bad.unmatched == ["after 12 months"]
    # reason-sourced spans cannot earn clause attribution
    rs = alignment_score(attr, [CitedSpan(text="36 months", source="reason")], CLAUSE)
    assert rs.aas == 0.0 and rs.unmatched


def test_feedback_lists_uncited_influential_and_weak_cited():
    attr = make_attr()
    res = alignment_score(attr, [CitedSpan(text="Pre-existing diseases")], CLAUSE)
    missing, weak = feedback_words(attr, set(res.cited_idx))
    assert missing[0] == "36" and "months" in missing
    assert "pre" in weak or "diseases" in weak


def test_parse_explanation_normalises_source():
    raw = {"explanation": "e", "cited_spans": [
        {"source": "Policy clause care_supreme-0051", "text": "36 months"},
        {"source": "rejection reason", "text": "x"}, "plain string span"]}
    e = parse_explanation(raw)
    assert [s.source for s in e.cited_spans] == ["clause", "reason", "clause"]


def test_reflection_improves_and_keeps_best():
    attr = make_attr()
    llm = FakeLLM([
        {
            "explanation": "weak",
            "cited_spans": [{"source": "clause", "text": "Pre-existing diseases"}],
        },
        {"explanation": "better", "cited_spans": [{"source": "clause", "text": "after 36 months"}]},
    ])
    res = reflect(llm, attr, "reason", "c", CLAUSE, 0.8, "supports the rejection")
    assert res.aas_initial == 0.0
    assert res.aas_final == pytest.approx(0.9) and res.final.text == "better"
    assert [h[0] for h in res.history] == ["initial", "revision_1"]
    assert "36" in llm.calls[1]  # feedback names the influential uncited word


def test_reflection_skipped_when_aas_high_and_best_kept_when_worse():
    attr = make_attr()
    good = {
        "explanation": "good",
        "cited_spans": [{"source": "clause", "text": "after 36 months"}],
    }
    llm = FakeLLM([good])
    res = reflect(llm, attr, "r", "c", CLAUSE, 0.8, "supports the rejection")
    assert len(llm.calls) == 1 and res.final.text == "good"

    poor = {"explanation": "poor", "cited_spans": [{"source": "clause", "text": "diseases"}]}
    worse = {"explanation": "worse", "cited_spans": [{"source": "clause", "text": "continuous"}]}
    llm2 = FakeLLM([poor, worse, worse])
    res2 = reflect(llm2, attr, "r", "c", CLAUSE, 0.8, "supports the rejection")
    assert res2.final.text == "poor" and len(llm2.calls) == 3  # initial + 2 rounds, best kept
