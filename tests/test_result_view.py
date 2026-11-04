"""Unit tests for the pure helpers in app/result_view.py (no Streamlit needed)."""
from app.result_view import (
    banner_html,
    cited_indices,
    esc,
    esc_multiline,
    marker_percent,
    shade,
    verdict_info,
    words_html,
)

WORDS = [{"text": t, "score": s} for t, s in
         [("The", 0.0), ("pre-existing", 0.5), ("disease", 0.4), ("is", 0.0), ("excluded.", 0.3)]]


def test_cited_indices_matches_consecutive_words_ignoring_case_and_punctuation():
    hit, missed = cited_indices(WORDS, [{"source": "clause", "text": "Pre-Existing disease"},
                                        {"source": "clause", "text": "excluded"}], "clause")
    assert hit == {1, 2, 4}
    assert missed == []


def test_cited_indices_reports_unmatched_and_skips_other_source():
    hit, missed = cited_indices(WORDS, [{"source": "clause", "text": "not in text"},
                                        {"source": "reason", "text": "disease"}], "clause")
    assert hit == set()
    assert missed == ["not in text"]


def test_escaping_blocks_html_and_math():
    assert "<script>" not in esc("<script>alert(1)</script>")
    assert "$" not in esc("cost $5 and $6")
    assert "<br>" in esc_multiline("a\n\nb")
    assert "\n" not in esc_multiline("a\n\nb")


def test_words_html_escapes_words_and_marks_citations():
    out = words_html([{"text": "<b>", "score": 1.0}, {"text": "ok", "score": 0.0}], {1})
    assert "&lt;b&gt;" in out
    assert out.count("cl-u") == 1


def test_shade_only_for_positive_scores_and_clamps():
    assert shade(-0.2, 1.0) == "transparent"
    assert shade(0.0, 1.0) == "transparent"
    assert shade(2.0, 1.0) == shade(1.0, 1.0)


def test_marker_percent_clamps():
    assert marker_percent(-1) == 0
    assert marker_percent(1.7) == 100
    assert marker_percent(0.87) == 87


def test_unknown_verdict_falls_back_to_abstain():
    assert verdict_info("???")[0] == "NOT ENOUGH EVIDENCE"


def test_banner_shows_verdict_and_thresholds():
    out = banner_html({"verdict": "SUPPORTED", "support_score": 0.87})
    assert "SUPPORTED" in out and "0.87" in out and "0.3" in out and "0.7" in out
