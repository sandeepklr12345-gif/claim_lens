"""Intake HTML and the live progress panel (no Streamlit needed)."""

import sys
from types import SimpleNamespace

from app.intake_view import (
    RunProgress,
    case_row_html,
    friendly_error,
    header_html,
    progress_html,
    short_title,
    steps_html,
)
from claimlens.llm.client import OllamaClient
from claimlens.schemas import Verdict


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def progress() -> tuple[RunProgress, Clock]:
    clock = Clock()
    return RunProgress("care_supreme", "qwen2.5:7b", 0.6, 2, clock=clock), clock


def clause(section: str = "4.2 Exclusions") -> SimpleNamespace:
    return SimpleNamespace(chunk=SimpleNamespace(section_path=section, chunk_id="c1", page=12))


def test_short_title_truncates_long_reasons_only():
    assert short_title("Claim not covered.") == "Claim not covered"
    assert short_title("one two three four five six seven eight nine.") == (
        "one two three four five six seven eight...")


def test_case_row_shows_chip_only_for_synthetic_and_escapes():
    row = case_row_html("case_001", "<b>reason</b>", "care", "synthetic", True)
    assert "Synthetic data" in row and "&lt;b&gt;" in row and "cl-row on" in row
    assert "Synthetic data" not in case_row_html("x", "r", "care", "user", False)


def test_header_has_disclaimer_and_steps_has_all_five():
    assert "not legal advice" in header_html()
    html = steps_html()
    for name in ("Read the letter", "Find clauses", "Match", "Explain and check", "Verdict"):
        assert name in html


def test_panel_walks_through_a_full_run_with_measured_times():
    p, clock = progress()
    p.begin()
    assert p.rows[0].status == "active" and "care_supreme" in p.rows[0].detail
    clock.now = 2.0
    p.handle("ready", {"device": "cuda"})
    clock.now = 3.0
    p.handle("parse", {"reason": "Not covered."})
    clock.now = 3.5
    p.handle("retrieve", {"top_clauses": [clause(), clause()]})
    clock.now = 3.6
    p.handle("match", {"match": SimpleNamespace(prob=0.91)})
    p.handle("attribution", {"words": 40})
    assert "40 words" in p.rows[4].detail
    p.handle("aas", {"label": "initial", "aas": 0.1})
    assert "revision 1 of 2" in p.rows[4].detail
    p.handle("aas", {"label": "revision_1", "aas": 0.2})
    assert "revision 2 of 2" in p.rows[4].detail
    p.handle("aas", {"label": "revision_2", "aas": 0.3})
    assert "No revisions left" in p.rows[4].detail
    clock.now = 13.6
    p.handle("explain_reflect", {"aas_final": 0.3, "aas_history": [("initial", 0.1),
                                                                  ("revision_1", 0.3)]})
    p.handle("verdict", {"verdict": Verdict.NOT_SUPPORTED})
    p.handle("draft", {})
    assert all(r.status == "done" for r in p.rows)
    assert p.rows[0].seconds == 2.0 and p.rows[4].seconds == 10.0
    assert p.total == 13.6
    html = progress_html(p)
    assert "Audit finished" in html and "NOT SUPPORTED" in html and "Matcher on cuda" in html
    assert "13.6 s" in html and "0.91" in html and "Best: 4.2 Exclusions, page 12" in html


def test_panel_ticks_while_a_step_runs_and_shows_last_run_time():
    p, clock = progress()
    p.last_seconds = 41.0
    p.begin()
    clock.now = 7.5
    html = progress_html(p)
    assert "Audit running" in html and "7.5 s" in html and "Last run took 41.0 s" in html


def test_panel_escapes_llm_text():
    p, _ = progress()
    p.begin()
    p.handle("ready", {"device": "cpu"})
    p.handle("parse", {"reason": "<script>alert(1)</script>"})
    assert "<script>" not in progress_html(p)


def test_target_met_stops_asking_for_revisions():
    p, _ = progress()
    p.handle("aas", {"label": "initial", "aas": 0.7})
    assert "Target met" in p.rows[4].detail


def test_nothing_retrieved_is_reported():
    p, _ = progress()
    p.begin()
    p.handle("ready", {"device": "cpu"})
    p.handle("parse", {"reason": "r"})
    p.handle("retrieve", {"top_clauses": []})
    p.handle("match", {})
    p.handle("explain_reflect", {})
    p.handle("verdict", {"verdict": Verdict.INSUFFICIENT_EVIDENCE})
    p.handle("draft", {"appeal_draft": None})
    html = progress_html(p)
    assert "No clause found" in html and "Skipped" in html and "NOT ENOUGH EVIDENCE" in html


def test_stop_marks_the_active_step():
    p, clock = progress()
    p.begin()
    clock.now = 4.0
    p.stop()
    assert p.rows[0].status == "stopped" and p.total == 4.0
    assert "Audit stopped" in progress_html(p)


def test_friendly_error_never_shows_the_raw_exception():
    conn = type("ConnectionError", (Exception,), {})("secret traceback text")
    assert "Check Ollama" in friendly_error(conn, "http://h")
    read = type("ReadTimeout", (Exception,), {})("x")
    assert "too long" in friendly_error(read, "http://h")
    assert "usable output" in friendly_error(ValueError("LLM did not return valid JSON"), "h")
    assert "secret" not in friendly_error(RuntimeError("secret"), "h")


def fake_requests(monkeypatch, tags=None, error=None) -> None:
    def get(url, timeout):  # noqa: ANN001, ANN202
        if error:
            raise error
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: tags)

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(get=get))


def test_health_reports_server_down_and_missing_model(monkeypatch):
    llm = OllamaClient(host="http://h", model="m:7b")
    fake_requests(monkeypatch, error=OSError("refused"))
    assert "not answering at http://h" in llm.health()
    fake_requests(monkeypatch, tags={"models": [{"name": "other"}]})
    assert "ollama pull m:7b" in llm.health()
    fake_requests(monkeypatch, tags={"models": [{"name": "m:7b"}]})
    assert llm.health() is None
