"""Streamlit smoke tests (local only: need streamlit, and the full run needs Ollama + models)."""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.model
APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def page(at) -> str:
    return " ".join(m.value for m in at.markdown)


def test_page_loads_and_lists_sample_cases():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    html = page(at)
    assert "ClaimLens" in html and "not legal advice" in html
    assert html.count("Synthetic data") == 20
    assert sum(b.key.startswith("case_") for b in at.button) == 20
    assert "care_supreme" in at.selectbox(key="policy").options
    assert all(s in html for s in ("Read the letter", "Find clauses", "Explain and check"))


def test_selecting_a_case_moves_the_policy():
    from streamlit.testing.v1 import AppTest

    cases = sorted((Path(APP).parents[1] / "data" / "cases").glob("*.json"))
    loaded = [json.loads(f.read_text(encoding="utf-8")) for f in cases]
    other = next(c for c in loaded if c["policy_id"] != loaded[0]["policy_id"])
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert at.selectbox(key="policy").value == loaded[0]["policy_id"]
    at.button(key=f"case_{other['case_id']}").click().run()
    assert not at.exception
    assert at.selectbox(key="policy").value == other["policy_id"]
    assert at.session_state["selected"] == other["case_id"]


def test_ollama_down_gives_plain_language_error(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    st.cache_resource.clear()
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:9")
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.button(key="run").click().run(timeout=60)
    assert not at.exception
    assert any("Ollama is not answering" in e.value for e in at.error)
    st.cache_resource.clear()


def test_run_button_shows_progress_summary_verdict_and_evidence():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=600).run()
    at.button(key="case_case_002").click().run()  # hdfc_optima_secure, not the first case's policy
    policy = at.selectbox(key="policy").value
    at.button(key="run").click().run(timeout=600)
    assert not at.exception
    html = page(at)
    assert any(v in html for v in ("SUPPORTED", "NOT SUPPORTED", "NOT ENOUGH EVIDENCE"))
    assert "Best matching clause" in html and "Decision support, not legal advice" in html
    assert "Audit finished" in html  # the run summary kept under the result
    at.radio(key="cl_version").set_value("After reflection").run()
    assert not at.exception
    assert "What the check changed" in page(at)
    at.button(key="again").click().run()
    assert not at.exception
    assert "Was this claim rejection justified?" in page(at)
    assert at.selectbox(key="policy").value == policy  # widget state is dropped on the result page
