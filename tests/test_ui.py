"""Streamlit smoke tests (local only: need streamlit, and the full run needs Ollama + models)."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.model
APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def test_page_loads_and_lists_sample_cases():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "ClaimLens"
    assert len(at.selectbox[0].options) == 20
    assert any("not legal advice" in c.value for c in at.caption)


def test_run_button_shows_verdict_and_evidence():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=600).run()
    at.button[0].click().run(timeout=600)
    assert not at.exception
    html = " ".join(m.value for m in at.markdown)
    assert any(v in html for v in ("SUPPORTED", "NOT SUPPORTED", "INSUFFICIENT EVIDENCE"))
    assert "Explanation (initial)" in html and "Explanation (after reflection)" in html
