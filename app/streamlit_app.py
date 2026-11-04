"""ClaimLens UI: pick a sample case or paste a letter, run the agent, inspect the evidence.

    streamlit run app\\streamlit_app.py

Screen logic only. HTML lives in app/intake_view.py and app/result_view.py, and the
CaseResult mapping in app/adapter.py. Results always come from the real pipeline.
"""

from __future__ import annotations

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a fresh clone

import streamlit as st  # noqa: E402

from app.adapter import has_evidence, to_view  # noqa: E402
from app.intake_view import (  # noqa: E402
    CSS,
    RunProgress,
    case_row_html,
    fmt_seconds,
    friendly_error,
    header_html,
    hero_html,
    progress_html,
    steps_html,
)
from app.result_view import CSS as BASE_CSS  # noqa: E402
from app.result_view import DISCLAIMER, render_result  # noqa: E402
from claimlens import core  # noqa: E402
from claimlens.agent.graph import run_case  # noqa: E402
from claimlens.agent.tools import DirectTools  # noqa: E402
from claimlens.llm.client import OllamaClient  # noqa: E402
from claimlens.rag.parse import list_policies  # noqa: E402
from claimlens.schemas import Case, CaseResult, Verdict  # noqa: E402
from claimlens.utils.config import ROOT, load_config  # noqa: E402

log = logging.getLogger(__name__)
REFRESH_SECONDS = 0.5  # how often the progress panel redraws during a run


def html(text: str) -> None:
    st.markdown(text, unsafe_allow_html=True)


@st.cache_resource
def resources() -> tuple[DirectTools, OllamaClient, dict]:
    return DirectTools(), OllamaClient(), load_config()


@st.cache_data
def sample_cases() -> list[Case]:
    files = sorted((ROOT / "data" / "cases").glob("*.json"))
    return [Case.model_validate_json(f.read_text(encoding="utf-8")) for f in files]


def select_case(case: Case) -> None:
    st.session_state["selected"] = case.case_id
    st.session_state["policy"] = case.policy_id


def audit(case: Case, progress: RunProgress, tools: DirectTools, llm: OllamaClient,
          cfg: dict) -> CaseResult:
    """Worker thread: no Streamlit calls here, the page reads `progress` as it changes."""
    progress.begin()
    core.get_retriever(case.policy_id)  # cached: only the first run pays for these two
    matcher = core.get_matcher()
    progress.handle("ready", {"device": matcher.device})
    return run_case(case, tools, llm, cfg, on_progress=progress.handle)


def run_audit(case: Case, slot) -> None:  # noqa: ANN001 - slot is a Streamlit placeholder
    tools, llm, cfg = resources()
    problem = llm.health()
    if problem:
        slot.error(problem)
        return
    agent = cfg["agent"]
    progress = RunProgress(case.policy_id, llm.model, agent["tau_aas"], agent["max_rounds"],
                           last_seconds=st.session_state.get("last_seconds"))
    with slot.container():
        panel = st.empty()
        with st.spinner("Running the audit. Keep this page open."):
            with ThreadPoolExecutor(max_workers=1) as pool:
                job = pool.submit(audit, case, progress, tools, llm, cfg)
                while not job.done():
                    panel.markdown(progress_html(progress), unsafe_allow_html=True)
                    time.sleep(REFRESH_SECONDS)
        try:
            result = job.result()
        except Exception as e:  # noqa: BLE001 - show plain language, keep the detail in the log
            log.exception("audit failed for %s", case.case_id)
            progress.stop()
            panel.markdown(progress_html(progress), unsafe_allow_html=True)
            st.error(friendly_error(e, llm.host))
            return
    st.session_state.update(result=result, result_policy=case.policy_id,
                            run_html=progress_html(progress),
                            last_seconds=progress.elapsed())
    st.rerun()


def case_list(cases: list[Case]) -> None:
    with st.container(height=520, border=False):
        for c in cases:
            on = c.case_id == st.session_state["selected"]
            text, action = st.columns([5, 1], vertical_alignment="center")
            with text:
                html(case_row_html(c.case_id, c.reason, c.policy_id, c.source, on))
            with action:
                st.button("Selected" if on else "Select", key=f"case_{c.case_id}", disabled=on,
                          on_click=select_case, args=(c,))


def intake() -> None:
    cases = sample_cases()
    if "selected" not in st.session_state:
        select_case(cases[0])
    chosen = next(c for c in cases if c.case_id == st.session_state["selected"])
    # Streamlit drops widget state while the result screen is up; restore it from the case.
    st.session_state.setdefault("policy", chosen.policy_id)
    html(hero_html())
    slot = st.empty()  # progress and errors go here, above the fold where the run was started
    left, right = st.columns([3, 2], gap="large")
    with left:
        html('<div class="cl-sub">Sample cases</div>')
        case_list(cases)
    with right:
        with st.container(key="intake_card"):
            policy = st.selectbox("Policy", list_policies(), key="policy")
            letter = st.text_area("Or paste a rejection letter", key="pasted", height=170,
                                  placeholder="Paste the letter text here")
            pasted = letter.strip()
            if pasted:
                html('<div class="cl-note">The pasted letter will be audited instead of the '
                     'selected sample.</div>')
            clicked = st.button("Run audit", key="run", type="primary", width="stretch")
            html('<div class="cl-note">Runs on this computer with a local model. Nothing is '
                 'sent to an outside service.</div>')
    html(steps_html())
    if clicked:
        if pasted:
            case = Case(case_id="pasted", policy_id=policy, letter_text=pasted, reason="",
                        gold_clause_ids=[], gold_verdict=Verdict.INSUFFICIENT_EVIDENCE,
                        source="user")
        else:
            case = chosen.model_copy(update={"policy_id": policy})
        run_audit(case, slot)


def result_screen() -> None:
    res: CaseResult = st.session_state["result"]
    policy = st.session_state["result_policy"]
    _, _, cfg = resources()
    if has_evidence(res):
        render_result(to_view(res, policy, cfg))
    else:
        st.warning(f"No clause was retrieved from {policy}, so ClaimLens cannot judge this "
                   f"rejection. {DISCLAIMER}.")
    elapsed = fmt_seconds(st.session_state["last_seconds"])
    with st.expander(f"How this run went ({elapsed})"):
        html(st.session_state["run_html"])
    if st.button("← Audit another letter", key="again"):
        for k in ("result", "result_policy", "run_html"):
            st.session_state.pop(k, None)
        st.rerun()
    html('<div class="cl-note">Influence shading comes from Integrated Gradients on the matching '
         'model.</div>')


def main() -> None:
    st.set_page_config(page_title="ClaimLens", layout="wide")
    html(BASE_CSS)
    html(CSS)
    html(header_html())
    if "result" in st.session_state:
        result_screen()
    else:
        intake()


main()
