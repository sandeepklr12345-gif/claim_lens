"""ClaimLens UI: pick a sample case (or paste a letter), run the agent, inspect the evidence.

    streamlit run app\\streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a fresh clone

import streamlit as st  # noqa: E402

from claimlens.agent.graph import run_case  # noqa: E402
from claimlens.agent.tools import DirectTools  # noqa: E402
from claimlens.agent.verdict import DISCLAIMER  # noqa: E402
from claimlens.explain.aas import span_indices  # noqa: E402
from claimlens.explain.render import STYLE, attribution_html  # noqa: E402
from claimlens.llm.client import OllamaClient  # noqa: E402
from claimlens.rag.parse import list_policies  # noqa: E402
from claimlens.schemas import Case, CaseResult, Verdict  # noqa: E402
from claimlens.utils.config import ROOT, load_config  # noqa: E402

BANNER = {
    Verdict.SUPPORTED: ("#d1fae5", "#065f46", "SUPPORTED: the policy supports the rejection"),
    Verdict.NOT_SUPPORTED: (
        "#fee2e2", "#991b1b", "NOT SUPPORTED: no clause supports the rejection"),
    Verdict.INSUFFICIENT_EVIDENCE: (
        "#fef3c7", "#92400e", "INSUFFICIENT EVIDENCE: the system abstains"),
}


@st.cache_resource
def resources() -> tuple[DirectTools, OllamaClient, dict]:
    return DirectTools(), OllamaClient(), load_config()


@st.cache_data
def sample_cases() -> dict[str, Case]:
    files = sorted((ROOT / "data" / "cases").glob("case_*.json"))
    cases = [Case.model_validate_json(f.read_text(encoding="utf-8")) for f in files]
    return {f"{c.case_id} ({c.policy_id}, gold {c.gold_verdict.value})": c for c in cases}


def show_spans(res: CaseResult, expl, aas: float | None, title: str) -> None:
    st.markdown(f"**{title}**" + (f" &nbsp; AAS = `{aas:.3f}`" if aas is not None else ""))
    st.write(expl.text)
    clause = res.top_clauses[0].chunk.text
    _, unmatched = span_indices(res.attribution, expl.cited_spans, clause)
    for s in expl.cited_spans:
        flag = " (not found in the clause, counts as 0)" if s.text in unmatched else ""
        st.markdown(f"- \"{s.text}\"{flag}")


def render(res: CaseResult) -> None:
    bg, fg, text = BANNER[res.verdict]
    st.markdown(
        f"<div style='background:{bg};color:{fg};padding:.8rem 1rem;border-radius:8px;"
        f"font-weight:600'>{text}</div>", unsafe_allow_html=True)
    st.caption(DISCLAIMER)
    st.markdown(f"**Extracted rejection reason:** {res.reason}")
    if not res.top_clauses:
        st.warning("No clause was retrieved, so the system cannot judge this rejection.")
        return
    best = res.top_clauses[0]
    st.markdown(f"**Best clause** `{best.chunk.chunk_id}` &middot; page {best.chunk.page} &middot; "
                f"{best.chunk.section_path} &middot; match probability "
                f"`{res.match.prob:.3f}`")
    st.markdown(f"<style>{STYLE}</style><div class='clause'>{attribution_html(res.attribution)}"
                "</div>", unsafe_allow_html=True)
    st.caption("Orange shading = positive Integrated Gradients attribution toward the match score.")
    c1, c2 = st.columns(2)
    with c1:
        show_spans(res, res.explanation_initial, res.aas_initial, "Explanation (initial)")
    with c2:
        show_spans(res, res.explanation_final, res.aas_final, "Explanation (after reflection)")
    if len(res.aas_history) > 1:
        st.caption("AAS per round: " + " → ".join(f"{k} {v:.3f}" for k, v in res.aas_history))
    with st.expander("Other retrieved clauses"):
        for h in res.top_clauses[1:]:
            st.markdown(
                f"`{h.chunk.chunk_id}` {h.chunk.section_path}  (re-rank logit {h.score:.2f})")
            st.write(h.chunk.text[:400] + "...")
    if res.appeal_draft:
        st.subheader("Appeal draft (template)")
        st.text_area("Draft", res.appeal_draft, height=320, label_visibility="collapsed")


def main() -> None:
    st.set_page_config(page_title="ClaimLens", layout="wide")
    st.title("ClaimLens")
    st.caption("Explainable audit of health-insurance claim rejections. " + DISCLAIMER)
    cases = sample_cases()
    mode = st.radio("Input", ["Sample case", "Paste a letter"], horizontal=True)
    if mode == "Sample case":
        label = st.selectbox("Case (synthetic)", list(cases))
        case = cases[label]
        st.text_area("Rejection letter", case.letter_text, height=140, disabled=True)
    else:
        policy = st.selectbox("Policy", list_policies())
        letter = st.text_area("Rejection letter text", height=160)
        case = Case(case_id="pasted", policy_id=policy, letter_text=letter, reason="",
                    gold_clause_ids=[], gold_verdict=Verdict.INSUFFICIENT_EVIDENCE,
                    source="user")
    if st.button("Run audit", type="primary", disabled=not case.letter_text.strip()):
        tools, llm, cfg = resources()
        with st.spinner("Retrieving, scoring, attributing and explaining (about 30-60 s)..."):
            try:
                st.session_state["result"] = run_case(case, tools, llm, cfg)
            except Exception as e:  # noqa: BLE001 - show the error instead of a stack trace
                st.error(f"Run failed: {e}. Is Ollama running at {cfg['llm']['host']}?")
                st.session_state.pop("result", None)
    if "result" in st.session_state:
        render(st.session_state["result"])


main()
