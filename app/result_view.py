"""Audit-result view for the ClaimLens Streamlit app.

Pure helpers (no Streamlit import) build the HTML so they can be unit-tested in CI,
which does not install Streamlit. `render_result` imports Streamlit lazily.

Input is a plain dict so this works before claimlens/schemas.py is final. Adapt it with
`CaseResult.model_dump()` plus word-level attributions. Expected keys:

    case_id, policy_id, sample (bool; True shows a "sample data" chip)
    reason:  list[{"text": str, "score": float}]            word-level attribution
    clause:  {"section_path": str, "page": int, "words": list[{"text", "score"}]}
    other_clauses: list[{"section_path": str, "summary": str, "score": float}]
    support_score: float, verdict: SUPPORTED | NOT_SUPPORTED | INSUFFICIENT_EVIDENCE
    explanations: {"initial": {...}, "final": {...}} each {"text", "spans", "aas"}
        spans = list[{"source": "reason" | "clause", "text": str}]
    reflection_note: str (optional), appeal_draft: str (optional)
    tau_low, tau_high, aas_target (optional; defaults 0.3, 0.7, 0.6)

All text is escaped before it goes into HTML: letters are user-pasted.
Streamlit markdown rules respected here: no blank lines and no indentation inside
HTML blocks, and "$" is escaped so it is not read as math.
"""
from __future__ import annotations

import html
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

DISCLAIMER = "Decision support, not legal advice"

VERDICTS: dict[str, tuple[str, str, str, str]] = {
    "SUPPORTED": (
        "SUPPORTED",
        "The insurer's reason matches a clause in your policy.",
        "The cited clause appears to apply. This is a read of the wording, not a ruling on your claim.",
        "#0F766E",
    ),
    "NOT_SUPPORTED": (
        "NOT SUPPORTED",
        "The insurer's reason does not match a clause in your policy.",
        "No clause we found backs the stated reason. You can ask the insurer to name the clause it relied on.",
        "#B42318",
    ),
    "INSUFFICIENT_EVIDENCE": (
        "NOT ENOUGH EVIDENCE",
        "The policy wording is not clear enough to decide.",
        "The best clause only partly matches the reason, so the tool abstains instead of guessing.",
        "#B45309",
    ),
}

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:wght@400;600&display=swap');
.cl-card{background:#fff;border:1px solid #D9DFDA;border-radius:14px;padding:24px;color:#14201F;
font-family:'IBM Plex Sans',system-ui,sans-serif;margin-bottom:16px}
.cl-h{margin:0 0 12px;font-size:13px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;color:#55615E}
.cl-serif{font-family:'Source Serif 4',Georgia,serif !important}
.cl-mono{font-family:'IBM Plex Mono',monospace}
.cl-chip{display:inline-block;font-size:12px;padding:4px 10px;border-radius:999px;background:#EEF1ED;color:#3F4B48;margin-right:6px}
.cl-w{padding:2px 1px;border-radius:3px}
.cl-u{text-decoration:underline;text-decoration-thickness:2px;text-underline-offset:5px;text-decoration-color:#14201F}
</style>"""


# ---------- pure helpers ----------

def esc(text: Any) -> str:
    """Escape for HTML and for Streamlit's markdown math parsing."""
    return html.escape(str(text), quote=True).replace("$", "&#36;")


def esc_multiline(text: Any) -> str:
    """Escape, and turn newlines into <br> (blank lines would end the HTML block)."""
    return esc(text).replace("\r\n", "\n").replace("\n", "<br>")


def norm_token(token: str) -> str:
    return re.sub(r"^\W+|\W+$", "", token.lower())


def shade(score: float, max_score: float) -> str:
    """Background colour for one word. Only positive attribution is shaded."""
    if max_score <= 0 or score <= 0:
        return "transparent"
    return f"rgba(15,118,110,{0.6 * min(score / max_score, 1.0):.2f})"


def cited_indices(
    words: list[dict[str, Any]], spans: list[dict[str, str]], source: str
) -> tuple[set[int], list[str]]:
    """Word indices covered by cited spans of `source`; also the spans that did not match.

    A span is matched as a run of consecutive words (case and edge punctuation ignored).
    If a span occurs more than once, the first occurrence is used. Unmatched spans count
    as zero in AAS (SPEC 7.5), so they are returned and logged.
    """
    tokens = [norm_token(w["text"]) for w in words]
    hit: set[int] = set()
    missed: list[str] = []
    for span in spans:
        if span.get("source") != source:
            continue
        want = [t for t in (norm_token(p) for p in span["text"].split()) if t]
        found = False
        for i in range(len(tokens) - len(want) + 1):
            if want and tokens[i : i + len(want)] == want:
                hit.update(range(i, i + len(want)))
                found = True
                break
        if not found:
            missed.append(span["text"])
            logger.warning("cited span not found in %s: %r", source, span["text"])
    return hit, missed


def words_html(words: list[dict[str, Any]], cited: set[int] | frozenset[int] = frozenset()) -> str:
    top = max((float(w.get("score", 0.0)) for w in words), default=0.0)
    parts = []
    for i, w in enumerate(words):
        cls = "cl-w cl-u" if i in cited else "cl-w"
        bg = shade(float(w.get("score", 0.0)), top)
        parts.append(f'<span class="{cls}" style="background:{bg}">{esc(w["text"])}</span>')
    return " ".join(parts)


def marker_percent(score: float) -> int:
    return round(min(max(score, 0.0), 1.0) * 100)


def verdict_info(verdict: str) -> tuple[str, str, str, str]:
    if verdict not in VERDICTS:
        logger.warning("unknown verdict %r, showing as not enough evidence", verdict)
        verdict = "INSUFFICIENT_EVIDENCE"
    return VERDICTS[verdict]


def banner_html(result: dict[str, Any]) -> str:
    label, headline, sub, color = verdict_info(result["verdict"])
    score = float(result["support_score"])
    low = float(result.get("tau_low", 0.3))
    high = float(result.get("tau_high", 0.7))
    pos = marker_percent(score)
    lo, hi = marker_percent(low), marker_percent(high)
    bar = (
        f"linear-gradient(90deg,#E7D2CF 0%,#E7D2CF {lo}%,#EBE0C4 {lo}%,#EBE0C4 {hi}%,"
        f"#BFDDD8 {hi}%,#BFDDD8 100%)"
    )
    return "".join([
        '<div class="cl-card" style="display:flex;flex-wrap:wrap;gap:24px 48px;align-items:center">',
        '<div style="flex:1 1 380px">',
        f'<span class="cl-mono" style="font-weight:500;font-size:14px;letter-spacing:.06em;padding:6px 14px;'
        f'border-radius:8px;background:{color};color:#fff">{esc(label)}</span>',
        f'<span style="font-size:14px;color:#55615E;margin-left:12px">Match score {score:.2f}</span>',
        f'<h1 class="cl-serif" style="margin:12px 0 8px;font-size:34px;line-height:1.15;font-weight:600">{esc(headline)}</h1>',
        f'<p style="margin:0;font-size:16px;line-height:1.55;color:#55615E">{esc(sub)}</p>',
        "</div>",
        '<div style="flex:1 1 320px">',
        '<div style="font-size:13px;color:#55615E;margin-bottom:10px">Where the score falls</div>',
        f'<div style="position:relative;height:14px;border-radius:7px;background:{bar}">',
        f'<div style="position:absolute;left:{pos}%;top:-5px;width:4px;height:24px;border-radius:2px;background:#14201F"></div>',
        "</div>",
        '<div class="cl-mono" style="position:relative;height:18px;font-size:12px;color:#55615E;margin-top:4px">',
        '<span style="position:absolute;left:0">0</span>',
        f'<span style="position:absolute;left:{lo}%;transform:translateX(-50%)">{low:g}</span>',
        f'<span style="position:absolute;left:{hi}%;transform:translateX(-50%)">{high:g}</span>',
        '<span style="position:absolute;right:0">1</span>',
        "</div></div></div>",
    ])


def aas_html(aas: float, target: float) -> str:
    ok = aas >= target
    color = "#0F766E" if ok else "#B45309"
    note = "Cited words carry most of the influence" if ok else "Cited words carry little of the influence"
    return "".join([
        '<div style="margin:6px 0 14px">',
        '<div style="display:flex;justify-content:space-between;align-items:baseline">',
        '<span style="font-size:14px;font-weight:500">Alignment with the evidence</span>',
        f'<span class="cl-mono" style="font-weight:500;font-size:22px">{aas:.2f}</span></div>',
        '<div style="position:relative;height:10px;border-radius:5px;background:#E6EAE5;margin:6px 0">',
        f'<div style="height:10px;border-radius:5px;background:{color};width:{marker_percent(aas)}%"></div>',
        f'<div style="position:absolute;left:{marker_percent(target)}%;top:-4px;width:2px;height:18px;background:#14201F"></div>',
        "</div>",
        f'<div style="display:flex;justify-content:space-between;font-size:13px;color:#55615E"><span>{esc(note)}</span>'
        f"<span>Target {target:.2f}</span></div></div>",
    ])


def spans_html(spans: list[dict[str, str]]) -> str:
    chips = "".join(
        f'<span class="cl-serif" style="display:inline-block;font-size:15px;padding:4px 10px;border-radius:8px;'
        f'border:1px solid #B9C2BC;margin:0 8px 8px 0">{esc(s["text"])}</span>'
        for s in spans
    )
    return f'<div style="font-size:13px;color:#55615E;margin-bottom:8px">Words this version cites</div><div>{chips}</div>'


# ---------- Streamlit rendering ----------

def render_result(result: dict[str, Any]) -> None:
    import streamlit as st  # lazy: CI does not install Streamlit

    md = lambda s: st.markdown(s, unsafe_allow_html=True)  # noqa: E731
    target = float(result.get("aas_target", 0.6))
    expl = result["explanations"]

    md(CSS)
    chips = f'<span class="cl-chip">{esc(result["case_id"])} · {esc(result["policy_id"])}</span>'
    if result.get("sample"):
        chips += '<span class="cl-chip">Sample data, wireframe values</span>'
    chips += f'<span class="cl-chip">{DISCLAIMER}</span>'
    md(f"<div>{chips}</div>")
    md(banner_html(result))

    version = st.radio(
        "Explanation version", ["First draft", "After reflection"], horizontal=True, key="cl_version"
    )
    key = "initial" if version == "First draft" else "final"
    current = expl[key]
    clause_words = result["clause"]["words"]
    reason_words = result["reason"]
    c_cited, _ = cited_indices(clause_words, current["spans"], "clause")
    r_cited, _ = cited_indices(reason_words, current["spans"], "reason")

    left, right = st.columns([3, 2], gap="large")
    with left:
        md(
            '<div class="cl-card"><div class="cl-h">Reason in the letter</div>'
            f'<p class="cl-serif" style="margin:0;font-size:19px;line-height:1.75">{words_html(reason_words, r_cited)}</p></div>'
        )
        clause = result["clause"]
        md(
            '<div class="cl-card"><div class="cl-h">Best matching clause</div>'
            f'<div class="cl-mono" style="font-size:13px;color:#55615E;margin-bottom:12px">'
            f'{esc(clause["section_path"])} · p. {esc(clause["page"])}</div>'
            f'<p class="cl-serif" style="margin:0 0 14px;font-size:20px;line-height:1.9">{words_html(clause_words, c_cited)}</p>'
            '<div style="font-size:13px;color:#3F4B48;border-top:1px solid #E6EAE5;padding-top:12px">'
            'Darker shading means more influence on the match. Underline means the selected explanation cites it.</div></div>'
        )
        others = result.get("other_clauses") or []
        if others:
            rows = "".join(
                '<div style="display:flex;gap:16px;padding:12px 14px;border-radius:10px;background:#F4F5F1;margin-bottom:10px">'
                f'<div style="flex:1"><div style="font-size:14px;font-weight:600">{esc(o["section_path"])}</div>'
                f'<div style="font-size:14px;line-height:1.5;color:#3F4B48">{esc(o["summary"])}</div></div>'
                f'<div class="cl-mono" style="font-size:13px;font-weight:500">{float(o["score"]):.2f}</div></div>'
                for o in others
            )
            md(f'<div class="cl-card"><div class="cl-h">Other clauses considered</div>{rows}</div>')
    with right:
        md(
            '<div class="cl-card"><div class="cl-h">Explanation, checked against the evidence</div>'
            f'{aas_html(float(current["aas"]), target)}'
            f'<p style="margin:0 0 16px;font-size:16px;line-height:1.6">{esc(current["text"])}</p>'
            f'{spans_html(current["spans"])}</div>'
        )
        if key == "final" and result.get("reflection_note"):
            md(
                '<div class="cl-card" style="background:#F4F5F1"><div style="font-size:14px;font-weight:600;margin-bottom:6px">'
                f'What the check changed</div><div style="font-size:14px;line-height:1.55;color:#3F4B48">{esc(result["reflection_note"])}</div></div>'
            )
        if result.get("appeal_draft"):
            md(
                '<div class="cl-card"><div class="cl-h">Draft message</div>'
                f'<div class="cl-serif" style="padding:16px;border-radius:10px;border:1px solid #D9DFDA;background:#FBFBF9;'
                f'font-size:16px;line-height:1.65">{esc_multiline(result["appeal_draft"])}</div>'
                f'<div style="font-size:13px;color:#55615E;margin-top:10px">{DISCLAIMER}. Fill in every bracketed detail yourself before sending.</div></div>'
            )
            appeal = result.get("appeal")
            if appeal:
                li = lambda items: "".join(f"<li>{esc(i)}</li>" for i in items)  # noqa: E731
                quotes = "".join(
                    f'<li><b>{esc(c["clause_id"])}</b>: "{esc(c["quote"])}"</li>'
                    for c in appeal["cited_clauses"])
                kind = "Appeal" if appeal["kind"] == "appeal" else "Request for clarification"
                src = ("checked against the clause" if appeal["source"] == "llm"
                       else "template fallback: the model draft failed the quote check")
                md(
                    f'<div class="cl-card"><div class="cl-h">{kind}, in sections ({esc(src)})</div>'
                    f'<div style="font-size:15px;line-height:1.6"><b>Facts</b><ul>{li(appeal["facts"])}</ul>'
                    f'<b>Policy wording relied on</b><ul>{quotes or "<li>none</li>"}</ul>'
                    f'<b>Reasoning</b><p>{esc(appeal["reasoning"])}</p>'
                    f'<b>Relief sought</b><p>{esc(appeal["relief_sought"])}</p>'
                    f'<b>Documents to attach</b><ul>{li(appeal["document_checklist"])}</ul>'
                    f'<b>Timing</b><p>{esc(appeal["deadline_note"])}</p></div>'
                    f'<div style="font-size:13px;color:#55615E">{esc(appeal["disclaimer"])}</div></div>'
                )
            st.download_button("Download draft (.txt)", result["appeal_draft"],
                               file_name="claimlens_appeal.txt")
