"""Preview the audit-result screen with SAMPLE data.

Run:  streamlit run app\\preview_result.py

Everything below is a hand-made placeholder (source: synthetic). The weights are
illustrative, not model output, and the AAS values are computed from those weights.
Do not quote any number from this file in the README.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # `streamlit run` adds app/ only

import streamlit as st  # noqa: E402

from app.result_view import cited_indices, render_result  # noqa: E402

CLAUSE = [
    ("Expenses", 0.05), ("related", 0.04), ("to", 0.01), ("the", 0.01), ("treatment", 0.10),
    ("of", 0.01), ("a", 0.02), ("pre-existing", 0.55), ("disease", 0.48), ("and", 0.01),
    ("its", 0.03), ("direct", 0.06), ("complications", 0.12), ("shall", 0.04), ("be", 0.02),
    ("excluded", 0.38), ("until", 0.22), ("the", 0.02), ("expiry", 0.18), ("of", 0.01),
    ("thirty-six", 0.92), ("months", 1.0), ("of", 0.03), ("continuous", 0.30), ("coverage", 0.20),
    ("from", 0.02), ("the", 0.01), ("date", 0.08), ("of", 0.01), ("inception.", 0.25),
]
REASON = [
    ("Claim", 0.02), ("for", 0.01), ("knee", 0.05), ("replacement", 0.06), ("rejected:", 0.05),
    ("osteoarthritis", 0.20), ("was", 0.01), ("declared", 0.15), ("at", 0.02), ("inception", 0.12),
    ("and", 0.01), ("the", 0.01), ("36-month", 0.85), ("waiting", 0.60), ("period", 0.50),
    ("for", 0.02), ("pre-existing", 0.40), ("disease", 0.30), ("is", 0.02), ("not", 0.30),
    ("complete.", 0.25),
]


def _words(pairs):
    return [{"text": t, "score": s} for t, s in pairs]


def _aas(words, spans):
    """Positive attribution mass inside cited spans / total positive mass (SPEC 7.5)."""
    hit, _ = cited_indices(words, spans, "clause")
    total = sum(max(w["score"], 0.0) for w in words)
    return sum(max(words[i]["score"], 0.0) for i in hit) / total if total else 0.0


_clause = _words(CLAUSE)
_initial = [{"source": "clause", "text": t} for t in
            ["treatment of a pre-existing disease", "direct complications", "excluded", "inception."]]
_final = [{"source": "clause", "text": t} for t in
          ["pre-existing disease", "excluded until", "thirty-six months"]]

SAMPLE = {
    "sample": True,
    "case_id": "case_001",
    "policy_id": "[POLICY A]",
    "verdict": "SUPPORTED",
    "support_score": 0.87,
    "reason": _words(REASON),
    "clause": {"section_path": "Section 4.2 · Exclusions › Pre-existing diseases", "page": 12, "words": _clause},
    "other_clauses": [
        {"section_path": "Section 4.1 · Waiting periods › Specified diseases",
         "summary": "Separate waiting period for listed conditions.", "score": 0.41},
        {"section_path": "Section 2.3 · Definitions › Continuous coverage",
         "summary": "How continuous coverage is counted across renewals.", "score": 0.22},
    ],
    "explanations": {
        "initial": {
            "text": "The claim was rejected because the knee surgery relates to a pre-existing disease. "
                    "The policy excludes treatment and direct complications of such conditions, so the "
                    "insurer's reason is consistent with the clause.",
            "spans": _initial, "aas": _aas(_clause, _initial),
        },
        "final": {
            "text": "The rejection is supported. The clause excludes a pre-existing disease until "
                    "thirty-six months of continuous coverage have passed, and the letter says that "
                    "period is not complete.",
            "spans": _final, "aas": _aas(_clause, _final),
        },
    },
    "reflection_note": "The first draft leaned on words with little influence on the match and skipped "
                       "the strongest ones. One revision round pointed this out. The verdict stayed the same.",
    "appeal_draft": "Subject: Request for clarification on claim rejection\n\n"
                    "To the Grievance Officer, [INSURER NAME],\n\n"
                    "I am writing about the rejection of claim [CLAIM NO.] under policy [POLICY NO.]. "
                    "Please confirm the date my continuous coverage began, the date the condition was "
                    "first disclosed, and how the waiting period was counted, with the clause reference.\n\n"
                    "Yours sincerely,\n[YOUR NAME]",
}

st.set_page_config(page_title="ClaimLens", layout="wide")
render_result(SAMPLE)
