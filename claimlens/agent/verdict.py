"""Verdict rule with an abstain option (SPEC.md 7.6) and the template appeal draft."""

from __future__ import annotations

from claimlens.schemas import Explanation, Verdict

DISCLAIMER = "Decision support, not legal advice."


def decide(prob: float, tau_high: float, tau_low: float) -> Verdict:
    """SUPPORTED if prob >= tau_high; NOT_SUPPORTED if prob < tau_low; otherwise abstain."""
    if prob >= tau_high:
        return Verdict.SUPPORTED
    if prob < tau_low:
        return Verdict.NOT_SUPPORTED
    return Verdict.INSUFFICIENT_EVIDENCE


def stance_for(verdict: Verdict) -> str:
    return {
        Verdict.SUPPORTED: "supports the rejection",
        Verdict.NOT_SUPPORTED: "does not support the rejection",
        Verdict.INSUFFICIENT_EVIDENCE: "only partly or unclearly supports the rejection",
    }[verdict]


def appeal_template(reason: str, clause_id: str, clause_text: str, expl: Explanation) -> str:
    """TEMPLATE appeal draft (no LLM). Only produced for NOT_SUPPORTED verdicts."""
    quoted = "\n".join(f'  - "{s.text}"' for s in expl.cited_spans if s.source == "clause")
    return (
        "To,\nThe Grievance Officer / Claims Review Committee\n\n"
        "Subject: Appeal against rejection of my health insurance claim\n\n"
        f"I received a rejection of my claim on the following ground: {reason}\n\n"
        f"I have reviewed the policy wording (clause {clause_id}). In my understanding, the "
        f"cited ground is not supported by it. {expl.text}\n\n"
        + (f"Relevant policy wording:\n{quoted}\n\n" if quoted else "")
        + "I request that you reconsider the claim and settle it, or give a clause-wise written "
        "explanation for the rejection.\n\nYours faithfully,\n"
        "[Policyholder name and policy number]\n\n"
        f"---\n{DISCLAIMER} This is a template draft; check it against the full policy and "
        "applicable rules before sending."
    )
