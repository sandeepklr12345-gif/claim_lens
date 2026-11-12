"""Fixed appeal wording used by the drafter's fallback and by the appeal MCP server.

Everything here is static text with visible [placeholders]; nothing is filled from the case.
"""

from __future__ import annotations

from claimlens.schemas import DISCLAIMER

DOCUMENT_CHECKLIST = [
    "Copy of the claim rejection letter",
    "Policy wording and policy schedule",
    "Claim form and discharge summary",
    "Medical bills, prescriptions and investigation reports",
    "Earlier correspondence with the insurer, including any grievance acknowledgement",
    "Identity proof of the policyholder",
]

DEADLINE_NOTE = (
    "A complaint to the Insurance Ombudsman needs a written representation to the insurer first; "
    "it must then be made within one year of the insurer's rejection, or of one month after the "
    "representation was sent (Insurance Ombudsman Rules, 2017, Rule 14(3), checked against the "
    "published text as amended till 09.11.2023; later amendments not checked). Confirm with the "
    "Ombudsman office before relying on a date, and compute dates with the deadline tool."
)

APPEAL_RELIEF = (
    "I request that you reconsider the claim and settle it, or give a clause-wise written "
    "explanation of the ground for rejection."
)
CLARIFICATION_RELIEF = (
    "I request a written, clause-wise explanation of the ground for rejection and the policy "
    "wording relied on, so that I can respond."
)

TEMPLATES: dict[str, dict[str, str]] = {
    "appeal_letter": {
        "description": "Appeal against a rejection that the policy wording does not support.",
        "text": (
            "To,\nThe Grievance Officer / Claims Review Committee\n[insurer name]\n\n"
            "Subject: Appeal against rejection of claim [claim number], policy [policy number]\n\n"
            "Dear Sir/Madam,\n\nMy claim was rejected on [date of rejection] on the ground that "
            "[ground stated in the rejection letter].\n\nPolicy wording relied on: "
            "[clause id and quotation].\n\n[Reasoning]\n\n" + APPEAL_RELIEF + "\n\n"
            "Yours faithfully,\n[policyholder name]\n\n---\n" + DISCLAIMER),
    },
    "clarification_request": {
        "description": "Request for clarification when the wording does not clearly support "
                       "the rejection.",
        "text": (
            "To,\nThe Grievance Officer / Claims Review Committee\n[insurer name]\n\n"
            "Subject: Request for clarification on rejection of claim [claim number], "
            "policy [policy number]\n\nDear Sir/Madam,\n\nMy claim was rejected on "
            "[date of rejection] on the ground that [ground stated in the rejection letter]. "
            "The policy wording [clause id and quotation] does not clearly establish this "
            "ground.\n\n" + CLARIFICATION_RELIEF + "\n\nYours faithfully,\n"
            "[policyholder name]\n\n---\n" + DISCLAIMER),
    },
    "document_checklist": {
        "description": "Documents to attach to an appeal or complaint.",
        "text": "\n".join(f"- {d}" for d in DOCUMENT_CHECKLIST),
    },
}
