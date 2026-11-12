"""Grounded appeal drafter (SPEC.md 14.4).

`draft_appeal` asks the LLM for a structured draft, then checks it mechanically:
  * every quote must appear verbatim (after whitespace normalisation) in the retrieved clause;
  * no number, date or amount may appear in the facts, reasoning or relief unless it is also in
    the letter, the stated reason or the clause (no invented figures);
  * letter details (name, policy/claim number, rejection date) are kept only if they appear in
    the letter; otherwise they become visible [placeholders].
One retry with the failures named; if that also fails the draft is replaced by the TEMPLATE
fallback (`source: "template"`, no LLM text except the already-validated explanation).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from claimlens.appeal.templates import (
    APPEAL_RELIEF,
    CLARIFICATION_RELIEF,
    DEADLINE_NOTE,
    DOCUMENT_CHECKLIST,
)
from claimlens.llm.client import LLM
from claimlens.schemas import AppealDraft, CaseResult, CitedClause, Explanation, Verdict

log = logging.getLogger(__name__)

MIN_QUOTE_WORDS = 3
MAX_QUOTES = 3
DETAIL_KEYS = {
    "policyholder_name": "[policyholder name]",
    "policy_number": "[policy number]",
    "claim_number": "[claim number]",
    "rejection_date": "[date of rejection]",
}
SYSTEM = (
    "You draft claim-appeal letters for a policyholder in India. Reply with JSON only. Use only "
    "facts that are in the letter. Never invent policy numbers, claim numbers, dates, amounts or "
    "names."
)
PROMPT = """{task}

REJECTION LETTER (the only source of facts):
\"\"\"{letter}\"\"\"

REJECTION GROUND: {reason}

POLICY CLAUSE {clause_id} (the only source of policy wording):
\"\"\"{clause}\"\"\"

AUDIT FINDING: {finding}

Reply as JSON:
{{"details": {{"policyholder_name": null, "policy_number": null, "claim_number": null,
"rejection_date": null}},
"facts": ["2-4 short factual sentences taken from the letter"],
"quotes": ["1-{max_quotes} passages copied EXACTLY, word for word, from the clause above"],
"reasoning": "2-4 sentences linking the quoted wording to the ground",
"relief": "one sentence on what you ask the insurer to do"}}
In "details" give the exact text as it appears in the letter, or null if the letter does not
have it. Do not paraphrase quotes. Do not add numbers or dates that are not in the letter or
the clause.{feedback}"""
TASKS = {
    "appeal": "Write an appeal against the rejection. The audit found the clause does not "
              "support the ground given.",
    "clarification": "Write a request for clarification. The audit found the clause does not "
                     "clearly establish the ground given, so ask the insurer to explain.",
}
NUM_RE = re.compile(r"\d[\d,./-]*\d|\d")


def normalise_ws(text: str) -> str:
    return " ".join(text.split())


def quote_is_verbatim(quote: str, clause_text: str) -> bool:
    """A quote counts only if it is long enough and a verbatim substring of the clause."""
    q = normalise_ws(quote)
    return len(q.split()) >= MIN_QUOTE_WORDS and q in normalise_ws(clause_text)


def _numbers(text: str) -> set[str]:
    return {m.strip(".,/-") for m in NUM_RE.findall(text)}


def invented_numbers(text: str, allowed_sources: list[str]) -> list[str]:
    """Numbers in `text` that appear in none of the allowed source texts."""
    allowed: set[str] = set()
    for s in allowed_sources:
        allowed |= _numbers(s)
    return sorted(_numbers(text) - allowed)


def find_placeholders(text: str) -> list[str]:
    return sorted(set(re.findall(r"\[[^\]]{2,40}\]", text)))


def clean_details(raw: Any, letter: str) -> dict[str, str]:
    """Keep a letter detail only if it appears in the letter; otherwise a visible placeholder."""
    raw = raw if isinstance(raw, dict) else {}
    out = {}
    for key, placeholder in DETAIL_KEYS.items():
        value = raw.get(key)
        ok = isinstance(value, str) and value.strip() and (
            normalise_ws(value) in normalise_ws(letter))
        out[key] = normalise_ws(value) if ok else placeholder
    return out


def validate_llm_draft(raw: dict[str, Any], letter: str, reason: str, clause_text: str
                       ) -> tuple[list[str], list[str]]:
    """Return (problems, accepted_quotes). Empty problems means the draft is grounded."""
    problems: list[str] = []
    quotes = [q for q in raw.get("quotes", []) if isinstance(q, str)][:MAX_QUOTES]
    bad = [q for q in quotes if not quote_is_verbatim(q, clause_text)]
    if not quotes:
        problems.append("no quotes were given")
    if bad:
        problems.append("these quotes are not word-for-word in the clause: "
                        + " | ".join(f'"{q}"' for q in bad))
    facts = [f for f in raw.get("facts", []) if isinstance(f, str) and f.strip()]
    if not facts:
        problems.append("no facts were given")
    if not str(raw.get("reasoning", "")).strip() or not str(raw.get("relief", "")).strip():
        problems.append("reasoning or relief is empty")
    prose = " ".join(facts + [str(raw.get("reasoning", "")), str(raw.get("relief", ""))])
    extra = invented_numbers(prose, [letter, reason, clause_text])
    if extra:
        problems.append("these numbers are not in the letter or the clause: " + ", ".join(extra))
    return problems, [q for q in quotes if q not in bad]


def _assemble(kind: str, source: str, attempts: int, clause_id: str, details: dict[str, str],
              facts: list[str], quotes: list[str], reasoning: str, relief: str,
              fallback_reason: str | None = None) -> AppealDraft:
    prose = " ".join(facts + [reasoning, relief])
    placeholders = sorted(set(find_placeholders(prose)) | {v for v in details.values()
                                                            if v.startswith("[")})
    return AppealDraft(
        kind=kind, source=source, attempts=attempts, fallback_reason=fallback_reason,
        details=details, facts=facts,
        cited_clauses=[CitedClause(clause_id=clause_id, quote=normalise_ws(q)) for q in quotes],
        reasoning=reasoning, relief_sought=relief, document_checklist=list(DOCUMENT_CHECKLIST),
        deadline_note=DEADLINE_NOTE, deadline_verified=True, placeholders=placeholders)


def template_draft(kind: str, reason: str, clause_id: str, clause_text: str,
                   expl: Explanation | None, attempts: int, why: str) -> AppealDraft:
    """FALLBACK (marked): static wording plus the audit's own explanation. Quotes are only the
    explanation's clause spans that pass the verbatim check; there may be none."""
    quotes = []
    for s in (expl.cited_spans if expl else []):
        if s.source == "clause" and quote_is_verbatim(s.text, clause_text) and s.text not in quotes:
            quotes.append(s.text)
    facts = [f"The insurer rejected the claim on this ground: {reason}",
             "Date of rejection: [date of rejection]; claim number: [claim number]; "
             "policy number: [policy number]."]
    if kind == "appeal":
        reasoning = (f"In my understanding the cited ground is not supported by clause {clause_id}."
                     + (f" {expl.text}" if expl else ""))
        relief = APPEAL_RELIEF
    else:
        reasoning = (f"Clause {clause_id} does not clearly establish the cited ground."
                     + (f" {expl.text}" if expl else ""))
        relief = CLARIFICATION_RELIEF
    details = dict(DETAIL_KEYS)
    return _assemble(kind, "template", attempts, clause_id, details, facts, quotes[:MAX_QUOTES],
                     reasoning, relief, fallback_reason=why)


def draft_appeal(result: CaseResult, letter_text: str, llm: LLM) -> AppealDraft | None:
    """Structured appeal for NOT_SUPPORTED, clarification request for INSUFFICIENT_EVIDENCE,
    None for SUPPORTED (or when no clause was matched)."""
    if result.verdict == Verdict.SUPPORTED or not result.top_clauses:
        return None
    kind = "appeal" if result.verdict == Verdict.NOT_SUPPORTED else "clarification"
    clause = result.top_clauses[0].chunk
    expl = result.explanation_final
    finding = expl.text if expl else "The clause does not clearly address the ground."
    feedback = ""
    problems: list[str] = []
    for attempt in (1, 2):
        prompt = PROMPT.format(task=TASKS[kind], letter=letter_text, reason=result.reason,
                               clause_id=clause.chunk_id, clause=clause.text, finding=finding,
                               max_quotes=MAX_QUOTES, feedback=feedback)
        try:
            raw = llm.generate_json(prompt, SYSTEM)
        except Exception as e:  # noqa: BLE001 - a failed call is a failed attempt
            problems = [f"the LLM call failed: {e}"]
        else:
            problems, quotes = validate_llm_draft(raw, letter_text, result.reason, clause.text)
            if not problems:
                facts = [normalise_ws(f) for f in raw["facts"] if isinstance(f, str) and f.strip()]
                return _assemble(kind, "llm", attempt, clause.chunk_id,
                                 clean_details(raw.get("details"), letter_text), facts, quotes,
                                 normalise_ws(str(raw["reasoning"])),
                                 normalise_ws(str(raw["relief"])))
        log.warning("appeal draft attempt %d rejected: %s", attempt, "; ".join(problems))
        feedback = "\n\nYour previous reply was rejected: " + "; ".join(problems) + ". Fix this."
    return template_draft(kind, result.reason, clause.chunk_id, clause.text, expl, 2,
                          "; ".join(problems))


def appeal_text(d: AppealDraft) -> str:
    """Plain-text letter for download and for CaseResult.appeal_draft."""
    det = d.details or {k: v for k, v in DETAIL_KEYS.items()}
    heading = ("Appeal against rejection of my health insurance claim" if d.kind == "appeal"
               else "Request for clarification on the rejection of my health insurance claim")
    lines = ["To,", "The Grievance Officer / Claims Review Committee", "",
             f"Subject: {heading}",
             f"Policy: {det.get('policy_number', '[policy number]')}   "
             f"Claim: {det.get('claim_number', '[claim number]')}   "
             f"Date of rejection: {det.get('rejection_date', '[date of rejection]')}", "",
             "Facts:"] + [f"  - {f}" for f in d.facts]
    if d.cited_clauses:
        lines += ["", "Policy wording relied on:"] + [
            f'  - Clause {c.clause_id}: "{c.quote}"' for c in d.cited_clauses]
    lines += ["", "Reasoning:", f"  {d.reasoning}", "", "Relief sought:", f"  {d.relief_sought}",
              "", "Documents to attach:"] + [f"  - {x}" for x in d.document_checklist]
    lines += ["", "Yours faithfully,", det.get("policyholder_name", "[policyholder name]"), "",
              f"Note on timing: {d.deadline_note}", "",
              f"---\n{d.disclaimer} Source: {d.source}"
              + (" (template fallback; check it against the full policy)" if d.source == "template"
                 else " (LLM draft, quotes checked against the clause)") + "."]
    return "\n".join(lines)
