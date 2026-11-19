"""R2 appeal drafter: grounding checks, retry and template fallback (mocked LLM), deadline
arithmetic, tool outputs and the API endpoint. No model, Ollama or GPU needed."""

import pytest

from claimlens.appeal import tools
from claimlens.appeal.deadlines import add_months, compute_deadlines
from claimlens.appeal.draft import (
    appeal_text,
    clean_details,
    draft_appeal,
    find_placeholders,
    invented_numbers,
    quote_is_verbatim,
    template_draft,
)
from claimlens.schemas import (
    AppealDraft,
    CaseResult,
    Chunk,
    CitedSpan,
    Explanation,
    RetrievedClause,
    Verdict,
)

CLAUSE = ("Expenses related to the treatment of a pre-existing Disease shall be excluded until\n"
          "the expiry of 48 months of continuous coverage after the date of inception.")
LETTER = "Dear Mr A, claim CL-1234 is rejected: pre-existing disease within 36 months. 12/03/2026."
REASON = "Pre-existing disease claimed within 36 months of policy start."
GOOD = {
    "details": {"policyholder_name": None, "policy_number": None, "claim_number": "CL-1234",
                "rejection_date": "invented 1 Jan"},
    "facts": ["The claim was rejected as a pre-existing disease within 36 months."],
    "quotes": ["the expiry of 48 months   of continuous coverage"],  # extra spaces: still verbatim
    "reasoning": "The clause sets a 48 month wait, which the insurer misapplied.",
    "relief": "Please reconsider the claim.",
}


class SeqLLM:
    """Returns queued answers (a dict, or an Exception to raise); records the prompts."""

    def __init__(self, *answers):
        self.answers, self.prompts = list(answers), []

    def generate_json(self, prompt, system=None):
        self.prompts.append(prompt)
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a


def result(verdict=Verdict.NOT_SUPPORTED, with_expl=True) -> CaseResult:
    chunk = Chunk(chunk_id="pol-0001", doc_id="pol", text=CLAUSE, section_path="Excl01", page=1)
    expl = Explanation(text="The clause is about a 48 month wait.", cited_spans=[
        CitedSpan(text="expiry of 48 months of continuous coverage", source="clause"),
        CitedSpan(text="not in the clause at all", source="clause"),
        CitedSpan(text="pre-existing", source="reason")]) if with_expl else None
    top = [RetrievedClause(chunk=chunk, score=1)]
    return CaseResult(case_id="c", reason=REASON, top_clauses=top, explanation_final=expl,
                      verdict=verdict)


def test_quote_check_is_verbatim_after_whitespace_normalisation():
    assert quote_is_verbatim("until the expiry of 48 months", CLAUSE)  # crosses the newline
    assert not quote_is_verbatim("until the expiry of 36 months", CLAUSE)
    assert not quote_is_verbatim("expiry of", CLAUSE)  # too short to count
    assert not quote_is_verbatim("Until The Expiry Of 48 Months", CLAUSE)  # case matters


def test_invented_numbers_are_caught():
    assert invented_numbers("48 months and 36 months", [CLAUSE, REASON]) == []
    assert invented_numbers("Rs 5,00,000 within 90 days", [CLAUSE, REASON]) == ["5,00,000", "90"]


def test_details_only_from_the_letter():
    d = clean_details(GOOD["details"], LETTER)
    assert d["claim_number"] == "CL-1234"
    assert d["rejection_date"] == "[date of rejection]"  # not in the letter -> placeholder
    assert d["policy_number"] == "[policy number]"
    assert clean_details("garbage", LETTER)["policyholder_name"] == "[policyholder name]"


def test_first_try_pass_is_an_llm_draft():
    llm = SeqLLM(GOOD)
    d = draft_appeal(result(), LETTER, llm)
    assert d.source == "llm" and d.attempts == 1 and d.kind == "appeal"
    assert d.cited_clauses[0].clause_id == "pol-0001"
    assert d.cited_clauses[0].quote == "the expiry of 48 months of continuous coverage"
    assert "[policy number]" in d.placeholders
    assert "not legal advice" in d.disclaimer and d.deadline_verified
    assert len(llm.prompts) == 1


def test_bad_quote_retries_once_then_passes():
    bad = {**GOOD, "quotes": ["the expiry of 99 months of continuous coverage"]}
    llm = SeqLLM(bad, GOOD)
    d = draft_appeal(result(), LETTER, llm)
    assert d.source == "llm" and d.attempts == 2
    assert "not word-for-word" in llm.prompts[1]  # the retry names the failure


def test_two_failures_fall_back_to_the_template():
    bad = {**GOOD, "quotes": ["a sentence the model made up entirely"]}
    llm = SeqLLM(bad, bad)
    d = draft_appeal(result(), LETTER, llm)
    assert d.source == "template" and d.attempts == 2 and d.fallback_reason
    assert len(llm.prompts) == 2  # at most one retry
    # template quotes are only explanation spans that pass the verbatim check
    assert [c.quote for c in d.cited_clauses] == ["expiry of 48 months of continuous coverage"]
    assert "[date of rejection]" in d.placeholders


def test_invented_number_in_prose_fails_the_attempt():
    invented = {**GOOD, "reasoning": "The insurer owes Rs 7,50,000 under the policy."}
    d = draft_appeal(result(), LETTER, SeqLLM(invented, GOOD))
    assert d.attempts == 2 and d.source == "llm"
    assert "7,50,000" not in " ".join(d.facts + [d.reasoning, d.relief_sought])


def test_llm_errors_count_as_failed_attempts():
    d = draft_appeal(result(), LETTER, SeqLLM(RuntimeError("down"), ValueError("no json")))
    assert d.source == "template" and "LLM call failed" in d.fallback_reason


def test_verdict_decides_the_kind():
    assert draft_appeal(result(Verdict.SUPPORTED), LETTER, SeqLLM()) is None
    clar = draft_appeal(result(Verdict.INSUFFICIENT_EVIDENCE), LETTER, SeqLLM(GOOD))
    assert clar.kind == "clarification"
    empty = result()
    empty.top_clauses = []
    assert draft_appeal(empty, LETTER, SeqLLM()) is None


def test_template_without_explanation_still_complete_but_has_no_quotes():
    d = template_draft("appeal", REASON, "pol-0001", CLAUSE, None, 2, "why")
    assert d.facts and d.reasoning and d.relief_sought and d.document_checklist
    assert d.cited_clauses == [] and d.source == "template"


def test_text_rendering_and_schema_roundtrip():
    d = draft_appeal(result(), LETTER, SeqLLM(GOOD))
    text = appeal_text(d)
    assert "Decision support, not legal advice" in text
    assert "Clause pol-0001" in text and "Confirm with the Ombudsman office" in text
    assert AppealDraft.model_validate_json(d.model_dump_json()) == d
    assert find_placeholders("see [policy number] and [x]") == ["[policy number]"]


# ---- deadlines ----

def test_add_months_clamps_month_ends():
    from datetime import date

    assert add_months(date(2026, 1, 31), 1) == date(2026, 2, 28)
    assert add_months(date(2028, 1, 31), 1) == date(2028, 2, 29)  # leap year
    assert add_months(date(2026, 11, 30), 3) == date(2027, 2, 28)
    assert add_months(date(2026, 12, 15), 1) == date(2027, 1, 15)  # year rollover
    assert add_months(date(2024, 2, 29), 12) == date(2025, 2, 28)  # 29 Feb + 1 year
    assert add_months(date(2024, 2, 29), 48) == date(2028, 2, 29)


def test_deadlines_from_grievance_and_reply():
    g = compute_deadlines("2026-01-10", grievance_filed_date="2026-01-31")
    assert g.ombudsman_earliest_date == "2026-02-28" and g.ombudsman_latest_date == "2027-02-28"
    r = compute_deadlines("2026-01-10", "2026-01-31", insurer_reply_date="2026-03-05")
    assert r.ombudsman_earliest_date == "2026-03-05" and r.ombudsman_latest_date == "2027-03-05"
    leap = compute_deadlines("2024-02-29", insurer_reply_date="2024-02-29")
    assert leap.ombudsman_latest_date == "2025-02-28"


def test_deadlines_without_a_representation_are_conservative_and_unverified():
    d = compute_deadlines("2026-03-31")
    assert d.conservative and d.ombudsman_earliest_date is None
    assert d.ombudsman_latest_date == "2027-03-31"
    # rules checked against the official text as amended till 09.11.2023, later ones not
    assert d.verified is True and "09.11.2023" in d.source
    assert "later amendments were not checked" in d.verify_note
    assert all(r.startswith("Rule 14(3)") for r in d.rules)


def test_deadline_input_errors():
    with pytest.raises(ValueError):
        compute_deadlines("31/03/2026")
    with pytest.raises(ValueError):
        compute_deadlines("2026-03-31", grievance_filed_date="2026-03-01")
    with pytest.raises(ValueError):
        compute_deadlines("2026-03-01", "2026-03-10", insurer_reply_date="2026-03-05")


# ---- the three MCP tools (plain functions) ----

def test_tool_outputs():
    names = [t["name"] for t in tools.list_templates()["templates"]]
    assert {"appeal_letter", "clarification_request", "document_checklist"} <= set(names)
    t = tools.get_template("appeal_letter")
    assert "[policyholder name]" in t["placeholders"] and "not legal advice" in t["text"]
    with pytest.raises(ValueError):
        tools.get_template("nope")
    out = tools.compute_deadlines("2026-03-31", "2026-04-30")
    assert out["ombudsman_earliest_date"] == "2026-05-30" and out["verified"] is True


# ---- API ----

def test_appeal_endpoint(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from api import main

    monkeypatch.setenv("CLAIMLENS_STATE_DIR", str(tmp_path))
    full = result()
    full.appeal = draft_appeal(full, LETTER, SeqLLM(GOOD))
    runners = {"with": lambda case: full,
               "none": lambda case: result(Verdict.SUPPORTED)}
    main.app.dependency_overrides[main.get_policies] = lambda: ["pol"]
    try:
        c = TestClient(main.app)
        for name, expected in (("with", 200), ("none", 404)):
            main.app.dependency_overrides[main.get_runner] = lambda n=name: runners[n]
            body = {"letter_text": LETTER, "policy_id": "pol"}
            cid = c.post("/cases", json=body).json()["case_id"]
            assert c.get(f"/cases/{cid}/appeal").status_code == 409  # not run yet
            c.post(f"/cases/{cid}/run")
            r = c.get(f"/cases/{cid}/appeal")
            assert r.status_code == expected
            if expected == 200:
                assert r.json()["kind"] == "appeal" and "not legal advice" in r.json()["disclaimer"]
        assert c.get("/cases/missing/appeal").status_code == 404
    finally:
        main.app.dependency_overrides.clear()
