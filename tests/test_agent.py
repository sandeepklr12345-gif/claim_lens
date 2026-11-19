"""End-to-end agent run with FakeLLM and fake tools (no torch, no Ollama): runs in CI."""

import numpy as np

from claimlens.agent.graph import run_case
from claimlens.agent.verdict import DISCLAIMER, appeal_template, decide
from claimlens.explain.aas import alignment_score, words
from claimlens.llm.client import FakeLLM
from claimlens.rag.chunk import chunk_pages
from claimlens.rag.retriever import HybridRetriever
from claimlens.schemas import Attribution, Case, MatchResult, Verdict

PAGES = [
    "4.1 Waiting Periods\nPre-existing diseases are covered only after 36 months of "
    "continuous coverage under the policy.\n"
    "4.2 Dental\nDental treatment is excluded unless it follows an accident.\n"
    "4.3 Room Rent\nRoom rent is limited to one percent of the sum insured per day.\n"
    "4.4 Maternity\nMaternity expenses are covered after a waiting period of 24 months.\n"
    "4.5 Ambulance\nRoad ambulance charges are reimbursed up to a fixed limit.\n"
]
CFG = {
    "rag": {"top_k_fused": 10, "top_k_return": 3},
    "agent": {"tau_high": 0.7, "tau_low": 0.3, "tau_aas": 0.6, "max_rounds": 2},
}


class FakeTools:
    """Overlap-based stand-ins for the matcher and IG; real BM25 retrieval."""

    def __init__(self) -> None:
        self.chunks = chunk_pages("t", PAGES, min_words=1, max_words=300)
        self.retriever = HybridRetriever(self.chunks, None)

    def search_clauses(self, query, doc_id, top_k=5):
        return self.retriever.retrieve(query, k=top_k)

    def get_clause(self, doc_id, clause_id):
        return next(c for c in self.chunks if c.chunk_id == clause_id)

    def score_pair(self, reason, clause, clause_id=""):
        overlap = len(set(words(reason)) & set(words(clause))) / max(1, len(set(words(reason))))
        logit = 6 * overlap - 3
        return MatchResult(clause_id=clause_id, score=logit, prob=float(1 / (1 + np.exp(-logit))))

    def attribute_ig(self, reason, clause, clause_id=""):
        toks = words(clause)
        r = set(words(reason))
        return Attribution(clause_id=clause_id, tokens=toks,
                           scores=[1.0 if t in r and len(t) > 2 else 0.0 for t in toks])

    def alignment_score(self, reason, clause, cited_spans, attribution=None):
        attr = attribution or self.attribute_ig(reason, clause)
        return alignment_score(attr, cited_spans, clause)

    def close(self):
        pass


def make_case(letter: str) -> Case:
    return Case(case_id="c1", policy_id="t", letter_text=letter, reason="r",
                gold_clause_ids=["t-0000"], gold_verdict=Verdict.SUPPORTED)


def test_decide_thresholds_and_abstain():
    assert decide(0.9, 0.7, 0.3) is Verdict.SUPPORTED
    assert decide(0.7, 0.7, 0.3) is Verdict.SUPPORTED
    assert decide(0.1, 0.7, 0.3) is Verdict.NOT_SUPPORTED
    assert decide(0.5, 0.7, 0.3) is Verdict.INSUFFICIENT_EVIDENCE


def test_end_to_end_with_reflection():
    llm = FakeLLM([
        {"reason": "pre-existing diseases covered only after 36 months continuous coverage"},
        {"explanation": "initial",
         "cited_spans": [{"source": "clause", "text": "continuous coverage under the policy"}]},
        {"explanation": "revised",
         "cited_spans": [
             {"source": "clause", "text": "Pre-existing diseases are covered only after 36 months"}
         ]},
    ])
    res = run_case(make_case("Dear Mr A, your claim is rejected ..."), FakeTools(), llm, CFG)
    assert res.reason.startswith("pre-existing")
    assert "Waiting Periods" in res.top_clauses[0].chunk.section_path
    assert res.verdict is Verdict.SUPPORTED
    assert res.aas_initial < res.aas_final
    assert res.explanation_initial.text == "initial" and res.explanation_final.text == "revised"
    assert [h[0] for h in res.aas_history][:2] == ["initial", "revision_1"]
    assert res.appeal_draft is None  # only drafted for NOT_SUPPORTED


def test_on_progress_reports_every_step_in_order_and_result_is_unchanged():
    def llm() -> FakeLLM:
        return FakeLLM([
            {"reason": "pre-existing diseases covered only after 36 months continuous coverage"},
            {"explanation": "initial",
             "cited_spans": [{"source": "clause", "text": "continuous coverage under the policy"}]},
            {"explanation": "revised", "cited_spans": [
                {"source": "clause",
                 "text": "Pre-existing diseases are covered only after 36 months"}]},
        ], default={"explanation": "again", "cited_spans": []})

    case = make_case("Dear Mr A, your claim is rejected ...")
    events: list[tuple[str, dict]] = []
    res = run_case(case, FakeTools(), llm(), CFG, on_progress=lambda e, d: events.append((e, d)))
    names = [e for e, _ in events]
    # R3 added the check_retrieval node after retrieve (tau_retr unset in CFG: it just passes)
    assert names[:4] == ["parse", "retrieve", "check_retrieval", "match"]
    assert names[4] == "attribution" and names[5] == "aas"
    assert names[-3:] == ["explain_reflect", "verdict", "draft"]
    assert [d["label"] for e, d in events if e == "aas"] == [h[0] for h in res.aas_history]
    assert res == run_case(case, FakeTools(), llm(), CFG)  # no callback, same result


def test_not_supported_gets_template_appeal_with_disclaimer():
    llm = FakeLLM(default={"reason": "dental claim for cosmetic tourism", "explanation": "e",
                           "cited_spans": []})
    res = run_case(make_case("letter"), FakeTools(), llm, CFG)
    assert res.verdict is Verdict.NOT_SUPPORTED
    assert res.appeal_draft and DISCLAIMER in res.appeal_draft


def test_nothing_retrieved_abstains():
    llm = FakeLLM(default={"reason": "zzz qqq"})
    res = run_case(make_case("letter"), FakeTools(), llm, CFG)
    assert res.top_clauses == [] and res.verdict is Verdict.INSUFFICIENT_EVIDENCE
    assert res.match is None and res.explanation_final is None and res.appeal_draft is None


def test_parse_failure_falls_back_to_letter_text():
    class Boom(FakeLLM):
        def generate_json(self, prompt, system=None):
            if "Claim rejection letter" in prompt:
                raise ValueError("no json")
            return super().generate_json(prompt, system)

    llm = Boom(default={"explanation": "e", "cited_spans": []})
    res = run_case(make_case("Dental treatment claim rejected"), FakeTools(), llm, CFG)
    assert res.reason == "Dental treatment claim rejected"


def test_appeal_template_quotes_only_clause_spans():
    from claimlens.schemas import CitedSpan, Explanation

    expl = Explanation(text="Because.", cited_spans=[CitedSpan(text="after 36 months"),
                                                      CitedSpan(text="rr", source="reason")])
    text = appeal_template("reason", "c-1", "clause", expl)
    assert '"after 36 months"' in text and '"rr"' not in text
