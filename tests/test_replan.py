"""R3 re-plan: decision rule, query handling, RRF, and the graph's conditional edge with a mocked
retriever and LLM (no torch, no Ollama)."""

import pytest

from claimlens.agent.graph import run_case
from claimlens.agent.replan import (
    ABSTAIN_REASON,
    parse_queries,
    replan_retrieve,
    retrieval_decision,
    rrf_merge,
    top_prob,
)
from claimlens.schemas import Case, Chunk, MatchResult, RetrievedClause, Verdict

CFG = {"rag": {"top_k_fused": 5, "top_k_return": 3},
       "agent": {"tau_high": 0.7, "tau_low": 0.3, "tau_aas": 0.6, "max_rounds": 2,
                 "tau_retr": 0.5}}


def chunk(i: int) -> Chunk:
    return Chunk(chunk_id=f"t-{i:04d}", doc_id="t", text=f"waiting period clause number {i}",
                 section_path=f"s{i}", page=1)


def hit(i: int) -> RetrievedClause:
    return RetrievedClause(chunk=chunk(i), score=0.0)


class ScriptedTools:
    """search_clauses returns a clause list per query; score_pair returns a logit by clause id."""

    def __init__(self, by_query: dict[str, list[int]], logit_by_clause: dict[int, float]):
        self.by_query, self.logits = by_query, logit_by_clause
        self.queries: list[str] = []

    def search_clauses(self, query, doc_id, top_k=5):
        self.queries.append(query)
        return [hit(i) for i in self.by_query.get(query, [0])][:top_k]

    def score_pair(self, reason, clause, clause_id=""):
        logit = self.logits[int(clause_id.split("-")[1])]
        return MatchResult(clause_id=clause_id, score=logit, prob=1 / (1 + 2.718281828 ** -logit))

    def attribute_ig(self, reason, clause, clause_id=""):
        from claimlens.schemas import Attribution

        toks = clause.split()
        return Attribution(clause_id=clause_id, tokens=toks, scores=[1.0] * len(toks))

    def alignment_score(self, reason, clause, cited_spans, attribution=None):
        from claimlens.explain.aas import alignment_score

        return alignment_score(attribution, cited_spans, clause)


class QueryLLM:
    """Answers the parse prompt, the re-plan prompt and anything else with an empty explanation."""

    def __init__(self, queries=("alt one", "alt two"), fail_replan=False):
        self.queries, self.fail_replan, self.replan_calls = list(queries), fail_replan, 0

    def generate_json(self, prompt, system=None):
        if "alternative search queries" in prompt:
            self.replan_calls += 1
            if self.fail_replan:
                raise RuntimeError("down")
            return {"queries": self.queries}
        if "Claim rejection letter" in prompt:
            return {"reason": "the reason"}
        return {"explanation": "e", "cited_spans": [], "reason": "the reason", "quotes": [],
                "facts": [], "reasoning": "", "relief": ""}


def case() -> Case:
    return Case(case_id="c", policy_id="t", letter_text="letter", reason="r",
                gold_clause_ids=["t-0000"], gold_verdict=Verdict.SUPPORTED)


def test_decision_rule():
    assert retrieval_decision(0.9, 0.5, False) == "ok"
    assert retrieval_decision(0.5, 0.5, False) == "ok"  # at the threshold passes
    assert retrieval_decision(0.2, 0.5, False) == "replan"
    assert retrieval_decision(0.2, 0.5, True) == "abstain"  # a second miss abstains
    assert retrieval_decision(None, 0.5, False) == "replan"  # nothing retrieved
    assert retrieval_decision(0.1, None, False) == "ok"  # check disabled


def test_top_prob_is_sigmoid_of_best_logit():
    c = [hit(0).model_copy(update={"score": -2.0}), hit(1).model_copy(update={"score": 0.0})]
    assert top_prob(c) == pytest.approx(0.5)
    assert top_prob([]) is None


def test_parse_queries_dedupes_and_caps():
    assert parse_queries({"queries": ["a b", "a b", "c", "d"]}, "r") == ["a b", "c"]
    assert parse_queries({"queries": ["r", " "]}, "r") == []  # the reason itself is not new
    assert parse_queries("junk", "r") == [] and parse_queries({"queries": "x"}, "r") == []


def test_rrf_prefers_clauses_found_by_several_queries():
    merged = rrf_merge([[hit(1), hit(2)], [hit(3), hit(2)], [hit(2)]])
    assert [m.chunk.chunk_id for m in merged][0] == "t-0002"
    assert len({m.chunk.chunk_id for m in merged}) == 3


def test_replan_retrieve_merges_and_reranks_against_the_original_reason():
    tools = ScriptedTools({"the reason": [1], "alt one": [2], "alt two": [3]},
                          {1: -3.0, 2: 3.0, 3: 0.0})
    clauses, queries = replan_retrieve(tools, QueryLLM(), "the reason", "t", 5, 3)
    assert queries == ["alt one", "alt two"]
    assert [c.chunk.chunk_id for c in clauses] == ["t-0002", "t-0003", "t-0001"]  # by logit
    assert tools.queries == ["the reason", "alt one", "alt two"]


def test_replan_survives_an_llm_failure():
    tools = ScriptedTools({"the reason": [1]}, {1: -3.0})
    clauses, queries = replan_retrieve(tools, QueryLLM(fail_replan=True), "the reason", "t", 5, 3)
    assert queries == [] and [c.chunk.chunk_id for c in clauses] == ["t-0001"]


def test_low_confidence_replans_once_then_abstains():
    tools = ScriptedTools({}, {0: -4.0})  # every search returns clause 0, always weak
    llm = QueryLLM()
    res = run_case(case(), tools, llm, CFG)
    assert llm.replan_calls == 1  # at most one re-plan
    assert res.verdict is Verdict.INSUFFICIENT_EVIDENCE
    assert res.abstain_reason == ABSTAIN_REASON
    assert res.match is None and res.explanation_final is None
    assert res.appeal_draft is None and res.appeal is None
    steps = [(e["step"], e.get("decision")) for e in res.plan_trace]
    assert steps == [("check_retrieval", "replan"), ("replan", None),
                     ("check_retrieval", "abstain")]


def test_replan_that_finds_a_good_clause_continues_to_the_match():
    tools = ScriptedTools({"the reason": [0], "alt one": [1], "alt two": [0]}, {0: -4.0, 1: 4.0})
    llm = QueryLLM()
    res = run_case(case(), tools, llm, CFG)
    assert llm.replan_calls == 1 and res.abstain_reason is None
    assert res.top_clauses[0].chunk.chunk_id == "t-0001"
    assert res.match is not None and res.match.prob > 0.9
    assert [e.get("decision") for e in res.plan_trace if e["step"] == "check_retrieval"] == [
        "replan", "ok"]
    assert res.plan_trace[1]["top_prob_before"] < 0.5 < res.plan_trace[1]["top_prob_after"]


def test_confident_retrieval_never_replans():
    tools = ScriptedTools({}, {0: 4.0})
    llm = QueryLLM()
    res = run_case(case(), tools, llm, CFG)
    assert llm.replan_calls == 0 and res.abstain_reason is None
    assert [e["decision"] for e in res.plan_trace] == ["ok"]


def test_check_is_off_without_tau_retr():
    cfg = {"rag": CFG["rag"], "agent": {k: v for k, v in CFG["agent"].items() if k != "tau_retr"}}
    llm = QueryLLM()
    res = run_case(case(), ScriptedTools({}, {0: -4.0}), llm, cfg)
    assert llm.replan_calls == 0 and res.abstain_reason is None
    assert res.plan_trace[0]["decision"] == "ok"
