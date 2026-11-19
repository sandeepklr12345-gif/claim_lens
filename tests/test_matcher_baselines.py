"""Matcher baselines: zero-shot parsing, failure handling, BM25 and evaluation."""

import numpy as np
import pytest

pytest.importorskip("rank_bm25")

from claimlens.llm.client import FakeLLM  # noqa: E402
from claimlens.matcher.baselines import (  # noqa: E402
    bm25_pair_scores,
    evaluate,
    llm_pair_scores,
    zero_shot_score,
)
from claimlens.schemas import Chunk, Pair  # noqa: E402


def pair(clause_id: str = "d-1", reason: str = "waiting period") -> Pair:
    return Pair(pair_id="p", policy_id="d", reason=reason, clause_id=clause_id,
                clause_text="c", label=1)


@pytest.mark.parametrize(("raw", "want"), [
    ({"justified": True, "confidence": 0.8}, 0.8),
    ({"justified": False, "confidence": 0.8}, pytest.approx(0.2)),
    ({"justified": True, "confidence": 3}, 1.0),
    ({"justified": "yes", "confidence": 0.8}, None),
    ({"justified": True}, None),
    ({"justified": True, "confidence": True}, None),
])
def test_zero_shot_score(raw, want):
    assert zero_shot_score(raw) == want


def test_unusable_answers_are_counted_and_score_half():
    llm = FakeLLM([{"justified": True, "confidence": 0.9}, {"nonsense": 1}])
    scores, failed = llm_pair_scores(llm, [pair(), pair()])
    assert scores == [0.9, 0.5]
    assert failed == 1


def test_an_llm_error_does_not_lose_the_run():
    class Broken:
        def generate_json(self, prompt, system=None):
            raise RuntimeError("down")

    assert llm_pair_scores(Broken(), [pair()]) == ([0.5], 1)


def test_bm25_prefers_the_clause_with_the_reason_words():
    chunks = [Chunk(chunk_id=f"d-{i}", doc_id="d", text=t, section_path="s", page=1) for i, t in
              enumerate(["a waiting period applies here", "room rent is paid", "maternity cover",
                         "ambulance cost"])]
    hit, miss = bm25_pair_scores([pair("d-0"), pair("d-1")], {"d": chunks})
    assert hit > miss


def test_evaluate_uses_the_given_threshold():
    y, s = np.array([1, 1, 0, 0]), np.array([0.9, 0.6, 0.7, 0.1])
    assert evaluate(y, s, 0.8)["accuracy"] == 0.75
    assert evaluate(y, s, 0.5)["roc_auc"] == 0.75
