"""R1b hard-negative miner with a fake retriever and judge: caps, same-split rule, confirmation."""

from claimlens.data.audit import Judgement
from claimlens.data.hardneg import PER_REASON, confirmed_zero, mine, positive_reasons
from claimlens.schemas import Chunk, Pair, RetrievedClause


def chunk(i: int) -> Chunk:
    return Chunk(chunk_id=f"d-{i:04d}", doc_id="d", text=f"clause text number {i}",
                 section_path="s", page=1)


class FakeRetriever:
    def retrieve(self, query: str, k: int = 10, mode: str = "hybrid"):
        return [RetrievedClause(chunk=chunk(i), score=1.0) for i in range(k)]


class FakeJudge:
    """label 0/high for every clause except those listed in `positive_ids`."""

    def __init__(self, positive_ids=()):
        self.positive_ids, self.calls = set(positive_ids), 0

    def generate_json(self, prompt, system=None):
        self.calls += 1
        label = 1 if any(f"number {i}" in prompt for i in self.positive_ids) else 0
        return {"label": label, "confidence": "high", "rationale": "x"}


def setup(n_pairs=10):
    splits = {f"d-{i:04d}": ("train" if i < 4 else "val" if i < 7 else "test") for i in range(10)}
    pairs = [Pair(pair_id=f"p{i}", policy_id="d", reason=f"reason {i}", clause_id=f"d-{i % 10:04d}",
                  clause_text="t", label=1) for i in range(n_pairs)]
    return pairs, splits


def test_confirmed_zero_needs_both_passes_high():
    ok = Judgement(label=0, confidence="high", rationale="")
    assert confirmed_zero(ok, ok)
    assert not confirmed_zero(ok, Judgement(label=0, confidence="medium", rationale=""))
    assert not confirmed_zero(ok, Judgement(label=1, confidence="high", rationale=""))
    assert not confirmed_zero(Judgement(label=None, confidence="low", rationale=""), ok)


def test_positive_reasons_use_the_generated_label():
    pairs, _ = setup(2)
    pairs[0] = pairs[0].model_copy(update={"label": 0, "original_label": 1})  # owner-corrected
    pairs[1] = pairs[1].model_copy(update={"label": 1, "original_label": 0})
    assert [p.pair_id for p in positive_reasons(pairs)] == ["p0"]


def test_negatives_stay_in_the_reasons_split_and_never_use_the_gold_clause():
    pairs, splits = setup()
    tv, test = mine(pairs, splits, {"d": FakeRetriever()}, FakeJudge())
    by_id = {p.pair_id: p for p in pairs}
    assert tv and test
    for hn in tv + test:
        assert hn.hard_negative and hn.label == 0 and hn.source == "llm_audit"
        assert not hn.verified
    for hn in tv:
        assert splits[hn.clause_id] in ("train", "val")
    for hn in test:
        assert splits[hn.clause_id] == "test"
    for hn in tv + test:
        gold = next(p.clause_id for p in by_id.values() if p.reason == hn.reason)
        assert hn.clause_id != gold and splits[hn.clause_id] == splits[gold]


def test_caps_and_per_reason_limit():
    pairs, splits = setup()
    tv, test = mine(pairs, splits, {"d": FakeRetriever()}, FakeJudge(), max_trainval=3, max_test=1)
    assert len(tv) == 3 and len(test) <= 1
    from collections import Counter

    assert max(Counter(p.reason for p in tv + test).values()) <= PER_REASON
    assert len({p.pair_id for p in tv + test}) == len(tv) + len(test)


def test_judge_disagreement_rejects_and_cache_avoids_repeat_calls():
    pairs, splits = setup(1)
    judge = FakeJudge(positive_ids=range(10))  # judge calls every clause a positive
    tv, test = mine(pairs, splits, {"d": FakeRetriever()}, judge)
    assert tv == [] and test == []
    cache: dict = {}
    mine(pairs, splits, {"d": FakeRetriever()}, FakeJudge(), cache=cache)
    n = len(cache)
    again = FakeJudge()
    mine(pairs, splits, {"d": FakeRetriever()}, again, cache=cache)
    assert again.calls == 0 and len(cache) == n


def test_committed_hard_negative_files_respect_the_frozen_splits():
    """Real files: train/val hard negatives only in train/val clause groups, the test file only in
    test groups, nothing verified, and no clause appears in two splits."""
    from pathlib import Path

    import pytest

    from claimlens.data.phase2 import leakage, load_pairs, load_splits

    data = Path(__file__).resolve().parents[1] / "data"
    if not (data / "hard_negatives.jsonl").exists():
        pytest.skip("hard negatives not generated yet")
    splits, _ = load_splits(data)
    tv = load_pairs(data / "hard_negatives.jsonl")
    test = load_pairs(data / "hard_negatives_test.jsonl")
    assert 0 < len(tv) <= 60 and 0 < len(test) <= 20
    assert all(splits[p.clause_id] in ("train", "val") for p in tv)
    assert all(splits[p.clause_id] == "test" for p in test)
    for p in tv + test:
        assert p.label == 0 and p.hard_negative and p.source == "llm_audit" and not p.verified
    base = load_pairs(data / "pairs_v2.jsonl")
    assert leakage(base + tv + test, splits) == set()
    assert len({p.pair_id for p in tv + test}) == len(tv) + len(test)
