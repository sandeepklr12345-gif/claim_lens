"""Phase 2 data: split rules, verification, and leakage checks on the committed files."""

import json
from pathlib import Path

import pytest

from claimlens.data.phase2 import (
    N_VERIFY,
    QA_START,
    apply_verification,
    group_key,
    leakage,
    load_all_cases,
    load_pairs,
    load_splits,
    make_splits,
    replace_block,
    sample_for_verification,
    verification_table,
    write_verify_csv,
)
from claimlens.schemas import Chunk, Pair

DATA = Path(__file__).resolve().parents[1] / "data"
has_v2 = pytest.mark.skipif(not (DATA / "splits.json").exists(), reason="no Phase 2 data yet")


def chunk(i: int, doc: str = "a", section: str | None = None) -> Chunk:
    return Chunk(chunk_id=f"{doc}-{i:04d}", doc_id=doc, text=f"text {i}",
                 section_path=section or f"{i % 6}.1 Part > {i}", page=1)


def pair(i: int, clause: str, label: int, reason: str = "r") -> Pair:
    return Pair(pair_id=f"p{i}", policy_id="a", reason=reason, clause_id=clause,
                clause_text="c", label=label)


def test_group_key_skips_section_wrapper():
    c = Chunk(chunk_id="x", doc_id="d", text="t", page=1,
              section_path="SECTION B. BENEFITS > B-1 Base > 1")
    assert group_key(c) == "d::B-1 Base"
    assert group_key(chunk(1)) == "a::1.1 Part"


def test_splits_keep_groups_whole_and_gold_out_of_train():
    chunks = [chunk(i) for i in range(60)]
    positive = {c.chunk_id for c in chunks[:30]}
    gold = {c.chunk_id for c in chunks[:9]}
    splits = make_splits(chunks, positive, gold)
    assert set(splits.values()) == {"train", "val", "test"}
    assert not [g for g in gold if splits[g] == "train"]
    test_groups = {group_key(c) for c in chunks if splits[c.chunk_id] == "test"}
    assert all(splits[c.chunk_id] == "test" for c in chunks if group_key(c) in test_groups)


def test_splits_are_deterministic():
    chunks = [chunk(i) for i in range(40)]
    ids = {c.chunk_id for c in chunks[:20]}
    assert make_splits(chunks, ids, set()) == make_splits(chunks, ids, set())


def test_test_policy_holds_out_the_whole_policy():
    chunks = [chunk(i) for i in range(20)] + [chunk(i, "b") for i in range(10)]
    splits = make_splits(chunks, {c.chunk_id for c in chunks[:10]}, set(), test_policy="b")
    assert all(splits[c.chunk_id] == "test" for c in chunks if c.doc_id == "b")
    assert all(splits[c.chunk_id] != "test" for c in chunks if c.doc_id == "a")


def test_leakage_flags_a_reason_spanning_two_splits():
    splits = {"c1": "train", "c2": "test"}
    assert leakage([pair(1, "c1", 1, "x"), pair(2, "c1", 0, "y")], splits) == set()
    assert leakage([pair(1, "c1", 1, "x"), pair(2, "c2", 0, "x")], splits) == {"c1", "c2"}


def test_sample_is_balanced_and_prefers_test():
    pairs = [pair(i, f"c{i}", i % 2) for i in range(60)]
    splits = {f"c{i}": "test" if i < 30 else "train" for i in range(60)}
    got = sample_for_verification(pairs, splits)
    assert len(got) == N_VERIFY
    assert sum(p.label for p in got) == N_VERIFY // 2
    assert all(splits[p.clause_id] == "test" for p in got)


def test_blank_rows_stay_unverified_and_owner_label_wins(tmp_path):
    pairs = [pair(1, "c1", 1), pair(2, "c2", 0)]
    path = tmp_path / "v.csv"
    write_verify_csv(pairs, {"c1": "test", "c2": "test"}, path)
    text = path.read_text(encoding="utf-8-sig").splitlines()
    text[1] = text[1].replace(",1,,", ",1,0,clause does not fit")
    path.write_text("\n".join(text) + "\n", encoding="utf-8-sig")
    out, rows = apply_verification(pairs, path)
    assert (out[0].label, out[0].verified) == (0, True)
    assert (out[1].label, out[1].verified) == (0, False)
    assert len(rows) == 1


def test_bad_label_is_rejected(tmp_path):
    path = tmp_path / "v.csv"
    write_verify_csv([pair(1, "c1", 1)], {"c1": "test"}, path)
    text = path.read_text(encoding="utf-8-sig").replace(",1,,", ",1,yes,")
    path.write_text(text, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="0 or 1"):
        apply_verification([pair(1, "c1", 1)], path)


def test_qa_block_is_replaced_not_duplicated():
    row = {"pair_id": "p1", "generated_label": "1", "your_label": "0", "note": ""}
    first = replace_block("# QA\n", verification_table([row]))
    second = replace_block(first, verification_table([row]))
    assert first == second
    assert second.count(QA_START) == 1


@has_v2
def test_committed_pairs_have_no_leakage_and_required_fields():
    splits, _ = load_splits(DATA)
    pairs = load_pairs(DATA / "pairs_v2.jsonl")
    assert leakage(pairs, splits) == set()
    assert all(p.source == "synthetic" for p in pairs)
    assert {splits[p.clause_id] for p in pairs} == {"train", "val", "test"}


@has_v2
def test_committed_cases_never_use_train_clauses():
    splits, _ = load_splits(DATA)
    cases = load_all_cases(DATA / "cases")
    assert len(cases) >= 30
    assert all(c.source == "synthetic" for c in cases)
    assert [c.case_id for c in cases if splits[c.gold_clause_ids[0]] == "train"] == []


@has_v2
def test_new_cases_are_valid_json_on_disk():
    for f in (DATA / "cases").glob("*.json"):
        assert json.loads(f.read_text(encoding="utf-8"))["gold_clause_ids"]
