import pytest
from pydantic import ValidationError

from claimlens.schemas import Case, Chunk, Pair, Verdict


def test_chunk_validation():
    Chunk(chunk_id="a-0", doc_id="a", text="t", section_path="1.1 X", page=1)
    with pytest.raises(ValidationError):
        Chunk(chunk_id="a-0", doc_id="a", text="t", section_path="1.1 X", page=0)


def test_case_defaults_to_synthetic():
    c = Case(case_id="c1", policy_id="p", letter_text="l", reason="r",
             gold_clause_ids=["p-0001"], gold_verdict="SUPPORTED")
    assert c.source == "synthetic" and c.gold_verdict is Verdict.SUPPORTED


def test_pair_label_binary():
    with pytest.raises(ValidationError):
        Pair(pair_id="1", policy_id="p", reason="r", clause_id="c", clause_text="t", label=2)
