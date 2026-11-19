"""R1a audit: flagging rule, judge-output cleaning, prompt order and the review-queue rules."""

from claimlens.data.audit import (
    MAX_FLAGGED,
    N_RANDOM,
    Judgement,
    audit_row,
    build_prompt,
    build_queue,
    clean_judgement,
    combined_confidence,
    combined_label,
    is_flagged,
    judge_pair,
)
from claimlens.schemas import Pair


def j(label: int | None, conf: str = "high") -> Judgement:
    return Judgement(label=label, confidence=conf, rationale="r")


class FakeLLM:
    def __init__(self, answers: list[dict] | Exception) -> None:
        self.answers, self.prompts = answers, []

    def generate_json(self, prompt: str, system: str | None = None) -> dict:
        self.prompts.append(prompt)
        if isinstance(self.answers, Exception):
            raise self.answers
        return self.answers.pop(0)


def test_flag_rule():
    assert not is_flagged(1, j(1), j(1))
    assert not is_flagged(0, j(0, "medium"), j(0, "high"))
    assert is_flagged(1, j(0), j(1))  # first pass disagrees
    assert is_flagged(1, j(1), j(0))  # second pass disagrees
    assert is_flagged(1, j(1, "low"), j(1))  # low confidence alone flags
    assert is_flagged(0, j(0), j(0, "low"))
    assert is_flagged(1, j(None, "low"), j(1))  # unusable judge output flags


def test_clean_judgement_normalises():
    c = clean_judgement({"label": "1", "confidence": "HIGH", "rationale": " ".join(["w"] * 50)})
    assert c.label == 1 and c.confidence == "high" and len(c.rationale.split()) == 30
    assert clean_judgement({"label": 2, "confidence": "high"}).label is None
    bad = clean_judgement({"label": "yes", "confidence": "high"})
    assert bad.label is None and bad.confidence == "low"
    assert clean_judgement({"label": 0, "confidence": "sure"}).confidence == "low"


def test_second_pass_changes_order():
    first = build_prompt("REASON-X", "CLAUSE-Y", swapped=False)
    second = build_prompt("REASON-X", "CLAUSE-Y", swapped=True)
    assert first.index("REASON-X") < first.index("CLAUSE-Y")
    assert second.index("CLAUSE-Y") < second.index("REASON-X")


def test_judge_pair_uses_both_orders_and_survives_errors():
    llm = FakeLLM([{"label": 1, "confidence": "high", "rationale": "ok"}] * 2)
    judge_pair(llm, "R", "C", swapped=False)
    judge_pair(llm, "R", "C", swapped=True)
    assert llm.prompts[0] != llm.prompts[1]
    failed = judge_pair(FakeLLM(RuntimeError("down")), "R", "C", swapped=False)
    assert failed.label is None and is_flagged(1, failed, j(1))


def test_combined_label_and_confidence():
    assert combined_label(j(1), j(1)) == 1
    assert combined_label(j(1), j(0)) is None
    assert combined_label(j(None), j(None)) is None
    assert combined_confidence(j(1, "high"), j(1, "low")) == "low"


def pairs_and_rows(n: int, verified: set[str], flag_every: int):
    pairs, rows = {}, []
    for i in range(n):
        p = Pair(pair_id=f"p{i}", policy_id="a", reason="r", clause_id=f"c{i}", clause_text="t",
                 label=i % 2, verified=f"p{i}" in verified)
        pairs[p.pair_id] = p
        row = audit_row(p, p.label, j(1 - p.label if i % flag_every == 0 else p.label), j(p.label),
                        "train")
        rows.append(row)
    return pairs, rows


def test_audit_row_is_llm_audit_and_never_verified():
    _, rows = pairs_and_rows(4, set(), 2)
    assert all(r["source"] == "llm_audit" and r["verified"] is False for r in rows)


def test_queue_caps_excludes_verified_and_adds_random():
    verified = {f"p{i}" for i in range(0, 10)}
    pairs, rows = pairs_and_rows(200, verified, flag_every=2)  # many flagged, some verified
    queue = build_queue(rows, pairs)
    ids = [q["pair_id"] for q in queue]
    assert len(queue) <= MAX_FLAGGED + N_RANDOM <= 40
    assert not verified & set(ids)
    assert len(set(ids)) == len(ids)
    flagged_ids = {r["pair_id"] for r in rows if r["flagged"]}
    assert sum(i not in flagged_ids for i in ids) == N_RANDOM  # the unflagged sample
    assert all(q["your_label"] == "" for q in queue)


def test_queue_with_few_flags_is_small():
    pairs, rows = pairs_and_rows(30, set(), flag_every=10)  # 3 flagged
    queue = build_queue(rows, pairs)
    assert len(queue) == 3 + N_RANDOM
