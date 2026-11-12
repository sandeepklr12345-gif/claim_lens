"""R1b hard negatives (SPEC.md 14.3): retrieved-but-wrong clauses, confirmed by the LLM judge.

    python -m claimlens.data.hardneg

For each generated-positive reason, take the top retrieved clauses that are not its gold clause.
A candidate is kept only if BOTH judge passes say label 0 with high confidence. Candidates must
sit in the same split as the reason's own pair (so no reason text crosses splits): train and val
candidates go to `data/hard_negatives.jsonl` (cap 60, added to training and validation only);
test-group candidates go to `data/hard_negatives_test.jsonl` (cap 20, evaluation only).
Hard negatives are `source: "llm_audit"`, `verified: false`, `hard_negative: true`.
Deviation from "top-5 retrieved": the top 10 are retrieved and the first 5 non-gold clauses in the
reason's own split are considered.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from claimlens.data.audit import Judgement, judge_pair
from claimlens.data.phase2 import load_pairs, load_splits, write_pairs
from claimlens.llm.client import LLM, OllamaClient
from claimlens.schemas import Pair
from claimlens.utils.config import ROOT

log = logging.getLogger(__name__)
SEED = 42
MAX_TRAINVAL = 60
MAX_TEST = 20
TOP_K = 10
CONSIDER = 5
PER_REASON = 2


def confirmed_zero(j1: Judgement, j2: Judgement) -> bool:
    """Both passes label 0 and both are high confidence."""
    return all(j.label == 0 and j.confidence == "high" for j in (j1, j2))


def positive_reasons(pairs: list[Pair]) -> list[Pair]:
    """Generated positives (by the label the generator produced), in a fixed order."""
    return [p for p in pairs if (p.original_label if p.original_label is not None
                                 else p.label) == 1]


def mine(pairs: list[Pair], splits: dict[str, str], retrievers: dict[str, Any], llm: LLM,
         seed: int = SEED, max_trainval: int = MAX_TRAINVAL, max_test: int = MAX_TEST,
         cache: dict[str, Any] | None = None, on_new: Any = None) -> tuple[list[Pair], list[Pair]]:
    """Returns (train/val hard negatives, test hard negatives)."""
    cache = cache if cache is not None else {}
    reasons = positive_reasons(pairs)
    random.Random(seed).shuffle(reasons)
    trainval: list[Pair] = []
    test: list[Pair] = []
    used: set[tuple[str, str]] = set()
    for pos in reasons:
        if len(trainval) >= max_trainval and len(test) >= max_test:
            break
        split = splits[pos.clause_id]
        bucket, cap = (test, max_test) if split == "test" else (trainval, max_trainval)
        if len(bucket) >= cap:
            continue
        hits = retrievers[pos.policy_id].retrieve(pos.reason, k=TOP_K)
        cands = [h.chunk for h in hits if h.chunk.chunk_id != pos.clause_id
                 and splits.get(h.chunk.chunk_id) == split][:CONSIDER]
        kept = 0
        for c in cands:
            if kept >= PER_REASON or len(bucket) >= cap or (pos.pair_id, c.chunk_id) in used:
                continue
            key = f"{pos.pair_id}|{c.chunk_id}"
            if key not in cache:
                j1 = judge_pair(llm, pos.reason, c.text, swapped=False)
                j2 = judge_pair(llm, pos.reason, c.text, swapped=True) if (
                    j1.label == 0 and j1.confidence == "high") else j1
                cache[key] = {"j1": j1.model_dump(), "j2": j2.model_dump()}
                if on_new:
                    on_new(key, cache[key])
            j1, j2 = Judgement(**cache[key]["j1"]), Judgement(**cache[key]["j2"])
            if confirmed_zero(j1, j2):
                used.add((pos.pair_id, c.chunk_id))
                bucket.append(Pair(
                    pair_id=f"hn_{len(trainval) + len(test) + 1:03d}", policy_id=pos.policy_id,
                    reason=pos.reason, clause_id=c.chunk_id, clause_text=c.text, label=0,
                    source="llm_audit", verified=False, original_label=0, hard_negative=True))
                kept += 1
    return trainval, test


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from claimlens.rag.retriever import build_retriever

    data = ROOT / "data"
    pairs = load_pairs(data / "pairs_v2.jsonl")
    splits, _ = load_splits(data)
    retrievers = {d: build_retriever(d) for d in {p.policy_id for p in pairs}}
    cache_path = ROOT / "logs" / "hardneg_cache.jsonl"
    cache: dict[str, Any] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            cache[d["key"]] = d["val"]
    out = cache_path.open("a", encoding="utf-8")

    def save(key: str, val: Any) -> None:
        out.write(json.dumps({"key": key, "val": val}) + "\n")
        out.flush()
        if len(cache) % 10 == 0:
            log.info("judged %d candidate clauses", len(cache))

    llm = OllamaClient()
    problem = llm.health()
    if problem:
        raise SystemExit(problem)
    tv, test = mine(pairs, splits, retrievers, llm, cache=cache, on_new=save)
    write_pairs(tv, data / "hard_negatives.jsonl")
    write_pairs(test, Path(data / "hard_negatives_test.jsonl"))
    log.info("hard negatives: %d train/val, %d test", len(tv), len(test))


if __name__ == "__main__":
    main()
