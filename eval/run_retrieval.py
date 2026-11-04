"""Retrieval quality: Recall@3 and MRR against gold_clause_ids (SPEC.md 8).

Query = the case's reason. Compares BM25 only, hybrid (BM25 + dense, RRF), and hybrid
re-ranked by the cross-encoder (top 10 -> top 3, as the agent does).
"""

from __future__ import annotations

import logging

import numpy as np

from claimlens.core import get_matcher, get_retriever
from claimlens.matcher.scorer import Matcher
from eval.common import load_cases, write_json


def rank_of(gold: set[str], ids: list[str]) -> int | None:
    for i, cid in enumerate(ids, start=1):
        if cid in gold:
            return i
    return None


def metrics(ranks: list[int | None]) -> dict[str, float]:
    return {
        "recall@3": float(np.mean([r is not None and r <= 3 for r in ranks])),
        "mrr@10": float(np.mean([1.0 / r if r else 0.0 for r in ranks])),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    cases = load_cases()
    matcher: Matcher = get_matcher()
    ranks: dict[str, list[int | None]] = {"bm25": [], "hybrid": [], "hybrid+rerank": []}
    for c in cases:
        r = get_retriever(c.policy_id)
        gold = set(c.gold_clause_ids)
        bm = [h.chunk.chunk_id for h in r.retrieve(c.reason, 10, mode="bm25")]
        hy_hits = r.retrieve(c.reason, 10, mode="hybrid")
        hy = [h.chunk.chunk_id for h in hy_hits]
        lg = matcher.logits([(c.reason, h.chunk.text) for h in hy_hits])
        rr = [hy[i] for i in np.argsort(-lg)]
        ranks["bm25"].append(rank_of(gold, bm))
        ranks["hybrid"].append(rank_of(gold, hy))
        ranks["hybrid+rerank"].append(rank_of(gold, rr))
    res = {"n_cases": len(cases), "query": "case.reason (clean synthetic reason)",
           "source": "synthetic", **{k: metrics(v) for k, v in ranks.items()}}
    write_json("retrieval.json", res)
    print(res)


if __name__ == "__main__":
    main()
