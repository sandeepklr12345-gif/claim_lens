"""Matcher baselines (SPEC.md 13.3): BM25 score and a zero-shot Ollama LLM."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from claimlens.llm.client import LLM
from claimlens.matcher.metrics import accuracy, f1, roc_auc
from claimlens.rag.retriever import tokenize
from claimlens.schemas import Chunk, Pair

log = logging.getLogger(__name__)

ZERO_SHOT_SYSTEM = "You audit Indian health-insurance claim rejections. Reply with JSON only."
ZERO_SHOT_PROMPT = """Rejection reason given by the insurer:
\"\"\"{reason}\"\"\"

Policy clause:
\"\"\"{clause}\"\"\"

Does this clause justify rejecting the claim for this reason?
Reply as JSON: {{"justified": true or false, "confidence": <number from 0 to 1>}}"""


def bm25_pair_scores(pairs: Sequence[Pair], chunks_by_doc: dict[str, list[Chunk]]) -> list[float]:
    """BM25 of the reason against the pair's clause, within that policy's corpus."""
    index = {}
    for doc, chunks in chunks_by_doc.items():
        bm25 = BM25Okapi([tokenize(c.text + " " + c.section_path) for c in chunks])
        index[doc] = (bm25, {c.chunk_id: i for i, c in enumerate(chunks)})
    out = []
    for p in pairs:
        bm25, ids = index[p.policy_id]
        out.append(float(bm25.get_scores(tokenize(p.reason))[ids[p.clause_id]]))
    return out


def zero_shot_score(raw: dict) -> float | None:
    """P(justified) from the LLM's answer; None if the answer is unusable."""
    justified, conf = raw.get("justified"), raw.get("confidence")
    if not isinstance(justified, bool) or isinstance(conf, bool):
        return None
    if not isinstance(conf, int | float):
        return None
    conf = min(max(float(conf), 0.0), 1.0)
    return conf if justified else 1.0 - conf


def llm_pair_scores(llm: LLM, pairs: Sequence[Pair]) -> tuple[list[float], int]:
    """Scores for each pair and how many failed; a failed pair scores the uninformative 0.5."""
    out: list[float] = []
    failed = 0
    for i, p in enumerate(pairs):
        try:
            raw = llm.generate_json(ZERO_SHOT_PROMPT.format(reason=p.reason, clause=p.clause_text),
                                    ZERO_SHOT_SYSTEM)
            score = zero_shot_score(raw)
        except Exception as e:  # noqa: BLE001 - one bad answer must not lose the whole run
            log.warning("zero-shot failed on %s: %s", p.pair_id, e)
            score = None
        failed += score is None
        out.append(0.5 if score is None else score)
        if (i + 1) % 20 == 0:
            log.info("zero-shot %d/%d", i + 1, len(pairs))
    return out, failed


def evaluate(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = (score >= threshold).astype(int)
    return {"n": len(y), "accuracy": accuracy(y, pred), "f1": f1(y, pred),
            "roc_auc": roc_auc(y, score)}
