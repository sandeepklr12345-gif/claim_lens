"""Comprehensiveness, sufficiency and rank correlation. Torch-free (SPEC.md 13.4).

Comprehensiveness at k = f(all words) - f(without the top k words); higher is better.
Sufficiency at k = f(all words) - f(only the top k words); lower is better.
f is the matcher's logit: fine-tuned probabilities saturate near 1.0, which would hide changes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

KS = (10, 20, 50)
ScoreFn = Callable[[list[str]], np.ndarray]


def top_k(scores: Sequence[float], k: int) -> set[int]:
    """Indices of the k highest scores; ties go to the earlier word."""
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    return {int(i) for i in order[:k]}


def without(tokens: Sequence[str], idx: set[int]) -> str:
    return " ".join(t for i, t in enumerate(tokens) if i not in idx)


def only(tokens: Sequence[str], idx: set[int]) -> str:
    return " ".join(t for i, t in enumerate(tokens) if i in idx)


def faithfulness(score_fn: ScoreFn, tokens: Sequence[str], scores: Sequence[float],
                 ks: Sequence[int] = KS) -> dict[int, tuple[float, float]]:
    """{k: (comprehensiveness, sufficiency)}. k is capped at the clause's word count."""
    texts = [" ".join(tokens)]
    for k in ks:
        idx = top_k(scores, k)
        texts += [without(tokens, idx), only(tokens, idx)]
    out = np.asarray(score_fn(texts), dtype=np.float64)
    return {k: (float(out[0] - out[1 + 2 * i]), float(out[0] - out[2 + 2 * i]))
            for i, k in enumerate(ks)}


def average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    for v in np.unique(x):
        same = x == v
        if same.sum() > 1:
            ranks[same] = ranks[same].mean()
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Spearman correlation; None when either side is constant or too short."""
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(x) != len(y):
        raise ValueError(f"length mismatch: {len(x)} and {len(y)}")
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(average_ranks(x), average_ranks(y))[0, 1])
