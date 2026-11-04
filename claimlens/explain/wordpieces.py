"""Torch-free helpers for turning token attributions into word attributions."""

from __future__ import annotations

import numpy as np


def merge_wordpieces(tokens: list[str], scores: list[float]) -> tuple[list[str], list[float]]:
    """Merge '##' sub-words into whole words; scores of the pieces are summed."""
    words: list[str] = []
    out: list[float] = []
    for t, s in zip(tokens, scores, strict=True):
        if t.startswith("##") and words:
            words[-1] += t[2:]
            out[-1] += s
        else:
            words.append(t)
            out.append(s)
    return words, out


def l2_normalise(x: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(x))
    return x / n if n > 0 else x
