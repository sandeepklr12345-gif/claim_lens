"""Accuracy, F1, ROC-AUC and threshold selection in plain NumPy (no scikit-learn)."""

from __future__ import annotations

import random

import numpy as np


def accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(y == pred)) if len(y) else 0.0


def f1(y: np.ndarray, pred: np.ndarray) -> float:
    tp = float(np.sum((y == 1) & (pred == 1)))
    fp = float(np.sum((y == 0) & (pred == 1)))
    fn = float(np.sum((y == 1) & (pred == 0)))
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U) with average ranks for ties."""
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    for v in np.unique(score):  # average tied ranks
        m = score == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    u = ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def best_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Threshold on `score` maximising F1 (ties broken by higher accuracy)."""
    best = (-1.0, -1.0, 0.0)
    for t in np.unique(score):
        pred = (score >= t).astype(int)
        key = (f1(y, pred), accuracy(y, pred))
        if key > best[:2]:
            best = (key[0], key[1], float(t))
    return best[2]


def split_groups(groups: list[str], seed: int = 42) -> tuple[set[str], set[str]]:
    """Split group ids into two halves (dev, test) so a reason's pairs stay together."""
    uniq = sorted(set(groups))
    random.Random(seed).shuffle(uniq)
    half = len(uniq) // 2
    return set(uniq[:half]), set(uniq[half:])
