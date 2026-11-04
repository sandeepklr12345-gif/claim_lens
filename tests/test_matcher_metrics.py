import numpy as np
import pytest

from claimlens.explain.wordpieces import l2_normalise, merge_wordpieces
from claimlens.matcher.metrics import accuracy, best_threshold, f1, roc_auc, split_groups


def test_roc_auc_perfect_inverse_and_ties():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    assert roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)


def test_accuracy_f1_threshold():
    y = np.array([0, 0, 1, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8, 0.9])
    t = best_threshold(y, s)
    pred = (s >= t).astype(int)
    assert f1(y, pred) >= f1(y, (s >= 0.5).astype(int))
    assert accuracy(y, np.array([0, 0, 1, 1, 0])) == pytest.approx(0.8)


def test_split_groups_is_disjoint_and_seeded():
    groups = [f"g{i}" for i in range(10)] * 2
    a1, b1 = split_groups(groups, 42)
    a2, b2 = split_groups(groups, 42)
    assert a1 == a2 and b1 == b2 and not (a1 & b1) and len(a1 | b1) == 10


def test_merge_wordpieces_sums_scores():
    words, scores = merge_wordpieces(["pre", "##exist", "##ing", "disease"], [1.0, 2.0, 3.0, 4.0])
    assert words == ["preexisting", "disease"]
    assert scores == [6.0, 4.0]


def test_l2_normalise():
    v = l2_normalise(np.array([3.0, 4.0]))
    assert np.linalg.norm(v) == pytest.approx(1.0)
    assert np.all(l2_normalise(np.zeros(3)) == 0)
