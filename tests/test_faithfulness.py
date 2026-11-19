"""Comprehensiveness, sufficiency, Spearman and attention rollout (no torch needed)."""

import numpy as np
import pytest

from claimlens.explain.explain import choose_tau, replay_reflection
from claimlens.explain.faithfulness import faithfulness, only, spearman, top_k, without
from claimlens.explain.rollout import rollout_scores


def key_word_model(texts: list[str]) -> np.ndarray:
    """A fake logit that depends only on whether the word 'key' is present."""
    return np.array([1.0 if "key" in t.split() else 0.0 for t in texts])


def test_top_k_is_capped_at_the_number_of_words():
    assert top_k([0.1, 0.9, 0.5], 2) == {1, 2}
    assert top_k([0.1, 0.9], 10) == {0, 1}


def test_without_and_only_are_complements():
    tokens = ["a", "b", "c"]
    assert without(tokens, {1}) == "a c"
    assert only(tokens, {1}) == "b"


def test_good_attribution_scores_better_than_bad():
    tokens = ["x", "key", "y", "z"]
    good = faithfulness(key_word_model, tokens, [0, 1, 0, 0], ks=(1,))[1]
    bad = faithfulness(key_word_model, tokens, [1, 0, 0, 0], ks=(1,))[1]
    assert good == (1.0, 0.0)  # removing it drops the score, keeping it alone is enough
    assert bad == (0.0, 1.0)


def test_spearman():
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 2, 2], [1, 1, 2, 2]) == pytest.approx(1.0)


def test_spearman_is_undefined_for_constant_input():
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    with pytest.raises(ValueError, match="length"):
        spearman([1, 2], [1, 2, 3])


HISTORY = [("initial", 0.3), ("revision_1", 0.5), ("revision_2", 0.45)]


@pytest.mark.parametrize(("tau", "want"), [
    (0.2, (0.3, 0)),  # the initial explanation already passes
    (0.4, (0.5, 1)),  # stops after the first revision
    (0.6, (0.5, 2)),  # runs out of rounds; the best is kept, not the last
])
def test_replay_stops_where_reflect_would(tau, want):
    assert replay_reflection(HISTORY, tau) == want


def test_replay_matches_a_real_reflect_run():
    from claimlens.explain.explain import reflect
    from claimlens.llm.client import FakeLLM
    from claimlens.schemas import Attribution

    clause = "alpha beta gamma delta"
    attr = Attribution(clause_id="c", tokens=clause.split(), scores=[1.0, 1.0, 1.0, 1.0])
    answers = [{"explanation": "e", "cited_spans": [{"text": t}]}
               for t in ("alpha", "alpha beta", "alpha beta gamma")]
    full = reflect(FakeLLM(answers), attr, "r", "c", clause, 0.9, "supports", tau=0.8)
    for tau in (0.2, 0.3, 0.5, 0.6, 0.8):
        run = reflect(FakeLLM(answers), attr, "r", "c", clause, 0.9, "supports", tau=tau)
        assert replay_reflection(full.history, tau) == (run.aas_final, len(run.history) - 1)


def test_choose_tau_prefers_the_best_then_the_lowest():
    climbs = [[("initial", 0.3), ("revision_1", 0.7)]]
    assert choose_tau(climbs, [0.2, 0.4, 0.6]) == 0.4
    flat = [[("initial", 0.3), ("revision_1", 0.3)]]
    assert choose_tau(flat, [0.4, 0.6, 0.8]) == 0.4


def test_rollout_follows_the_attention():
    attention = np.array([[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])  # CLS -> token 1
    scores = rollout_scores([attention, attention])
    assert scores[1] > scores[2]
    assert scores.sum() == pytest.approx(1.0)
    assert scores[2] == 0.0
