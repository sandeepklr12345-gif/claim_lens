"""Local-only tests (need torch, Captum and the HF cross-encoder). Deselected in CI."""

import math

import pytest

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def matcher():
    from claimlens.matcher.scorer import Matcher

    return Matcher()


def test_matcher_prefers_matching_clause(matcher):
    reason = "Claim rejected because the pre-existing disease waiting period has not ended."
    good = "Pre-existing diseases are covered only after 36 months of continuous coverage."
    bad = "The insured must intimate the company of any change of address in writing."
    lg = matcher.logits([(reason, good), (reason, bad)])
    assert lg[0] > lg[1]


def test_ig_attribution_shape_and_norm(matcher):
    from claimlens.explain.ig import attribute

    reason = "Claim rejected because of the 36 month waiting period."
    clause = "Pre-existing diseases are covered only after 36 months of continuous coverage."
    a = attribute(matcher, reason, clause, "c1", n_steps=20)
    assert len(a.tokens) == len(a.scores) > 5
    assert all(math.isfinite(s) for s in a.scores)
    assert any(s != 0 for s in a.scores)
    assert a.method == "integrated_gradients"
    assert "36" in a.tokens


def test_unknown_method_is_rejected(matcher):
    from claimlens.explain.ig import attribute

    with pytest.raises(ValueError, match="unknown attribution method"):
        attribute(matcher, "r", "clause text", method="lime")
    with pytest.raises(ValueError, match="unknown IG baseline"):
        attribute(matcher, "r", "clause text", baseline="noise")


def test_all_methods_return_the_same_words(matcher):
    from claimlens.explain.ig import attribute

    reason = "Claim rejected because of the 36 month waiting period."
    clause = "Pre-existing diseases are covered only after 36 months of continuous coverage."
    ig = attribute(matcher, reason, clause, "c1", n_steps=20)
    others = [attribute(matcher, reason, clause, "c1", method=m) for m in ("shap", "rollout")]
    others.append(attribute(matcher, reason, clause, "c1", n_steps=20, baseline="zero"))
    for other in others:
        assert other.tokens == ig.tokens
        assert len(other.scores) == len(ig.scores)
        assert any(s != 0 for s in other.scores)
    assert [o.method for o in others] == ["shap", "rollout", "integrated_gradients"]
