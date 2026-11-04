"""Matcher metrics on data/pairs.jsonl -> eval/results/matcher.json (SPEC.md 8).

The decision threshold is picked on half of the pairs (dev) and reported on the other half
(test); the split is by reason so a positive and its hard negative stay together.
"""

from __future__ import annotations

import json
import logging

import numpy as np

from claimlens.matcher.metrics import accuracy, best_threshold, f1, roc_auc, split_groups
from claimlens.matcher.scorer import Matcher, sigmoid
from claimlens.schemas import Pair
from claimlens.utils.config import ROOT

RESULTS = ROOT / "eval" / "results"


def load_pairs() -> list[Pair]:
    path = ROOT / "data" / "pairs.jsonl"
    return [Pair.model_validate_json(ln) for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    pairs = load_pairs()
    matcher = Matcher()
    logits = matcher.logits([(p.reason, p.clause_text) for p in pairs])
    prob = np.asarray(sigmoid(logits))
    y = np.array([p.label for p in pairs])
    groups = [p.pair_id.rsplit("_", 1)[0] for p in pairs]  # pair_007_pos -> pair_007
    dev_g, test_g = split_groups(groups, seed=42)
    dev = np.array([g in dev_g for g in groups])
    test = np.array([g in test_g for g in groups])

    thr = best_threshold(y[dev], prob[dev])
    pred = (prob >= thr).astype(int)
    res = {
        "model": matcher.name,
        "n_pairs": len(pairs),
        "n_positive": int(y.sum()),
        "source": "synthetic",
        "threshold_on_dev": thr,
        "dev": {"n": int(dev.sum()), "accuracy": accuracy(y[dev], pred[dev]),
                "f1": f1(y[dev], pred[dev]), "roc_auc": roc_auc(y[dev], prob[dev])},
        "test": {"n": int(test.sum()), "accuracy": accuracy(y[test], pred[test]),
                 "f1": f1(y[test], pred[test]), "roc_auc": roc_auc(y[test], prob[test])},
        "all_roc_auc": roc_auc(y, prob),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "matcher.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
