"""Render IG attributions for 3 sample positive pairs -> eval/results/ig_samples.html."""

from __future__ import annotations

import logging

from claimlens.explain.ig import attribute
from claimlens.explain.render import write_html
from claimlens.matcher.scorer import Matcher, sigmoid
from eval.run_matcher import RESULTS, load_pairs


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    positives = [p for p in load_pairs() if p.label == 1][:3]
    matcher = Matcher()
    items = []
    for p in positives:
        attr = attribute(matcher, p.reason, p.clause_text, p.clause_id)
        logit = float(matcher.logits([(p.reason, p.clause_text)])[0])
        items.append((f"{p.pair_id} ({p.clause_id})", p.reason, float(sigmoid(logit)), attr))
    write_html(RESULTS / "ig_samples.html", items)
    print("wrote", RESULTS / "ig_samples.html")


if __name__ == "__main__":
    main()
