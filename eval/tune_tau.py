"""Tune verdict thresholds tau_high / tau_low (SPEC.md 7.6) and report honestly.

Phase 2: tuned on the validation cases of data/splits.json and reported once on the test
cases, for the checkpoint in configs/app.yaml. The Phase 1 run is kept at tag phase1-presented.
Writes eval/results/match_probs.json and eval/results/tau.json.
"""

from __future__ import annotations

import json
import logging

import numpy as np

from claimlens.agent.graph import extract_reason, retrieve_rerank
from claimlens.agent.tools import DirectTools
from claimlens.agent.verdict import decide
from claimlens.data.phase2 import load_splits
from claimlens.llm.client import OllamaClient
from claimlens.matcher.metrics import roc_auc
from claimlens.schemas import Verdict
from claimlens.utils.config import ROOT, load_config
from eval.common import load_cases, write_json


def collect() -> list[dict]:
    cfg = load_config()
    llm, tools = OllamaClient(), DirectTools()
    splits, _ = load_splits(ROOT / "data")
    rows = []
    for c in load_cases():
        reason = extract_reason(llm, c.letter_text)
        top = retrieve_rerank(tools, reason, c.policy_id, cfg["rag"]["top_k_fused"],
                              cfg["rag"]["top_k_return"])
        prob = tools.score_pair(reason, top[0].chunk.text, top[0].chunk.chunk_id).prob
        rows.append({"case_id": c.case_id, "gold": c.gold_verdict.value, "reason": reason,
                     "split": splits[c.gold_clause_ids[0]],
                     "best_clause": top[0].chunk.chunk_id, "prob": prob,
                     "gold_clause_hit": top[0].chunk.chunk_id in c.gold_clause_ids})
        logging.info("%s prob=%.3f gold=%s", c.case_id, prob, c.gold_verdict.value)
    return rows


def accuracy(rows: list[dict], hi: float, lo: float) -> float:
    ok = sum(decide(r["prob"], hi, lo).value == r["gold"] for r in rows)
    return ok / len(rows) if rows else 0.0


def tune(dev: list[dict]) -> tuple[float, float]:
    best = (-1.0, -1.0, 0.7, 0.3)  # (accuracy, abstain width, hi, lo)
    for lo in np.arange(0.02, 0.9, 0.02):
        for hi in np.arange(lo + 0.02, 0.98, 0.02):
            key = (accuracy(dev, hi, lo), hi - lo)  # ties -> wider abstain band
            if key > best[:2]:
                best = (key[0], key[1], float(hi), float(lo))
    return round(best[2], 2), round(best[3], 2)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    rows = collect()
    write_json("match_probs.json", rows)
    dev = [r for r in rows if r["split"] == "val"]
    test = [r for r in rows if r["split"] == "test"]
    hi, lo = tune(dev)
    y = np.array([1 if r["gold"] == Verdict.SUPPORTED else 0 for r in rows])
    p = np.array([r["prob"] for r in rows])
    y_test = np.array([1 if r["gold"] == Verdict.SUPPORTED else 0 for r in test])
    p_test = np.array([r["prob"] for r in test])
    res = {
        "tau_high": hi, "tau_low": lo, "n_val": len(dev), "n_test": len(test),
        "default_0.7_0.3": {"val": accuracy(dev, 0.7, 0.3), "test": accuracy(test, 0.7, 0.3)},
        "tuned": {"val": accuracy(dev, hi, lo), "test": accuracy(test, hi, lo)},
        "always_supported": {"val": float(np.mean([r["gold"] == "SUPPORTED" for r in dev])),
                             "test": float(np.mean([r["gold"] == "SUPPORTED" for r in test]))},
        "abstain_rate_test": float(np.mean([decide(r["prob"], hi, lo)
                                            == Verdict.INSUFFICIENT_EVIDENCE for r in test])),
        "prob_roc_auc_supported_vs_not_test": roc_auc(y_test, p_test),
        "prob_roc_auc_supported_vs_not_all": roc_auc(y, p),
        "top1_gold_clause_hit_rate_test": float(np.mean([r["gold_clause_hit"] for r in test])),
        "note": "tuned on validation cases only; test reported once",
    }
    write_json("tau.json", res)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
