"""Matcher comparison on pairs_v2 -> eval/results/matcher_v2.md (SPEC.md 13.3).

    python -m eval.run_matcher_v2            score every system, then write the table
    python -m eval.run_matcher_v2 --render   rebuild the table from the saved scores

Scores live in eval/results/matcher_v2_scores.json, so owner-verified labels can be scored
without a rerun and a fresh clone (no models/) can still rebuild the table. Each system's
threshold is picked on validation only. The selected system is the best on validation; test
is never used to choose.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from claimlens.data.phase2 import load_pairs, load_splits
from claimlens.llm.client import OllamaClient
from claimlens.matcher.baselines import bm25_pair_scores, evaluate, llm_pair_scores
from claimlens.matcher.metrics import best_threshold
from claimlens.matcher.scorer import Matcher, sigmoid
from claimlens.rag.parse import list_policies, load_chunks
from claimlens.schemas import Pair
from claimlens.utils.config import ROOT, load_config
from eval.common import RESULTS, read_json, write_json

log = logging.getLogger(__name__)
SCORES = "matcher_v2_scores.json"
FINE_TUNED = {
    "fine-tuned legal-bert": "models/matcher_legalbert",
    "fine-tuned InLegalBERT": "models/matcher_inlegalbert",
    "fine-tuned ms-marco-MiniLM": "models/matcher_minilm",
}


def eval_pairs() -> tuple[list[Pair], dict[str, str], str]:
    data = ROOT / "data"
    splits, mode = load_splits(data)
    pairs = [p for p in load_pairs(data / "pairs_v2.jsonl") if splits[p.clause_id] != "train"]
    return pairs, splits, mode


def score_all() -> None:
    pairs, _, _ = eval_pairs()
    ids = [p.pair_id for p in pairs]
    texts = [(p.reason, p.clause_text) for p in pairs]
    systems: dict[str, list[float]] = {}
    chunks = {d: load_chunks(d) for d in list_policies()}
    systems["BM25 score (baseline)"] = bm25_pair_scores(pairs, chunks)
    matcher = Matcher(load_config()["matcher"]["fallback_model"])
    systems["Phase 1 pretrained cross-encoder (baseline)"] = sigmoid(
        matcher.logits(texts)).tolist()
    llm_scores, failed = llm_pair_scores(OllamaClient(), pairs)
    systems["zero-shot Ollama LLM (baseline)"] = llm_scores
    for name, path in FINE_TUNED.items():
        if (ROOT / path).is_dir():
            systems[name] = sigmoid(Matcher(path).logits(texts)).tolist()
        else:
            log.warning("no checkpoint for %s at %s", name, path)
    write_json(SCORES, {"llm_failed": failed, "pair_ids": ids, "systems": systems})


def render() -> str:
    saved = read_json(SCORES)
    pairs, splits, mode = eval_pairs()
    by_id = {p.pair_id: p for p in pairs}
    if set(saved["pair_ids"]) != set(by_id):
        raise ValueError("saved scores do not match pairs_v2; rerun without --render")
    ordered = [by_id[i] for i in saved["pair_ids"]]
    y = np.array([p.label for p in ordered])
    split = np.array([splits[p.clause_id] for p in ordered])
    verified = np.array([p.verified for p in ordered])
    n_ver = int(verified.sum())

    rows, chosen = [], ("", (-1.0, -1.0))
    for name, values in saved["systems"].items():
        score = np.array(values)
        val, test = split == "val", split == "test"
        thr = best_threshold(y[val], score[val])
        r_val, r_test = evaluate(y[val], score[val], thr), evaluate(y[test], score[test], thr)
        row = {"name": name, "thr": thr, "val": r_val, "test": r_test}
        if n_ver:
            row["verified"] = evaluate(y[verified], score[verified], thr)
        rows.append(row)
        if name.startswith("fine-tuned") and (r_val["f1"], r_val["roc_auc"]) > chosen[1]:
            chosen = (name, (r_val["f1"], r_val["roc_auc"]))

    L = ["# Matcher comparison (Phase 2)", "",
         f"All data is synthetic. Split: {mode}. Thresholds are picked on validation only. "
         f"Zero-shot LLM unusable answers: {saved['llm_failed']} (scored 0.5).", "",
         "| System | Threshold | Val acc | Val F1 | Val AUC | Test acc | Test F1 | Test AUC |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        v, t = r["val"], r["test"]
        L.append(f"| {r['name']} | {r['thr']:.3f} | {v['accuracy']:.3f} | {v['f1']:.3f} "
                 f"| {v['roc_auc']:.3f} | {t['accuracy']:.3f} | {t['f1']:.3f} "
                 f"| {t['roc_auc']:.3f} |")
    n_val, n_test = rows[0]["val"]["n"], rows[0]["test"]["n"]
    best = max(rows, key=lambda r: (r["val"]["f1"], r["val"]["roc_auc"]))["name"]
    L += ["", f"n: validation {n_val}, test {n_test}. Test sets this small make small "
          "differences unreliable.", "",
          f"Best fine-tuned model on validation (the agent's checkpoint): {chosen[0] or 'none'}.",
          f"Best system overall on validation F1: {best}."]
    if best != chosen[0]:
        L.append("The fine-tuned model does NOT beat the best baseline on validation.")
    L.append("")
    if n_ver:
        L += ["## Owner-verified subset", "", f"n = {n_ver}.", "",
              "| System | Accuracy | F1 | ROC-AUC |", "|---|---:|---:|---:|"]
        L += [f"| {r['name']} | {r['verified']['accuracy']:.3f} | {r['verified']['f1']:.3f} "
              f"| {r['verified']['roc_auc']:.3f} |" for r in rows]
    else:
        L.append("## Owner-verified subset\n\nPENDING: 0 pairs verified so far.")
    return "\n".join(L) + "\n"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true", help="only rebuild the table")
    args = ap.parse_args()
    if not args.render:
        score_all()
    (RESULTS / "matcher_v2.md").write_text(render(), encoding="utf-8")
    log.info("wrote %s", RESULTS / "matcher_v2.md")


if __name__ == "__main__":
    main()
