"""R1b matcher re-evaluation with cleaned labels, hard negatives and three seeds -> matcher_v3.md.

    python -m eval.run_matcher_v3            score every system, then write the tables
    python -m eval.run_matcher_v3 --render   rebuild the tables from the saved scores

Rules (SPEC.md 14.3): splits are frozen (no pair moves); thresholds are picked on validation only;
test is never used to choose anything. The agent's checkpoint is the seed-42 run, fixed in advance.
Old labels are each pair's `original_label` (what the generator produced); new labels are the
current `label` (owner-reviewed where `verified`, judge labels where `source == "llm_audit"`).
Bootstrap: 1000 resamples of the test pairs, seed 42, 95% percentile intervals.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from claimlens.data.phase2 import load_pairs, load_splits
from claimlens.matcher.baselines import bm25_pair_scores
from claimlens.matcher.metrics import accuracy, best_threshold, f1, roc_auc
from claimlens.matcher.scorer import Matcher, sigmoid
from claimlens.rag.parse import list_policies, load_chunks
from claimlens.schemas import Pair
from claimlens.utils.config import ROOT, load_config
from eval.common import RESULTS, read_json, write_json

log = logging.getLogger(__name__)
SCORES = "matcher_v3_scores.json"
SEEDS = (42, 43, 44)
N_BOOT = 1000
V2_CHECKPOINT = "models/matcher_minilm"  # Phase 2 fine-tune on the original (unaudited) labels
V3_CHECKPOINTS = {s: f"models/matcher_v3_s{s}" for s in SEEDS}
METRICS = ("accuracy", "f1", "roc_auc")


def load_sets() -> dict[str, list[Pair]]:
    data = ROOT / "data"
    splits, _ = load_splits(data)
    pairs = load_pairs(data / "pairs_v2.jsonl")
    hn_file = data / "hard_negatives.jsonl"
    hn = load_pairs(hn_file) if hn_file.exists() else []
    hn_test = (load_pairs(data / "hard_negatives_test.jsonl")
               if (data / "hard_negatives_test.jsonl").exists() else [])
    return {"val": [p for p in pairs if splits[p.clause_id] == "val"],
            "val_hn": [p for p in hn if splits[p.clause_id] == "val"],
            "test": [p for p in pairs if splits[p.clause_id] == "test"],
            "test_hn": hn_test}


def score_all() -> None:
    sets = load_sets()
    allp = [p for ps in sets.values() for p in ps]
    texts = [(p.reason, p.clause_text) for p in allp]
    systems: dict[str, list[float]] = {}
    chunks = {d: load_chunks(d) for d in list_policies()}
    systems["BM25 score (baseline)"] = bm25_pair_scores(allp, chunks)
    systems["Phase 1 pretrained cross-encoder (baseline)"] = sigmoid(
        Matcher(load_config()["matcher"]["fallback_model"]).logits(texts)).tolist()
    if (ROOT / V2_CHECKPOINT).is_dir():
        systems["Phase 2 fine-tuned MiniLM (original labels)"] = sigmoid(
            Matcher(V2_CHECKPOINT).logits(texts)).tolist()
    for seed, path in V3_CHECKPOINTS.items():
        if (ROOT / path).is_dir():
            logits = Matcher(path).logits(texts)
            systems[f"v3 fine-tuned MiniLM seed {seed}"] = sigmoid(logits).tolist()
        else:
            log.warning("no checkpoint at %s", path)
    write_json(SCORES, {"pair_ids": [p.pair_id for p in allp], "systems": systems})


def metric_row(y: np.ndarray, s: np.ndarray, thr: float) -> dict[str, float]:
    pred = (s >= thr).astype(int)
    return {"accuracy": accuracy(y, pred), "f1": f1(y, pred), "roc_auc": roc_auc(y, s)}


def fmt(v: float) -> str:
    return "n/a" if np.isnan(v) else f"{v:.3f}"


def ci(values: np.ndarray) -> tuple[float, float]:
    v = values[~np.isnan(values)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (
        float("nan"), float("nan"))


def render() -> tuple[str, dict]:
    saved = read_json(SCORES)
    sets = load_sets()
    allp = [p for ps in sets.values() for p in ps]
    if [p.pair_id for p in allp] != saved["pair_ids"]:
        raise ValueError("saved scores do not match the current pairs; rerun without --render")
    pos = {name: slice(a, a + len(ps)) for name, ps, a in zip(
        sets, sets.values(), np.cumsum([0] + [len(ps) for ps in sets.values()])[:-1], strict=True)}
    sys_scores = {k: np.array(v) for k, v in saved["systems"].items()}

    def labels(name: str, which: str) -> np.ndarray:
        ps = sets[name]
        return np.array([(p.original_label if which == "old" and p.original_label is not None
                          else p.label) for p in ps])

    def sc(name: str, system: str) -> np.ndarray:
        return sys_scores[system][pos[name]]

    def threshold(system: str, which: str) -> float:
        y = np.concatenate([labels("val", which), labels("val_hn", which)])
        s = np.concatenate([sc("val", system), sc("val_hn", system)])
        return best_threshold(y, s)

    seeds_present = [s for s in SEEDS if f"v3 fine-tuned MiniLM seed {s}" in sys_scores]
    base = "Phase 1 pretrained cross-encoder (baseline)"
    v2 = "Phase 2 fine-tuned MiniLM (original labels)"
    names = [n for n in ("BM25 score (baseline)", base, v2) if n in sys_scores] + [
        f"v3 fine-tuned MiniLM seed {s}" for s in seeds_present]

    point = {}
    for which in ("old", "new"):
        for n in names:
            thr = threshold(n, which)
            y = labels("test", which)
            point[(n, which)] = (thr, metric_row(y, sc("test", n), thr))
    n_test = len(sets["test"])
    changed = sum((p.original_label is not None and p.original_label != p.label)
                  for p in sets["test"])
    n_train_changed = None  # train labels are not rescored here

    # bootstrap on the test pairs (new labels), shared resamples across systems and seeds
    rng = np.random.default_rng(42)
    idx = rng.integers(0, n_test, size=(N_BOOT, n_test))
    y_new = labels("test", "new")
    boot = {}
    for n in names:
        thr, s = point[(n, "new")][0], sc("test", n)
        boot[n] = {m: np.array([metric_row(y_new[i], s[i], thr)[m] for i in idx])
                   for m in METRICS}
    v3_names = [f"v3 fine-tuned MiniLM seed {s}" for s in seeds_present]
    if v3_names:
        boot["v3 mean of seeds"] = {m: np.mean([boot[n][m] for n in v3_names], axis=0)
                                    for m in METRICS}

    L = ["# Matcher re-evaluation with cleaned labels (R1b)", "",
         "All data is synthetic. Split: frozen from Phase 2 (held-out section groups, no third "
         "policy). Thresholds are picked on validation only (validation includes the "
         "train/val-group hard negatives); the test set is never used to choose anything. "
         "The agent's checkpoint is the seed-42 run, fixed in advance.", "",
         f"Test pairs: {n_test}. Labels that differ between old and new in the test split: "
         f"{changed}. Old labels = what the generator produced; new labels = owner-reviewed "
         "where `verified`, otherwise as listed in `docs/PROGRESS.md` (judge labels are marked "
         "`llm_audit`, not verified).", "",
         "## Test results, old labels and new labels side by side", "",
         "| System | Old: acc | Old: F1 | Old: AUC | New: acc | New: F1 | New: AUC |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for n in names:
        o, w = point[(n, "old")][1], point[(n, "new")][1]
        L.append(f"| {n} | {fmt(o['accuracy'])} | {fmt(o['f1'])} | {fmt(o['roc_auc'])} "
                 f"| {fmt(w['accuracy'])} | {fmt(w['f1'])} | {fmt(w['roc_auc'])} |")
    L += ["", "Thresholds are re-picked on the validation split under each label set. The "
          "Phase 2 model was trained on the original labels; the v3 models on the cleaned "
          "labels plus hard negatives.", ""]

    summary: dict = {"n_test": n_test, "seeds": seeds_present, "n_boot": N_BOOT}
    if v3_names:
        L += [f"## Seeds {', '.join(map(str, seeds_present))}: mean, standard deviation, "
              "95% bootstrap intervals (new labels)", "",
              "| System | Metric | Mean | SD across seeds | 95% bootstrap CI |",
              "|---|---|---:|---:|---|"]
        for m in METRICS:
            vals = np.array([point[(n, "new")][1][m] for n in v3_names])
            lo, hi = ci(boot["v3 mean of seeds"][m])
            sd = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else float("nan")
            L.append(f"| v3 fine-tuned MiniLM (mean of {len(v3_names)} seeds) | {m} | "
                     f"{fmt(float(np.nanmean(vals)))} | {fmt(sd)} | [{fmt(lo)}, {fmt(hi)}] |")
            summary[f"v3_{m}"] = {"mean": float(np.nanmean(vals)), "ci": [lo, hi]}
        for n in (base, v2):
            if n not in boot:
                continue
            for m in METRICS:
                lo, hi = ci(boot[n][m])
                L.append(f"| {n} | {m} | {fmt(point[(n, 'new')][1][m])} | n/a | "
                         f"[{fmt(lo)}, {fmt(hi)}] |")
        L += ["", "## Paired bootstrap: v3 (mean of seeds) minus the pretrained baseline", "",
              "Same resamples for both systems; the interval is for the difference.", "",
              "| Metric | Difference | 95% CI | CI excludes 0 |", "|---|---:|---|---|"]
        for m in METRICS:
            d = boot["v3 mean of seeds"][m] - boot[base][m]
            lo, hi = ci(d)
            diff = float(np.nanmean([point[(n, "new")][1][m] for n in v3_names])
                         - point[(base, "new")][1][m])
            L.append(f"| {m} | {diff:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
                     f"{'yes' if (lo > 0 or hi < 0) else 'no'} |")
            summary[f"diff_{m}"] = {"diff": diff, "ci": [lo, hi]}
        L.append("")

    # validation (new labels, with hard negatives)
    L += ["## Validation (new labels, includes val-group hard negatives)", "",
          "| System | Threshold | Acc | F1 | AUC |", "|---|---:|---:|---:|---:|"]
    for n in names:
        thr = point[(n, "new")][0]
        y = np.concatenate([labels("val", "new"), labels("val_hn", "new")])
        s = np.concatenate([sc("val", n), sc("val_hn", n)])
        r = metric_row(y, s, thr)
        L.append(f"| {n} | {thr:.3f} | {fmt(r['accuracy'])} | {fmt(r['f1'])} | "
                 f"{fmt(r['roc_auc'])} |")
    L.append(f"\nn: validation {len(sets['val'])} pairs + {len(sets['val_hn'])} hard negatives.\n")

    # hard-negative test file: every pair is a negative, so only accuracy is meaningful
    if sets["test_hn"]:
        L += [f"## Hard-negative test file (n = {len(sets['test_hn'])}, all label 0)", "",
              "Built from test-group clauses, judge-confirmed (`llm_audit`, not verified). "
              "Accuracy = share scored below the validation threshold.", "",
              "| System | Accuracy on hard negatives |", "|---|---:|"]
        for n in names:
            thr = point[(n, "new")][0]
            acc = float(np.mean(sc("test_hn", n) < thr))
            L.append(f"| {n} | {acc:.3f} |")
            summary.setdefault("hn_test_accuracy", {})[n] = acc
        L.append("")
    L += ["Test sets this small make differences unreliable: read the intervals, not the point "
          "estimates.", ""]
    _ = n_train_changed
    return "\n".join(L), summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()
    if not args.render:
        score_all()
    text, summary = render()
    (RESULTS / "matcher_v3.md").write_text(text, encoding="utf-8")
    write_json("matcher_v3.json", summary)
    log.info("wrote %s", RESULTS / "matcher_v3.md")


if __name__ == "__main__":
    main()
