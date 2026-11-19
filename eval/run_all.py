"""Run every evaluation and write eval/results/summary.md (SPEC.md 8, 13.4).

    python -m eval.run_all              # everything; per-case e2e results are cached
                                        # in eval/results/e2e/ (delete to rerun)
    python -m eval.run_all --summary    # only rebuild the tables from the saved results

All numbers in the README come from eval/results/. Data is synthetic and small.
Without a saved eval/results/matcher_v2_scores.json, the matcher comparison needs the
checkpoints in models/ (see eval.train_matcher).
"""

from __future__ import annotations

import argparse
import logging
import sys

from claimlens.data.phase2 import write_report
from claimlens.utils.config import ROOT
from eval import (
    run_appeal,
    run_audit,
    run_e2e,
    run_faithfulness,
    run_faithfulness_v2,
    run_matcher,
    run_matcher_v2,
    run_matcher_v3,
    run_replan,
    run_retrieval,
    tune_reflection,
    tune_tau,
)
from eval.common import RESULTS, read_json


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def build_summary() -> str:
    m, r, e, f, t = (read_json(n) for n in
                     ("matcher.json", "retrieval.json", "e2e.json", "faithfulness.json",
                      "tau.json"))
    golds = [row["gold"] for row in e["per_case"]]
    max_gold = max(golds.count(g) for g in set(golds))  # majority-class baseline
    base = max_gold / len(golds)
    L = ["# Evaluation summary", "",
         f"All data is **synthetic** and small ({e['n_cases']} cases); labels are by construction "
         "and only 5 were hand-checked (see `data/QA.md`). Treat numbers as indicative. Phase 2 "
         "tables: `data_v2.md`, `matcher_v2.md`, `faithfulness_v2.md`, `reflection_v2.md`.", "",
         f"## Retrieval (query = case reason, {e['n_cases']} cases)", "",
         "| Method | Recall@3 | MRR@10 |", "|---|---|---|"]
    for k in ("bm25", "hybrid", "hybrid+rerank"):
        L.append(f"| {k} | {r[k]['recall@3']:.3f} | {r[k]['mrr@10']:.3f} |")
    L += ["", "## Phase 1 matcher (pretrained cross-encoder, no fine-tuning, 80 pairs)", "",
          f"Model `{m['model']}`. Threshold picked on the dev half ({m['threshold_on_dev']:.3f}), "
          "reported on the held-out half.", "",
          "| Split | n | Accuracy | F1 | ROC-AUC |", "|---|---|---|---|---|"]
    for k in ("dev", "test"):
        d = m[k]
        L.append(f"| {k} | {d['n']} | {d['accuracy']:.3f} | {d['f1']:.3f} | {d['roc_auc']:.3f} |")
    by = e["by_split"]
    L += ["", "## Verdict thresholds and end-to-end agreement", "",
          f"Tuned on {t['n_val']} validation cases: tau_high = {t['tau_high']}, "
          f"tau_low = {t['tau_low']}. Reported once on {t['n_test']} test cases.", "",
          "| Thresholds | validation | test |", "|---|---:|---:|",
          f"| default 0.7 / 0.3 | {t['default_0.7_0.3']['val']:.2f} | "
          f"{t['default_0.7_0.3']['test']:.2f} |",
          f"| tuned {t['tau_high']} / {t['tau_low']} | {t['tuned']['val']:.2f} | "
          f"{t['tuned']['test']:.2f} |",
          f"| always SUPPORTED (baseline) | {t['always_supported']['val']:.2f} | "
          f"{t['always_supported']['test']:.2f} |", "",
          f"End-to-end (letter in, LLM-extracted reason, agent verdict) vs `gold_verdict`: "
          f"**test {by['test']['n_agree']}/{by['test']['n']}, validation "
          f"{by['val']['n_agree']}/{by['val']['n']}**, all {e['n_cases']} cases "
          f"{e['n_agree']}/{e['n_cases']} ({pct(e['verdict_agreement'])}). Abstain rate "
          f"{pct(e['abstain_rate'])}, gold clause in top-3 {pct(e['gold_clause_in_top3_rate'])}. "
          f"ROC-AUC of the match probability for SUPPORTED vs NOT_SUPPORTED on test: "
          f"{t['prob_roc_auc_supported_vs_not_test']:.3f}. "
          f"Baseline: always answering SUPPORTED agrees on {pct(base)} of all cases "
          f"({max_gold}/{e['n_cases']}). "
          "The reason is re-extracted by the LLM on every run, so these verdicts can differ "
          "slightly from the `tau.json` run.", "",
          "Confusion (gold -> predicted):", ""]
    for g, row in e["confusion_gold_to_pred"].items():
        L.append(f"- {g}: " + ", ".join(f"{p} {n}" for p, n in row.items()))
    L += ["", "## Faithfulness: AAS before vs after reflection", "",
          "| | Mean AAS |", "|---|---|",
          f"| initial explanation | {f['mean_aas_initial']:.3f} |",
          f"| final (after reflection, best kept) | {f['mean_aas_final']:.3f} |",
          f"| random spans of the same size (chance) | {f['mean_random_span_aas']:.3f} |", "",
          f"Over {f['n_cases']} cases: improved {f['n_improved']}, unchanged "
          f"{f['n_unchanged']}, worse {f['n_worse']} (best kept, so worse should be 0). "
          f"Reflection was triggered (initial AAS < 0.6) in {f['n_reflection_triggered']} cases."]
    if f["n_reflection_triggered"]:
        L.append(f"On those: mean AAS {f['mean_aas_initial_triggered']:.3f} -> "
                 f"{f['mean_aas_final_triggered']:.3f}.")
    L += ["", f"Caveat: {f['caveat']}", "", "Raw results: `matcher.json`, `retrieval.json`, "
          "`tau.json`, `e2e.json`, `faithfulness.json`, `e2e/` (per case).", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true", help="only rebuild summary.md")
    ap.add_argument("--retune", action="store_true",
                    help="re-run threshold tuning (then copy tau_high/low into configs/app.yaml)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    argv = sys.argv
    if not args.summary:
        sys.argv = [argv[0]]  # the steps parse their own arguments
        run_matcher.main()
        if not (RESULTS / run_matcher_v2.SCORES).exists():
            run_matcher_v2.score_all()
        run_retrieval.main()
        if args.retune or not (RESULTS / "tau.json").exists():
            tune_tau.main()  # then update configs/app.yaml tau_high/tau_low by hand
        run_e2e.main()
        tune_reflection.main()
        run_faithfulness.main()
        run_faithfulness_v2.main()
        # Phase 2b (SPEC.md 14): audit report, v3 matcher table, appeal drafts, re-plan check
        run_audit.main()
        run_matcher_v3.main()
        run_appeal.main()  # drafts are cached in eval/results/appeal/ (delete to redraft)
        if not (RESULTS / "replan_v3.json").exists():  # slow (re-plans cases); delete to rerun
            run_replan.main()
    (RESULTS / "matcher_v2.md").write_text(run_matcher_v2.render(), encoding="utf-8")
    write_report(ROOT / "data", RESULTS / "data_v2.md")
    (RESULTS / "summary.md").write_text(build_summary(), encoding="utf-8")
    print("wrote", RESULTS / "summary.md")


if __name__ == "__main__":
    main()
