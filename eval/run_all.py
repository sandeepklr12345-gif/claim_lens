"""Run every evaluation and write eval/results/summary.md (SPEC.md 8).

    python -m eval.run_all              # everything; per-case e2e results are cached
                                        # in eval/results/e2e/ (delete to rerun, ~10 min)
    python -m eval.run_all --summary    # only rebuild summary.md from the saved JSON

All numbers in the README come from eval/results/. Data is synthetic and small (20 cases).
"""

from __future__ import annotations

import argparse
import logging

from eval import run_e2e, run_faithfulness, run_matcher, run_retrieval, tune_tau
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
         "All data is **synthetic** and small (20 cases, 80 pairs); labels are by construction "
         "and only 5 were hand-checked (see `data/QA.md`). Treat numbers as indicative.", "",
         "## Retrieval (query = case reason, 20 cases)", "",
         "| Method | Recall@3 | MRR@10 |", "|---|---|---|"]
    for k in ("bm25", "hybrid", "hybrid+rerank"):
        L.append(f"| {k} | {r[k]['recall@3']:.3f} | {r[k]['mrr@10']:.3f} |")
    L += ["", "## Matcher (pretrained cross-encoder, no fine-tuning, 80 pairs)", "",
          f"Model `{m['model']}`. Threshold picked on the dev half ({m['threshold_on_dev']:.3f}), "
          "reported on the held-out half.", "",
          "| Split | n | Accuracy | F1 | ROC-AUC |", "|---|---|---|---|---|"]
    for k in ("dev", "test"):
        d = m[k]
        L.append(f"| {k} | {d['n']} | {d['accuracy']:.3f} | {d['f1']:.3f} | {d['roc_auc']:.3f} |")
    L += ["", "## Verdict thresholds and end-to-end agreement", "",
          f"Tuned on a stratified dev split of 10 cases (SPEC asked for the first 10, but those "
          f"are all SUPPORTED): tau_high = {t['tau_high']}, tau_low = {t['tau_low']}.", "",
          "| Thresholds | dev (10) | held-out (10) | all 20 |", "|---|---|---|---|",
          f"| default 0.7 / 0.3 | {t['default_0.7_0.3']['dev']:.2f} | "
          f"{t['default_0.7_0.3']['test']:.2f} | {t['default_0.7_0.3']['all20']:.2f} |",
          f"| tuned {t['tau_high']} / {t['tau_low']} | {t['tuned']['dev']:.2f} | "
          f"{t['tuned']['test']:.2f} | {t['tuned']['all20']:.2f} |", "",
          f"End-to-end (letter in, LLM-extracted reason, agent verdict) vs `gold_verdict` on all "
          f"{e['n_cases']} cases: **agreement {e['n_agree']}/{e['n_cases']} "
          f"({pct(e['verdict_agreement'])})**, abstain rate {pct(e['abstain_rate'])}, gold clause "
          f"in top-3 {pct(e['gold_clause_in_top3_rate'])}. ROC-AUC of the match probability for "
          f"SUPPORTED vs NOT_SUPPORTED: {t['prob_roc_auc_supported_vs_not_all20']:.3f}. "
          f"Baseline: always answering SUPPORTED agrees on {pct(base)} of cases "
          f"({max_gold}/{e['n_cases']}); compare the agreement above with this. "
          "Thresholds were tuned on 10 of these 20 cases, so the all-20 number is optimistic. "
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
    if not args.summary:
        run_matcher.main()
        run_retrieval.main()
        if args.retune or not (RESULTS / "tau.json").exists():
            tune_tau.main()  # then update configs/app.yaml tau_high/tau_low by hand
        run_e2e.main()
        run_faithfulness.main()
    (RESULTS / "summary.md").write_text(build_summary(), encoding="utf-8")
    print("wrote", RESULTS / "summary.md")


if __name__ == "__main__":
    main()
