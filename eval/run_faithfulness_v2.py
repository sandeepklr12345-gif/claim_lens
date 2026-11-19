"""Attribution methods compared -> eval/results/faithfulness_v2.{md,json} (SPEC.md 13.4).

    python -m eval.run_faithfulness_v2            [--limit 10]
    python -m eval.run_faithfulness_v2 --render

Every case is explained on (gold reason, gold clause), so no LLM runs and retrieval errors do
not enter. Comprehensiveness and sufficiency use the checkpoint's logit, at k = 10, 20, 50
words. The random baseline averages RANDOM_DRAWS random orderings per case.
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from claimlens.data.phase2 import load_all_cases
from claimlens.explain.faithfulness import KS, faithfulness, spearman
from claimlens.explain.ig import attribute
from claimlens.matcher.scorer import Matcher
from claimlens.rag.parse import list_policies, load_chunks
from claimlens.utils.config import ROOT
from eval.common import RESULTS, read_json, write_json

log = logging.getLogger(__name__)
SEED = 42
RANDOM_DRAWS = 20
METHODS = ("ig", "shap", "rollout")
NAMES = {"ig": "Integrated Gradients", "shap": "SHAP", "rollout": "Attention rollout",
         "random": "Random words (baseline)"}


def run(limit: int | None) -> dict:
    matcher = Matcher()
    clauses = {c.chunk_id: c for d in list_policies() for c in load_chunks(d)}
    cases = load_all_cases(ROOT / "data" / "cases")[:limit]
    rng = np.random.default_rng(SEED)
    rows = []
    seconds = dict.fromkeys(METHODS, 0.0)
    for case in cases:
        clause = clauses[case.gold_clause_ids[0]]

        def score(texts: list[str], reason: str = case.reason) -> np.ndarray:
            return matcher.logits([(reason, t) for t in texts])

        attrs = {}
        for m in METHODS:
            start = time.time()
            attrs[m] = attribute(matcher, case.reason, clause.text, clause.chunk_id, method=m)
            seconds[m] += time.time() - start
        zero = attribute(matcher, case.reason, clause.text, clause.chunk_id, baseline="zero")
        tokens = attrs["ig"].tokens
        row = {"case_id": case.case_id, "n_words": len(tokens),
               "spearman_ig_shap": spearman(attrs["ig"].scores, attrs["shap"].scores),
               "spearman_ig_pad_zero": spearman(attrs["ig"].scores, zero.scores),
               "metrics": {m: faithfulness(score, tokens, attrs[m].scores) for m in METHODS}}
        draws = [faithfulness(score, tokens, rng.random(len(tokens)).tolist())
                 for _ in range(RANDOM_DRAWS)]
        row["metrics"]["random"] = {k: tuple(np.mean([d[k] for d in draws], axis=0).tolist())
                                    for k in KS}
        rows.append(row)
        log.info("%s done (%d words)", case.case_id, len(tokens))
    return {"n_cases": len(rows), "model": matcher.name.rsplit("\\", 1)[-1].rsplit("/", 1)[-1],
            "seconds_total": {k: round(v, 1) for k, v in seconds.items()}, "per_case": rows}


def _mean_std(values: list[float | None]) -> tuple[float, float, int]:
    kept = [v for v in values if v is not None]
    return float(np.mean(kept)), float(np.std(kept)), len(kept)


def render(res: dict) -> str:
    rows = res["per_case"]
    L = ["# Attribution faithfulness (Phase 2)", "",
         f"Model `{res['model']}`, {res['n_cases']} synthetic cases, each on its gold reason and "
         "gold clause. Change in logit after deleting words; higher comprehensiveness is better, "
         "lower sufficiency is better. No significance test.", "",
         "| Method | " + " | ".join(f"Compr. k={k}" for k in KS) + " | "
         + " | ".join(f"Suff. k={k}" for k in KS) + " |",
         "|---|" + "---:|" * (2 * len(KS))]
    for m in (*METHODS, "random"):
        comp = [np.mean([r["metrics"][m][str(k)][0] for r in rows]) for k in KS]
        suff = [np.mean([r["metrics"][m][str(k)][1] for r in rows]) for k in KS]
        cells = " | ".join(f"{v:.3f}" for v in (*comp, *suff))
        L.append(f"| {NAMES[m]} | {cells} |")
    mean_shap, sd_shap, n_shap = _mean_std([r["spearman_ig_shap"] for r in rows])
    mean_zero, sd_zero, n_zero = _mean_std([r["spearman_ig_pad_zero"] for r in rows])
    L += ["", "## Agreement between methods (word-level Spearman, per case)", "",
          "| Comparison | Mean | SD | Cases |", "|---|---:|---:|---:|",
          f"| IG vs SHAP | {mean_shap:.3f} | {sd_shap:.3f} | {n_shap} |",
          f"| IG pad baseline vs zero-embedding baseline | {mean_zero:.3f} | {sd_zero:.3f} "
          f"| {n_zero} |", "",
          "Time over all cases (s): " + ", ".join(
              f"{NAMES[m]} {v}" for m, v in res["seconds_total"].items()) + ".", ""]
    return "\n".join(L)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="first N cases only (SPEC time-box: 10)")
    ap.add_argument("--render", action="store_true", help="only rebuild the table")
    args = ap.parse_args()
    if not args.render:
        write_json("faithfulness_v2.json", run(args.limit))
    res = read_json("faithfulness_v2.json")  # read back so keys match the --render path
    (RESULTS / "faithfulness_v2.md").write_text(render(res), encoding="utf-8")
    log.info("wrote %s", RESULTS / "faithfulness_v2.md")


if __name__ == "__main__":
    main()
