"""End-to-end run of the agent on all cases; verdict agreement with gold_verdict (SPEC.md 8).

Resumable: per-case results are cached in eval/results/e2e/ (use --force to rerun).
"""

from __future__ import annotations

import argparse
import logging
import time

from claimlens.agent.graph import run_case
from claimlens.agent.tools import DirectTools
from claimlens.data.phase2 import load_splits
from claimlens.llm.client import OllamaClient
from claimlens.schemas import CaseResult, Verdict
from claimlens.utils.config import ROOT, load_config
from eval.common import RESULTS, load_cases, write_json

OUT = RESULTS / "e2e"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tau-aas", type=float, help="override the reflection threshold")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    cfg, llm, tools = load_config(), OllamaClient(), DirectTools()
    if args.tau_aas is not None:
        cfg["agent"]["tau_aas"] = args.tau_aas
    cases = load_cases()
    splits, _ = load_splits(ROOT / "data")
    rows = []
    for c in cases:
        f = OUT / f"{c.case_id}.json"
        if f.exists() and not args.force:
            res = CaseResult.model_validate_json(f.read_text(encoding="utf-8"))
        else:
            t0 = time.time()
            res = run_case(c, tools, llm, cfg)
            f.write_text(res.model_dump_json(indent=2), encoding="utf-8")
            logging.info("%s %s gold=%s aas %.3f -> %.3f (%.0fs)", c.case_id, res.verdict.value,
                         c.gold_verdict.value, res.aas_initial or 0, res.aas_final or 0,
                         time.time() - t0)
        rows.append({"case_id": c.case_id, "gold": c.gold_verdict.value,
                     "pred": res.verdict.value, "split": splits[c.gold_clause_ids[0]],
                     "gold_clause_in_top3": any(h.chunk.chunk_id in c.gold_clause_ids
                                                for h in res.top_clauses)})
    n = len(rows)
    agree = sum(r["gold"] == r["pred"] for r in rows)
    abstain = sum(r["pred"] == Verdict.INSUFFICIENT_EVIDENCE.value for r in rows)
    confusion: dict[str, dict[str, int]] = {}
    for r in rows:
        confusion.setdefault(r["gold"], {}).setdefault(r["pred"], 0)
        confusion[r["gold"]][r["pred"]] += 1
    out = {"n_cases": n, "verdict_agreement": agree / n, "n_agree": agree,
           "abstain_rate": abstain / n, "gold_clause_in_top3_rate":
           sum(r["gold_clause_in_top3"] for r in rows) / n, "confusion_gold_to_pred": confusion,
           "by_split": {s: {"n": sum(r["split"] == s for r in rows),
                            "n_agree": sum(r["split"] == s and r["gold"] == r["pred"]
                                           for r in rows)} for s in ("val", "test")},
           "tau_high": cfg["agent"]["tau_high"], "tau_low": cfg["agent"]["tau_low"],
           "source": "synthetic", "per_case": rows}
    write_json("e2e.json", out)
    print({k: v for k, v in out.items() if k != "per_case"})


if __name__ == "__main__":
    main()
