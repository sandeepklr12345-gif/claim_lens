"""R3: calibrate tau_retr on VALIDATION cases only (SPEC.md 14.5).

    python -m eval.tune_tau_retr [--force]

For each case: extract the reason, retrieve and re-rank (current matcher in configs/app.yaml),
record the best clause's match probability and whether a gold clause is in the top 3. tau_retr is
the cut that best separates cases whose gold clause is in the top 3 (should pass) from those
whose gold clause is not (should trigger a re-plan): it maximises Youden's J = TPR + TNR - 1 on
validation, ties broken towards the lower value (fewer re-plans). Test cases and the five
no-clause cases are collected here too (for run_replan) but never used to choose the value.
Writes eval/results/retrieval_v3.json and eval/results/tau_retr.json.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np

from claimlens.agent.graph import extract_reason, retrieve_rerank
from claimlens.agent.replan import top_prob
from claimlens.agent.tools import DirectTools
from claimlens.data.phase2 import load_splits
from claimlens.llm.client import OllamaClient
from claimlens.schemas import Case
from claimlens.utils.config import ROOT, load_config
from eval.common import RESULTS, load_cases, read_json, write_json

log = logging.getLogger(__name__)
NOCLAUSE_DIR = ROOT / "data" / "cases_noclause"
RETRIEVAL = "retrieval_v3.json"


def load_noclause_cases(directory: Path = NOCLAUSE_DIR) -> list[Case]:
    return [Case.model_validate_json(f.read_text(encoding="utf-8-sig"))
            for f in sorted(directory.glob("*.json"))]


def collect(force: bool = False) -> list[dict[str, Any]]:
    cfg = load_config()
    model = cfg["matcher"]["model"]
    cached = RESULTS / RETRIEVAL
    if cached.exists() and not force:
        saved = read_json(RETRIEVAL)
        if saved.get("matcher_model") == model:
            return saved["rows"]
    llm, tools = OllamaClient(), DirectTools()
    splits, _ = load_splits(ROOT / "data")
    rows = []
    for c in [*load_cases(), *load_noclause_cases()]:
        reason = extract_reason(llm, c.letter_text)
        top = retrieve_rerank(tools, reason, c.policy_id, cfg["rag"]["top_k_fused"],
                              cfg["rag"]["top_k_return"])
        ids = [h.chunk.chunk_id for h in top]
        rows.append({
            "case_id": c.case_id, "policy_id": c.policy_id, "reason": reason,
            "gold_clause_ids": c.gold_clause_ids, "gold_verdict": c.gold_verdict.value,
            "split": splits[c.gold_clause_ids[0]] if c.gold_clause_ids else "noclause",
            "top3": ids, "top_prob": top_prob(top),
            "gold_in_top3": bool(set(ids) & set(c.gold_clause_ids))})
        log.info("%s top_prob=%.3f gold_in_top3=%s", c.case_id, rows[-1]["top_prob"] or -1,
                 rows[-1]["gold_in_top3"])
    write_json(RETRIEVAL, {"matcher_model": model, "rows": rows})
    return rows


def pick_tau(probs: np.ndarray, ok: np.ndarray) -> dict[str, Any]:
    """Cut maximising Youden's J for 'gold in top 3' (ok) versus not; None if one class only."""
    if ok.all() or not ok.any():
        return {"tau_retr": None, "reason": "validation has only one class", "j": None}
    best_j, best_tau = -2.0, None
    order = np.unique(probs)
    cuts = [float(order[0]) - 1e-6] + [float((a + b) / 2) for a, b in zip(order[:-1], order[1:],
                                                                         strict=True)]
    for tau in cuts:
        pass_ = probs >= tau
        j = float(pass_[ok].mean() + (~pass_[~ok]).mean() - 1)
        if j > best_j + 1e-12:  # strict: ties keep the lower cut
            best_j, best_tau = j, tau
    return {"tau_retr": best_tau, "j": best_j, "reason": "max Youden J on validation"}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    rows = collect(ap.parse_args().force)
    val = [r for r in rows if r["split"] == "val"]
    probs = np.array([r["top_prob"] for r in val])
    ok = np.array([r["gold_in_top3"] for r in val])
    out = {"n_val": len(val), "n_val_gold_in_top3": int(ok.sum()), **pick_tau(probs, ok),
           "source": "validation cases only", "note": "tau_retr compares the best re-ranked "
           "clause's matcher probability"}
    write_json("tau_retr.json", out)
    print(out)


if __name__ == "__main__":
    main()
