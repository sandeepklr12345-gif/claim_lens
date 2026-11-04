"""Faithfulness: AAS initial vs final per case and averaged, plus a random-span baseline.

Reads the per-case results written by eval.run_e2e. The core research question (SPEC.md 1):
does checking the explanation against IG attributions and revising it raise AAS?
"""

from __future__ import annotations

import numpy as np

from claimlens.explain.aas import alignment_score, random_span_aas, words
from claimlens.schemas import CaseResult
from eval.common import RESULTS, load_cases, write_json


def main() -> None:
    rows = []
    for c in load_cases():
        f = RESULTS / "e2e" / f"{c.case_id}.json"
        if not f.exists():
            continue
        r = CaseResult.model_validate_json(f.read_text(encoding="utf-8"))
        if r.attribution is None or r.explanation_final is None:
            continue
        lens_final = [len(words(s.text)) for s in r.explanation_final.cited_spans
                      if s.source == "clause"]
        rows.append({
            "case_id": c.case_id,
            "aas_initial": r.aas_initial, "aas_final": r.aas_final,
            "revisions_run": max(0, len(r.aas_history) - 1),
            "history": r.aas_history,
            "random_span_aas": random_span_aas(r.attribution, lens_final),
            "n_unmatched_final_spans": len(alignment_score(
                r.attribution, r.explanation_final.cited_spans,
                r.top_clauses[0].chunk.text).unmatched),
        })
    a0 = np.array([x["aas_initial"] for x in rows])
    a1 = np.array([x["aas_final"] for x in rows])
    rnd = np.array([x["random_span_aas"] for x in rows])
    triggered = np.array([x["revisions_run"] > 0 for x in rows])
    out = {
        "n_cases": len(rows),
        "mean_aas_initial": float(a0.mean()), "mean_aas_final": float(a1.mean()),
        "mean_random_span_aas": float(rnd.mean()),
        "n_improved": int((a1 > a0 + 1e-9).sum()), "n_unchanged": int((abs(a1 - a0) <= 1e-9).sum()),
        "n_worse": int((a1 < a0 - 1e-9).sum()),
        "n_reflection_triggered": int(triggered.sum()),
        "mean_aas_initial_triggered": float(a0[triggered].mean()) if triggered.any() else None,
        "mean_aas_final_triggered": float(a1[triggered].mean()) if triggered.any() else None,
        "caveat": "AAS can be raised by quoting top-attributed words; compare with the "
                  "random-span baseline. Small synthetic set (20 cases); no significance test.",
        "source": "synthetic", "per_case": rows,
    }
    write_json("faithfulness.json", out)
    print({k: v for k, v in out.items() if k != "per_case"})


if __name__ == "__main__":
    main()
