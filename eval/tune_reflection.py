"""Tune the reflection threshold tau_aas -> eval/results/reflection_v2.{md,json} (SPEC.md 13.4).

Reads the per-case e2e results, which must be recorded at the top of the grid:
    python -m eval.run_e2e --force --tau-aas 0.8
    python -m eval.tune_reflection

tau is chosen on validation cases only, then reported once on test. Also writes
docs/reflection_check.md, whose better / same / worse marks are for the project owner.
"""

from __future__ import annotations

import logging

import numpy as np

from claimlens.data.phase2 import load_splits
from claimlens.explain.aas import random_span_aas, words
from claimlens.explain.explain import choose_tau, replay_reflection
from claimlens.schemas import Case, CaseResult, Explanation
from claimlens.utils.config import ROOT
from eval.common import RESULTS, load_cases, write_json

log = logging.getLogger(__name__)
GRID = [0.4, 0.5, 0.6, 0.7, 0.8]
RECORDED_AT = 0.8
N_CHECK = 10


def load_results() -> list[tuple[Case, CaseResult]]:
    out = []
    for c in load_cases():
        f = RESULTS / "e2e" / f"{c.case_id}.json"
        r = CaseResult.model_validate_json(f.read_text(encoding="utf-8"))
        if r.explanation_initial is not None and r.aas_history:
            out.append((c, r))
    return out


def summarise(rows: list[tuple[Case, CaseResult]], tau: float) -> dict:
    replayed = [replay_reflection(r.aas_history, tau) for _, r in rows]
    initial = np.array([r.aas_history[0][1] for _, r in rows])
    final = np.array([f for f, _ in replayed])
    return {"n": len(rows), "mean_aas_initial": float(initial.mean()),
            "mean_aas_final": float(final.mean()),
            "n_improved": int((final > initial + 1e-9).sum()),
            "mean_revisions": float(np.mean([n for _, n in replayed]))}


def chance_level(rows: list[tuple[Case, CaseResult]]) -> float:
    """Mean AAS of random spans as long as each case's initial spans."""
    return float(np.mean([random_span_aas(r.attribution, [len(words(s.text)) for s in
                                                          r.explanation_initial.cited_spans])
                          for _, r in rows]))


def _spans(e: Explanation) -> str:
    return "; ".join(f'"{s.text}"' for s in e.cited_spans)


def check_sheet(rows: list[tuple[Case, CaseResult]]) -> str:
    """Each case's highest-AAS revision whose text changed: raised AAS first, then test cases."""
    splits, _ = load_splits(ROOT / "data")
    items = []
    for c, r in rows:
        changed = [(h[1], e) for h, e in zip(r.aas_history[1:], r.revisions, strict=True)
                   if e.text != r.explanation_initial.text]
        if changed:
            aas, revision = max(changed, key=lambda x: x[0])
            items.append((c, r, aas, revision))
    items.sort(key=lambda x: (x[2] <= x[1].aas_history[0][1],
                              splits[x[0].gold_clause_ids[0]] != "test", x[0].case_id))
    chosen = items[:N_CHECK]
    L = ["# Reflection check", "",
         f"{len(chosen)} revised explanations (of {len(items)} cases with a changed text), "
         f"recorded at tau_aas = {RECORDED_AT}. Some revisions did not raise AAS, so the agent "
         "did not keep them; they are here so the check is not limited to its successes.", "",
         "**To be marked by the project owner:** put one x per case. Judge which text explains "
         "the rejection better, not the AAS number.", ""]
    for c, r, aas, revision in chosen:
        initial = r.aas_history[0][1]
        L += [f"## {c.case_id} ({c.gold_verdict.value})", "", f"**Reason:** {r.reason}", "",
              f"**Original (AAS {initial:.3f}):** {r.explanation_initial.text}", "",
              f"Cited: {_spans(r.explanation_initial)}", "",
              f"**Revision (AAS {aas:.3f}):** {revision.text}", "",
              f"Cited: {_spans(revision)}", "",
              f"**AAS change:** {aas - initial:+.3f}", "",
              "Owner mark: [ ] better  [ ] same  [ ] worse", ""]
    return "\n".join(L)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    splits, mode = load_splits(ROOT / "data")
    rows = load_results()
    val = [x for x in rows if splits[x[0].gold_clause_ids[0]] == "val"]
    test = [x for x in rows if splits[x[0].gold_clause_ids[0]] == "test"]
    tau = choose_tau([r.aas_history for _, r in val], GRID)
    grid = {str(t): summarise(val, t) for t in GRID}
    res = {"tau_aas": tau, "objective": "highest mean final AAS on validation; ties go low",
           "split_mode": mode, "validation_grid": grid, "test_at_tau": summarise(test, tau),
           "test_chance_aas": chance_level(test), "source": "synthetic"}
    write_json("reflection_v2.json", res)

    t = res["test_at_tau"]
    L = ["# Reflection threshold (Phase 2)", "",
         f"Chosen on validation only: **tau_aas = {tau}**. Objective: {res['objective']}. Replayed "
         f"from one run recorded at tau_aas = {RECORDED_AT}, so every tau sees the same LLM "
         "outputs. All data is synthetic.", "",
         "| tau (validation) | n | Mean AAS initial | Mean AAS final | Improved | Mean revisions |",
         "|---:|---:|---:|---:|---:|---:|"]
    for k, g in grid.items():
        L.append(f"| {k} | {g['n']} | {g['mean_aas_initial']:.3f} | {g['mean_aas_final']:.3f} "
                 f"| {g['n_improved']} | {g['mean_revisions']:.2f} |")
    L += ["", f"Test, reported once at tau_aas = {tau} (n = {t['n']}): mean AAS "
          f"{t['mean_aas_initial']:.3f} -> {t['mean_aas_final']:.3f}, improved {t['n_improved']}, "
          f"mean revisions {t['mean_revisions']:.2f}. Random spans of the same lengths score "
          f"{res['test_chance_aas']:.3f}. Best of the revisions is kept, so final cannot fall "
          "below initial. No significance test.", ""]
    (RESULTS / "reflection_v2.md").write_text("\n".join(L), encoding="utf-8")
    (ROOT / "docs" / "reflection_check.md").write_text(check_sheet(rows), encoding="utf-8")
    log.info("tau_aas = %s; set it in configs/app.yaml", tau)


if __name__ == "__main__":
    main()
