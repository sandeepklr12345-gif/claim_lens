"""R3 evaluation of the low-confidence re-plan (SPEC.md 14.5) -> eval/results/replan_v3.md.

    python -m eval.run_replan

Needs tau_retr in configs/app.yaml (from `python -m eval.tune_tau_retr`) and the cached 30-case
run in eval/results/e2e/ produced WITHOUT the re-plan rule (run `python -m eval.run_e2e` first).
Cases whose first retrieval is confident follow the identical code path as the cached run, so
their cached result is reused; only cases that re-plan (or the no-clause cases) are rerun.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from claimlens.agent.graph import run_case
from claimlens.agent.replan import replan_retrieve, retrieval_decision, top_prob
from claimlens.agent.tools import DirectTools
from claimlens.llm.client import OllamaClient
from claimlens.schemas import CaseResult, Verdict
from claimlens.utils.config import load_config
from eval.common import RESULTS, load_cases, write_json
from eval.tune_tau_retr import collect, load_noclause_cases

log = logging.getLogger(__name__)
E2E = RESULTS / "e2e"


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({n / d:.0%})" if d else "n/a"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    tau = cfg["agent"].get("tau_retr")
    if tau is None:
        raise SystemExit("tau_retr is not set in configs/app.yaml: run eval.tune_tau_retr first")
    llm, tools = OllamaClient(), DirectTools()
    cases = {c.case_id: c for c in [*load_cases(), *load_noclause_cases()]}
    rows = collect()
    base_cfg = {**cfg, "agent": {**cfg["agent"], "tau_retr": None}}  # rule off = the old agent
    out_rows: list[dict[str, Any]] = []
    for r in rows:
        case = cases[r["case_id"]]
        first = retrieval_decision(r["top_prob"], tau, False)
        row = {**r, "first_decision": first, "replanned": first == "replan",
               "after_top3": r["top3"], "after_top_prob": r["top_prob"],
               "after_gold_in_top3": r["gold_in_top3"], "final_decision": first,
               "queries": []}
        if first == "replan":
            clauses, queries = replan_retrieve(
                tools, llm, r["reason"], case.policy_id, cfg["rag"]["top_k_fused"],
                cfg["rag"]["top_k_return"])
            ids = [h.chunk.chunk_id for h in clauses]
            row.update(queries=queries, after_top3=ids, after_top_prob=top_prob(clauses),
                       after_gold_in_top3=bool(set(ids) & set(case.gold_clause_ids)),
                       final_decision=retrieval_decision(top_prob(clauses), tau, True))
        # verdicts: before = the agent without the rule, after = the agent with it
        is_nc = r["split"] == "noclause"
        cached = E2E / f"{r['case_id']}.json"
        if not is_nc and cached.exists():
            before = CaseResult.model_validate_json(cached.read_text(encoding="utf-8")).verdict
        else:
            before = run_case(case, tools, llm, base_cfg).verdict
        if row["final_decision"] == "abstain":
            after = Verdict.INSUFFICIENT_EVIDENCE
        elif first == "ok" and not is_nc:
            after = before  # same code path as the cached run
        else:
            after = run_case(case, tools, llm, cfg).verdict
        row.update(verdict_before=before.value, verdict_after=after.value)
        out_rows.append(row)
        log.info("%s %s -> %s (%s -> %s)", r["case_id"], first, row["final_decision"],
                 before.value, after.value)

    real = [r for r in out_rows if r["split"] != "noclause"]
    nc = [r for r in out_rows if r["split"] == "noclause"]
    summary: dict[str, Any] = {"tau_retr": tau, "source": "synthetic", "per_case": out_rows}
    L = ["# Re-plan on low retrieval confidence (R3)", "",
         f"`tau_retr` = {tau:.3f} (match probability of the best re-ranked clause), calibrated on "
         "validation cases only (`eval/results/tau_retr.json`). A case below it gets one re-plan "
         "(two LLM-written alternative queries, reciprocal-rank-fused with the original search, "
         "re-ranked by the matcher against the original reason); still below it, the agent "
         f"returns INSUFFICIENT_EVIDENCE with `abstain_reason: \"no clause above confidence "
         "threshold\"`. Synthetic data, 30 cases + 5 no-clause cases; small samples.", "",
         "## Gold clause in the top 3, before and after the re-plan", "",
         "| Split | Cases | Re-planned | Before | After | Rescued | Lost |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for split in ("val", "test"):
        part = [r for r in real if r["split"] == split]
        b = sum(r["gold_in_top3"] for r in part)
        a = sum(r["after_gold_in_top3"] for r in part)
        rescued = sum((not r["gold_in_top3"]) and r["after_gold_in_top3"] for r in part)
        lost = sum(r["gold_in_top3"] and not r["after_gold_in_top3"] for r in part)
        L.append(f"| {split} | {len(part)} | {sum(r['replanned'] for r in part)} | "
                 f"{pct(b, len(part))} | {pct(a, len(part))} | {rescued} | {lost} |")
        summary[f"top3_{split}"] = {"n": len(part), "before": b, "after": a,
                                    "rescued": rescued, "lost": lost}
    n_ab_nc = sum(r["final_decision"] == "abstain" for r in nc)
    v_before = sum(r["verdict_before"] == "INSUFFICIENT_EVIDENCE" for r in nc)
    v_after = sum(r["verdict_after"] == "INSUFFICIENT_EVIDENCE" for r in nc)
    L += ["", "## Abstention on the 5 no-clause cases", "",
          "Reasons with no basis in the policy (`data/cases_noclause/`, synthetic, gold verdict "
          "INSUFFICIENT_EVIDENCE by construction).", "",
          "| Measure | Value |", "|---|---:|",
          f"| Abstained by the re-plan rule | {pct(n_ab_nc, len(nc))} |",
          f"| Verdict INSUFFICIENT_EVIDENCE without the rule | {pct(v_before, len(nc))} |",
          f"| Verdict INSUFFICIENT_EVIDENCE with the rule | {pct(v_after, len(nc))} |", ""]
    summary["noclause"] = {"n": len(nc), "abstained_by_rule": n_ab_nc,
                           "insufficient_before": v_before, "insufficient_after": v_after}
    agree_b = sum(r["verdict_before"] == r["gold_verdict"] for r in real)
    agree_a = sum(r["verdict_after"] == r["gold_verdict"] for r in real)
    changed = [r for r in real if r["verdict_before"] != r["verdict_after"]]
    false_abstain = sum(r["final_decision"] == "abstain" for r in real)
    L += ["## Regression check on the 30 cases", "",
          f"Verdict agreement with the gold verdict: {pct(agree_b, len(real))} without the rule, "
          f"{pct(agree_a, len(real))} with it. Cases whose verdict changed: {len(changed)}. "
          f"Real cases that abstained on the retrieval rule: {false_abstain}.", ""]
    if changed:
        L += ["| Case | Gold | Before | After | Final decision |", "|---|---|---|---|---|"]
        L += [f"| {r['case_id']} | {r['gold_verdict']} | {r['verdict_before']} | "
              f"{r['verdict_after']} | {r['final_decision']} |" for r in changed]
        L.append("")
    dec = Counter(r["first_decision"] for r in out_rows)
    L += [f"First-check decisions over all {len(out_rows)} cases: "
          + ", ".join(f"{k} {v}" for k, v in sorted(dec.items())) + ".", "",
          "Only validation cases were used to set `tau_retr`; the test rows above are reported "
          "once. The cached 30-case run is reused for cases whose first retrieval was "
          "confident (same code path), and rerun for cases that re-plan.", ""]
    summary["verdict_agreement"] = {"before": agree_b, "after": agree_a, "n": len(real),
                                    "changed": len(changed)}
    (RESULTS / "replan_v3.md").write_text("\n".join(L), encoding="utf-8")
    write_json("replan_v3.json", summary)
    log.info("wrote replan_v3.md")


if __name__ == "__main__":
    main()
