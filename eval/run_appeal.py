"""R2: evaluate the grounded appeal drafter on the 30 cases (SPEC.md 14.4).

Uses the cached agent results in eval/results/e2e/ (run `python -m eval.run_e2e` first) and
asks the local LLM for a draft for every case that is not SUPPORTED. Drafts are cached in
eval/results/appeal/ so an interrupted run resumes (--force to redraft).

    python -m eval.run_appeal      # writes eval/results/appeal_v3.md and appeal_v3.json
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter

from claimlens.appeal.draft import draft_appeal, quote_is_verbatim
from claimlens.llm.client import OllamaClient
from claimlens.schemas import AppealDraft, CaseResult, Verdict
from eval.common import RESULTS, load_cases, write_json

log = logging.getLogger(__name__)
OUT = RESULTS / "appeal"
E2E = RESULTS / "e2e"


def is_complete(d: AppealDraft, clause_text: str) -> bool:
    """Every section present, at least one quote, and every quote re-verified against the clause."""
    return bool(
        d.facts and d.reasoning.strip() and d.relief_sought.strip() and d.document_checklist
        and d.deadline_note.strip() and d.disclaimer and d.cited_clauses
        and all(quote_is_verbatim(c.quote, clause_text) for c in d.cited_clauses))


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({n / d:.0%})" if d else "n/a"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    llm = OllamaClient()
    rows = []
    for case in load_cases():
        res = CaseResult.model_validate_json((E2E / f"{case.case_id}.json").read_text(
            encoding="utf-8"))
        f = OUT / f"{case.case_id}.json"
        if f.exists() and not args.force:
            draft = AppealDraft.model_validate_json(f.read_text(encoding="utf-8")) if (
                f.read_text(encoding="utf-8").strip() != "null") else None
        else:
            draft = draft_appeal(res, case.letter_text, llm)
            f.write_text(draft.model_dump_json(indent=2) if draft else "null", encoding="utf-8")
            log.info("%s %s -> %s", case.case_id, res.verdict.value,
                     f"{draft.source}/{draft.attempts}" if draft else "no draft")
        clause = res.top_clauses[0].chunk.text if res.top_clauses else ""
        rows.append({"case_id": case.case_id, "verdict": res.verdict.value,
                     "kind": draft.kind if draft else None,
                     "source": draft.source if draft else None,
                     "attempts": draft.attempts if draft else None,
                     "n_quotes": len(draft.cited_clauses) if draft else 0,
                     "complete": bool(draft and is_complete(draft, clause)),
                     "fallback_reason": draft.fallback_reason if draft else None})

    # Stress test: the cached run predicts few non-SUPPORTED verdicts, so also draft an appeal for
    # every case whose GOLD verdict is NOT_SUPPORTED, forcing the verdict (marked, not the agent's).
    stress = []
    for case in (c for c in load_cases() if c.gold_verdict == Verdict.NOT_SUPPORTED):
        res = CaseResult.model_validate_json((E2E / f"{case.case_id}.json").read_text(
            encoding="utf-8"))
        if not res.top_clauses:
            continue
        forced = res.model_copy(update={"verdict": Verdict.NOT_SUPPORTED})
        f = OUT / f"forced_{case.case_id}.json"
        if f.exists() and not args.force:
            d = AppealDraft.model_validate_json(f.read_text(encoding="utf-8"))
        else:
            d = draft_appeal(forced, case.letter_text, llm)
            f.write_text(d.model_dump_json(indent=2), encoding="utf-8")
            log.info("forced %s -> %s/%d", case.case_id, d.source, d.attempts)
        stress.append({"case_id": case.case_id, "source": d.source, "attempts": d.attempts,
                       "complete": is_complete(d, res.top_clauses[0].chunk.text)})
    drafts = [r for r in rows if r["source"]]
    verdicts = Counter(r["verdict"] for r in rows)
    first_try = sum(r["source"] == "llm" and r["attempts"] == 1 for r in drafts)
    retried = sum(r["source"] == "llm" and r["attempts"] == 2 for r in drafts)
    fallback = sum(r["source"] == "template" for r in drafts)
    ns = [r for r in rows if r["verdict"] == Verdict.NOT_SUPPORTED.value]
    out = {"n_cases": len(rows), "verdicts": dict(verdicts), "n_drafts": len(drafts),
           "first_try_pass": first_try, "pass_after_retry": retried, "template_fallback": fallback,
           "not_supported_cases": len(ns), "not_supported_complete": sum(r["complete"] for r in ns),
           "supported_with_draft": sum(r["verdict"] == "SUPPORTED" and r["source"] is not None
                                       for r in rows), "source": "synthetic", "per_case": rows,
           "forced_stress": stress}
    write_json("appeal_v3.json", out)

    md = ["# Appeal drafter (R2)", "",
          "Synthetic cases (30), local Ollama `qwen2.5:7b`. Agent results are the cached "
          "`eval/results/e2e/` run (the 30-case run with the thresholds in `configs/app.yaml`; "
          "delete `eval/results/appeal/` and rerun if the verdict mix changes).", "",
          "Verdicts in the cached run: " + ", ".join(f"{k} {v}" for k, v in sorted(
              verdicts.items())) + ". Drafts are made for NOT_SUPPORTED (appeal) and "
          "INSUFFICIENT_EVIDENCE (clarification request) only.", "",
          "| Measure | Value |", "|---|---:|",
          f"| Drafts produced | {len(drafts)} |",
          f"| Quotes pass on the first try (LLM draft, 1 attempt) | "
          f"{pct(first_try, len(drafts))} |",
          f"| Pass after one retry | {pct(retried, len(drafts))} |",
          f"| Fall back to the template (`source: template`) | {pct(fallback, len(drafts))} |",
          f"| NOT_SUPPORTED cases with a complete draft | "
          f"{pct(out['not_supported_complete'], len(ns))} |",
          f"| SUPPORTED cases that got a draft (should be 0) | {out['supported_with_draft']} |", "",
          "A draft is complete when facts, at least one quote, reasoning, relief, checklist, "
          "deadline note and disclaimer are all present and every quote is re-checked verbatim "
          "against the clause text (whitespace-normalised, case-sensitive). Template fallbacks "
          "are counted as complete only if they carry a verified quote.", "",
          "## Stress test on gold NOT_SUPPORTED cases (verdict forced)", "",
          "With only a few predicted NOT_SUPPORTED cases above, an appeal was also drafted for "
          f"all {len(stress)} cases whose gold verdict is NOT_SUPPORTED, using the cached "
          "retrieval and explanation but forcing the verdict. This tests the drafter, not the "
          "agent's verdicts.", "",
          "| Measure | Value |", "|---|---:|",
          f"| Quotes pass on the first try | "
          f"{pct(sum(s['source'] == 'llm' and s['attempts'] == 1 for s in stress), len(stress))} |",
          f"| Pass after one retry | "
          f"{pct(sum(s['source'] == 'llm' and s['attempts'] == 2 for s in stress), len(stress))} |",
          f"| Fall back to the template | "
          f"{pct(sum(s['source'] == 'template' for s in stress), len(stress))} |",
          f"| Complete draft | {pct(sum(s['complete'] for s in stress), len(stress))} |", "",
          "The check is mechanical: it shows quotes are grounded and no figure is invented, not "
          "that the legal argument is right. No human has rated these drafts.", ""]
    if fallback:
        md += ["Fallback reasons:", ""] + [
            f"- {r['case_id']}: {r['fallback_reason']}" for r in drafts
            if r["source"] == "template"]
        md.append("")
    (RESULTS / "appeal_v3.md").write_text("\n".join(md), encoding="utf-8")
    log.info("wrote appeal_v3.md")


if __name__ == "__main__":
    main()
