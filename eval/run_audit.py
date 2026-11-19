"""R1a: report on the LLM-judge label audit (reads data/label_audit.csv, no LLM calls).

    python -m eval.run_audit      # writes eval/results/audit_v3.md

The judge itself is run with `python -m claimlens.data.audit judge` (slow, local Ollama).
"""

from __future__ import annotations

import csv
import logging
from collections import Counter
from pathlib import Path

from claimlens.data.audit import calibration, read_audit
from claimlens.data.phase2 import load_pairs
from claimlens.utils.config import ROOT

log = logging.getLogger(__name__)


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({n / d:.0%})" if d else "n/a"


def report(data: Path) -> str:
    rows = read_audit(data / "label_audit.csv")
    pairs = {p.pair_id: p for p in load_pairs(data / "pairs_v2.jsonl")}
    with (data / "review_queue.csv").open(encoding="utf-8-sig", newline="") as f:
        queue = list(csv.DictReader(f))
    flagged_ids = {r["pair_id"] for r in rows if r["flagged"]}
    n = len(rows)
    failed = sum(r["judge_label_1"] is None or r["judge_label_2"] is None for r in rows)
    pass_agree = sum(r["judge_label_1"] == r["judge_label_2"] for r in rows)
    both_contradict = sum(
        r["judge_label_1"] is not None and r["judge_label_1"] == r["judge_label_2"]
        and r["judge_label_1"] != r["generated_label"] for r in rows)

    lines = ["# Label audit (R1a)", "",
             "An LLM judge (local Ollama `qwen2.5:7b`) re-labelled every pair twice (second pass "
             "with the reason and clause order swapped). Judge labels are "
             "`source: \"llm_audit\"`, never "
             "`verified`. The judge and the pair generator are the same model family, so their "
             "errors may be correlated; agreement with the generated label is not independent "
             "evidence.", "",
             f"Pairs judged: {n}. Judge failures (unusable output): {failed}.", "",
             "## Flag rates", "",
             "A pair is flagged if either pass disagrees with the generated label or reports low "
             "confidence.", "",
             f"- Flagged overall: {pct(len(flagged_ids), n)}",
             f"- Both passes agree with each other: {pct(pass_agree, n)}",
             f"- Both passes agree with each other and contradict the generated label: "
             f"{pct(both_contradict, n)}", "",
             "| Group | Pairs | Flagged |", "|---|---:|---:|"]
    for key in ("split", "policy_id", "generated_label"):
        for val in sorted({str(r[key]) for r in rows}):
            part = [r for r in rows if str(r[key]) == val]
            nflag = sum(r["flagged"] for r in part)
            lines.append(f"| {key} = {val} | {len(part)} | {pct(nflag, len(part))} |")

    owner = {p.pair_id: p.label for p in pairs.values() if p.verified}
    c = calibration(rows, owner)
    lines += ["", "## Calibration against the owner-verified pairs", "",
              f"n = {c['n']}. With so few pairs this is only a sanity check, not an accuracy "
              "estimate. The 20 pairs are all from the test split and were chosen for "
              "verification, not at random. The owner labels came from an AI-assisted review "
              "that the owner confirmed.", "",
              "| Measure | Agreement with owner label |", "|---|---:|",
              f"| Judge pass 1 | {pct(c['pass1_agree'], c['n'])} |",
              f"| Judge pass 2 (swapped order) | {pct(c['pass2_agree'], c['n'])} |",
              f"| Judge combined (both passes agree; others undecided) | "
              f"{pct(c['combined_agree'], c['combined_decided'])} decided |",
              f"| Generator's own label (for comparison) | "
              f"{pct(c['generated_agree'], c['n'])} |", "",
              f"Of {c['generator_errors']} generated labels the owner overturned, the audit "
              f"flagged "
              f"{c['flagged_among_generator_errors']}. It flagged {c['flagged']} of the {c['n']} "
              "verified pairs in total.", "",
              "## Review queue for the owner", "",
              f"`data/review_queue.csv`: {len(queue)} rows (cap 40) = flagged pairs (cap 30) "
              "plus 10 "
              "random unflagged pairs. Owner-verified pairs are excluded. The random rows are not "
              "marked in the file; they are the rows whose `pair_id` is not flagged in "
              "`data/label_audit.csv`. Their error rate estimates how many errors the judge "
              "misses.",
              f"- Flagged pairs available to review (excluding verified): "
              f"{sum(1 for i in flagged_ids if not pairs[i].verified)}",
              f"- Queue rows that are flagged: {sum(q['pair_id'] in flagged_ids for q in queue)}",
              f"- Queue rows that are random unflagged: "
              f"{sum(q['pair_id'] not in flagged_ids for q in queue)}",
              f"- Queue rows where the two passes disagreed (blank judge_label): "
              f"{sum(q['judge_label'] == '' for q in queue)}", "",
              "## Judge confidence", "",
              "| Confidence (lower of the two passes) | Pairs |", "|---|---:|"]
    conf = Counter(r["confidence"] for r in rows)
    lines += [f"| {k} | {conf.get(k, 0)} |" for k in ("high", "medium", "low")]
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = ROOT / "eval" / "results" / "audit_v3.md"
    out.write_text(report(ROOT / "data"), encoding="utf-8")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
