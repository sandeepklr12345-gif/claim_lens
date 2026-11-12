"""R1a label audit (SPEC.md 14.3): an LLM judge re-labels every pair in two passes.

    python -m claimlens.data.audit judge     # judge all pairs, write label_audit + review_queue
    python -m claimlens.data.audit queue     # rebuild review_queue.csv from label_audit.csv only
    python -m claimlens.data.audit apply-review   # R1b: apply the owner's your_label values

Judge labels are `source: "llm_audit"` and never `verified: true`. The judge and the pair
generator are the same model family (qwen2.5), so their errors may be correlated: agreement
with the generated label is not independent evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from claimlens.data.phase2 import load_pairs, load_splits
from claimlens.llm.client import LLM, OllamaClient
from claimlens.schemas import Pair
from claimlens.utils.config import ROOT

log = logging.getLogger(__name__)

SEED = 42
MAX_FLAGGED = 30  # SPEC 14.3: cap on flagged pairs in the review queue
N_RANDOM = 10  # unflagged pairs added to estimate what the judge misses
MAX_WORDS = 30
CONFIDENCE = ("low", "medium", "high")

AUDIT_COLS = ["pair_id", "generated_label", "judge_label_1", "judge_label_2", "confidence",
              "rationale", "flagged", "policy_id", "clause_id", "split", "source", "verified"]
QUEUE_COLS = ["pair_id", "reason", "clause_text", "generated_label", "judge_label", "rationale",
              "your_label", "note", "policy_id", "clause_id"]

SYSTEM = (
    "You audit labels for a health-insurance claim-rejection dataset (India). You judge whether a "
    "policy clause supports the ground given for rejecting a claim. Judge the reason exactly as "
    "written. Answer with JSON only."
)

RUBRIC = (
    "Rubric. Label 1 ONLY if the clause contains the operative rule the reason relies on, so that "
    "if the facts stated in the reason are true, the rejection is justified. Label 0 if the clause "
    "is on another topic, shares only keywords with the reason, or contradicts the reason or does "
    "not establish its ground.\n"
    'Return JSON: {"label": 0 or 1, "confidence": "high" | "medium" | "low", '
    f'"rationale": "at most {MAX_WORDS} words"}}'
)


class Judgement(BaseModel):
    """One judge pass. label is None when the LLM output could not be used (counts as a flag)."""

    label: int | None
    confidence: str  # high | medium | low
    rationale: str


def build_prompt(reason: str, clause_text: str, swapped: bool) -> str:
    """Pass 1 shows the reason first; pass 2 (swapped) shows the clause first."""
    reason_block = f"REJECTION REASON:\n{reason}"
    clause_block = f"POLICY CLAUSE:\n{clause_text}"
    blocks = [clause_block, reason_block] if swapped else [reason_block, clause_block]
    return "\n\n".join(blocks) + "\n\n" + RUBRIC


def clean_judgement(raw: dict[str, Any]) -> Judgement:
    """Normalise the LLM's JSON; anything unusable becomes label None, confidence low."""
    label = raw.get("label")
    try:
        label = int(label)
        label = label if label in (0, 1) else None
    except (TypeError, ValueError):
        label = None
    conf = str(raw.get("confidence", "")).strip().lower()
    if conf not in CONFIDENCE:
        conf = "low"
    words = str(raw.get("rationale", "")).split()
    return Judgement(label=label, confidence=conf if label is not None else "low",
                     rationale=" ".join(words[:MAX_WORDS]))


def judge_pair(llm: LLM, reason: str, clause_text: str, swapped: bool) -> Judgement:
    try:
        raw = llm.generate_json(build_prompt(reason, clause_text, swapped), SYSTEM)
    except Exception as e:  # noqa: BLE001 - one bad pair must not stop a 400-call run
        log.warning("judge call failed: %s", e)
        return Judgement(label=None, confidence="low", rationale="judge call failed")
    return clean_judgement(raw)


def is_flagged(generated: int, j1: Judgement, j2: Judgement) -> bool:
    """SPEC 14.3: flagged if either pass disagrees with the generated label or is low confidence."""
    for j in (j1, j2):
        if j.label is None or j.label != generated or j.confidence == "low":
            return True
    return False


def combined_confidence(j1: Judgement, j2: Judgement) -> str:
    return min((j1.confidence, j2.confidence), key=CONFIDENCE.index)


def combined_label(j1: Judgement, j2: Judgement) -> int | None:
    """The judge's label when both passes agree, otherwise None (undecided)."""
    return j1.label if j1.label is not None and j1.label == j2.label else None


def audit_row(pair: Pair, generated: int, j1: Judgement, j2: Judgement,
              split: str) -> dict[str, Any]:
    return {
        "pair_id": pair.pair_id, "generated_label": generated,
        "judge_label_1": j1.label, "judge_label_2": j2.label,  # None is written as an empty cell
        "confidence": combined_confidence(j1, j2),
        "rationale": f"{j1.rationale} || {j2.rationale}",
        "flagged": is_flagged(generated, j1, j2), "policy_id": pair.policy_id,
        "clause_id": pair.clause_id, "split": split, "source": "llm_audit", "verified": False,
    }


def generated_labels(data: Path, pairs: list[Pair]) -> dict[str, int]:
    """The label the generator produced. For owner-verified pairs the jsonl label was replaced by
    the owner's, so the original comes from verify_pairs.csv."""
    gen = {p.pair_id: p.label for p in pairs}
    with (data / "verify_pairs.csv").open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            gen[r["pair_id"]] = int(r["generated_label"])
    return gen


def _priority(row: dict[str, Any]) -> tuple[int, int]:
    """Flagged pairs most worth a human look first: both passes contradict the generated label,
    then high confidence."""
    j1, j2 = row["judge_label_1"], row["judge_label_2"]
    both = j1 is not None and j1 == j2 and j1 != row["generated_label"]
    return (0 if both else 1, -CONFIDENCE.index(row["confidence"]))


def build_queue(rows: list[dict[str, Any]], pairs: dict[str, Pair],
                seed: int = SEED) -> list[dict[str, Any]]:
    """Flagged pairs (cap 30) plus 10 random unflagged ones, never an owner-verified pair."""
    free = [r for r in rows if not pairs[r["pair_id"]].verified]
    flagged = sorted((r for r in free if r["flagged"]), key=_priority)[:MAX_FLAGGED]
    unflagged = [r for r in free if not r["flagged"]]
    random.Random(seed).shuffle(unflagged)
    picked = flagged + unflagged[:N_RANDOM]
    out = []
    for r in picked:
        p = pairs[r["pair_id"]]
        agree = r["judge_label_1"] if r["judge_label_1"] == r["judge_label_2"] else None
        out.append({"pair_id": p.pair_id, "reason": p.reason, "clause_text": p.clause_text,
                    "generated_label": r["generated_label"],
                    "judge_label": "" if agree is None else agree,  # blank: passes disagreed
                    "rationale": r["rationale"], "your_label": "", "note": "",
                    "policy_id": p.policy_id, "clause_id": p.clause_id})
    random.Random(seed + 1).shuffle(out)  # hide which rows are the random ones
    return out


def calibration(rows: list[dict[str, Any]], owner: dict[str, int]) -> dict[str, Any]:
    """Judge vs owner-verified labels (n = 20 at most: a sanity check, not a measurement).

    `owner` maps pair_id to the owner's label. Reports the pass-wise agreement, the agreement of
    the combined label (both passes agree; otherwise undecided), and, for comparison, how often
    the generator's own label matched the owner.
    """
    sub = [r for r in rows if r["pair_id"] in owner]

    def hit(key: str) -> int:
        return sum(r[key] == owner[r["pair_id"]] for r in sub)

    both = [r for r in sub if r["judge_label_1"] is not None
            and r["judge_label_1"] == r["judge_label_2"]]
    return {
        "n": len(sub), "pass1_agree": hit("judge_label_1"), "pass2_agree": hit("judge_label_2"),
        "combined_decided": len(both),
        "combined_agree": sum(r["judge_label_1"] == owner[r["pair_id"]] for r in both),
        "generated_agree": hit("generated_label"),
        "flagged": sum(r["flagged"] for r in sub),
        "flagged_among_generator_errors": sum(
            r["flagged"] for r in sub if r["generated_label"] != owner[r["pair_id"]]),
        "generator_errors": sum(r["generated_label"] != owner[r["pair_id"]] for r in sub),
    }


def _int(v: Any) -> int | None:
    return None if v in ("", None) else int(v)


def _write_csv(path: Path, cols: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def read_audit(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["generated_label"] = int(r["generated_label"])
        r["judge_label_1"] = _int(r["judge_label_1"])
        r["judge_label_2"] = _int(r["judge_label_2"])
        r["flagged"] = r["flagged"] == "True"
    return rows


def run_judge(llm: LLM, data: Path, cache: Path) -> list[dict[str, Any]]:
    """Judge every pair twice. Passes are cached in `cache` (JSONL); an interrupted run resumes."""
    pairs = load_pairs(data / "pairs_v2.jsonl")
    gen = generated_labels(data, pairs)
    splits, _ = load_splits(data)
    done: dict[str, dict[str, Any]] = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            done[d["pair_id"]] = d
    rows = []
    with cache.open("a", encoding="utf-8") as out:
        for i, p in enumerate(pairs, 1):
            if p.pair_id not in done:
                j1 = judge_pair(llm, p.reason, p.clause_text, swapped=False)
                j2 = judge_pair(llm, p.reason, p.clause_text, swapped=True)
                done[p.pair_id] = {"pair_id": p.pair_id, "j1": j1.model_dump(),
                                   "j2": j2.model_dump()}
                out.write(json.dumps(done[p.pair_id]) + "\n")
                out.flush()
            d = done[p.pair_id]
            rows.append(audit_row(p, gen[p.pair_id], Judgement(**d["j1"]), Judgement(**d["j2"]),
                                  splits[p.clause_id]))
            if i % 10 == 0:
                log.info("judged %d/%d pairs", i, len(pairs))
    return rows


def write_outputs(rows: list[dict[str, Any]], data: Path) -> None:
    _write_csv(data / "label_audit.csv", AUDIT_COLS, rows)
    pairs = {p.pair_id: p for p in load_pairs(data / "pairs_v2.jsonl")}
    queue = build_queue(read_audit(data / "label_audit.csv"), pairs)
    _write_csv(data / "review_queue.csv", QUEUE_COLS, queue)
    log.info("wrote label_audit.csv (%d rows, %d flagged) and review_queue.csv (%d rows)",
             len(rows), sum(r["flagged"] for r in rows), len(queue))


def backfill_original_labels(pairs: list[Pair], data: Path) -> list[Pair]:
    """Record the generator's label as `original_label` where it is not stored yet. For the
    owner-verified 20 the jsonl label was already replaced, so it comes from verify_pairs.csv."""
    gen = generated_labels(data, pairs)
    return [p if p.original_label is not None else p.model_copy(update={"original_label": gen[
        p.pair_id]}) for p in pairs]


def apply_owner_review(pairs: list[Pair], queue: list[dict[str, str]]
                       ) -> tuple[list[Pair], list[dict[str, str]]]:
    """Apply the owner's `your_label` values: label updated, `verified: true`. Splits are
    untouched (pairs keep their clause_id). Rows with a blank your_label are skipped."""
    by_id = {p.pair_id: p for p in pairs}
    done = []
    for row in queue:
        val = row.get("your_label", "").strip()
        if not val:
            continue
        if val not in ("0", "1"):
            raise ValueError(f"your_label for {row['pair_id']} must be 0 or 1, got {val!r}")
        if row["pair_id"] not in by_id:
            raise ValueError(f"unknown pair id {row['pair_id']}")
        p = by_id[row["pair_id"]]
        by_id[p.pair_id] = p.model_copy(update={
            "label": int(val), "verified": True,
            "original_label": p.original_label if p.original_label is not None else p.label})
        done.append(row)
    return [by_id[p.pair_id] for p in pairs], done


def apply_judge_fallback(pairs: list[Pair], audit: list[dict[str, Any]]
                         ) -> tuple[list[Pair], list[str]]:
    """FALLBACK (SPEC 14.3) when the owner has not returned the queue: for flagged, unverified
    pairs whose two judge passes agree on a label that differs from the current one (and are not
    low confidence), take the judge label, `source: "llm_audit"`, `verified: false`."""
    by_id = {p.pair_id: p for p in pairs}
    changed = []
    for r in audit:
        p = by_id[r["pair_id"]]
        j = r["judge_label_1"]
        if (r["flagged"] and not p.verified and j is not None and j == r["judge_label_2"]
                and j != p.label and r["confidence"] != "low"):
            by_id[p.pair_id] = p.model_copy(update={
                "label": j, "source": "llm_audit", "verified": False,
                "original_label": p.original_label if p.original_label is not None else p.label})
            changed.append(p.pair_id)
    return [by_id[p.pair_id] for p in pairs], changed


def review_table(rows: list[dict[str, str]], pairs: dict[str, Pair]) -> str:
    """Markdown table of the owner's queue decisions for data/QA.md."""
    lines = ["| Pair | Generated label | Judge label | Owner label | Changed | Note |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        old = pairs[r["pair_id"]].original_label
        changed = "yes" if str(old) != r["your_label"].strip() else "no"
        note = r.get("note", "").replace("|", "/").strip()
        lines.append(f"| {r['pair_id']} | {old} | {r['judge_label'] or 'split'} "
                     f"| {r['your_label'].strip()} | {changed} | {note} |")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["judge", "queue", "apply-review"])
    ap.add_argument("--judge-fallback", action="store_true",
                    help="apply-review: owner has not returned the queue; use judge labels")
    args = ap.parse_args()
    data = ROOT / "data"
    if args.command == "judge":
        client = OllamaClient()
        problem = client.health()
        if problem:
            raise SystemExit(problem)
        (ROOT / "logs").mkdir(exist_ok=True)
        write_outputs(run_judge(client, data, ROOT / "logs" / "audit_cache.jsonl"), data)
    elif args.command == "queue":
        pairs = {p.pair_id: p for p in load_pairs(data / "pairs_v2.jsonl")}
        rows = read_audit(data / "label_audit.csv")
        _write_csv(data / "review_queue.csv", QUEUE_COLS, build_queue(rows, pairs))
    else:
        from claimlens.data.phase2 import write_pairs

        pairs = backfill_original_labels(load_pairs(data / "pairs_v2.jsonl"), data)
        if args.judge_fallback:
            pairs, changed = apply_judge_fallback(pairs, read_audit(data / "label_audit.csv"))
            log.info("judge fallback changed %d labels (llm_audit, not verified): %s",
                     len(changed), changed)
        else:
            with (data / "review_queue.csv").open(encoding="utf-8-sig", newline="") as f:
                queue = list(csv.DictReader(f))
            pairs, done = apply_owner_review(pairs, queue)
            log.info("applied %d owner labels from review_queue.csv", len(done))
            qa = data / "QA.md"
            block = ("<!-- review:start -->\n## Review-queue decisions (R1a/R1b, owner)\n\n"
                     + review_table(done, {p.pair_id: p for p in pairs})
                     + "\n\n<!-- review:end -->")
            text = qa.read_text(encoding="utf-8")
            pat = re.compile(r"<!-- review:start -->.*?<!-- review:end -->", re.DOTALL)
            qa.write_text(pat.sub(lambda _: block, text) if pat.search(text)
                          else text.rstrip() + "\n\n" + block + "\n", encoding="utf-8")
        write_pairs(pairs, data / "pairs_v2.jsonl")


if __name__ == "__main__":
    main()
