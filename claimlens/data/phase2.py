"""Phase 2 data (SPEC.md 13.2): splits, data/pairs_v2.jsonl, 30 cases, verification.

    python -m claimlens.data.phase2 generate [--test-policy <doc_id>]
    python -m claimlens.data.phase2 apply-verify
    python -m claimlens.data.phase2 report

Splits are fixed before any text is generated, and a hard negative is drawn only from its
positive's split, so no clause can appear in two splits. Without --test-policy the test set
is whole section groups: this is the SPEC 13.2 FALLBACK for a missing third policy PDF.

Gold clauses of cases are kept out of train (beyond SPEC) so the fine-tuned matcher is never
audited on clauses it was trained on. A case takes the split of its gold clause.

Labels come from construction, not from a human. `verified` is set only by apply_verification,
from what the owner typed in data/verify_pairs.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
from collections import Counter
from pathlib import Path

from claimlens.data.generate import (
    NOT_SUPPORTED_PROMPT,
    SUPPORTED_PROMPT,
    _ask,
    select_benefit,
    select_restrictive,
)
from claimlens.llm.client import LLM, OllamaClient
from claimlens.rag.parse import list_policies, load_chunks
from claimlens.rag.retriever import HybridRetriever
from claimlens.schemas import Case, Chunk, Pair, Verdict
from claimlens.utils.config import ROOT

log = logging.getLogger(__name__)
SEED = 42
TARGET_SUPPORTED = 18  # 30 cases, same 60/40 mix as Phase 1
TARGET_NOT_SUPPORTED = 12
N_VERIFY = 20
SPLITS = ("train", "val", "test")
QA_START = "<!-- verification:start -->"
QA_END = "<!-- verification:end -->"

REASONS_PROMPT = """Policy clause:
\"\"\"{clause}\"\"\"

Write TWO different short reasons (max 30 words each) that an Indian health insurer could give
to reject a claim because of this clause. Each must describe a different patient situation in
plain words. Do not quote the clause, and do not mention clause numbers.
Reply as JSON: {{"reason_1": "...", "reason_2": "..."}}"""


def group_key(chunk: Chunk) -> str:
    """Section group of a clause: its first section below any 'SECTION X.' wrapper."""
    parts = [p.strip() for p in chunk.section_path.split(">")]
    head = parts[1] if parts[0].upper().startswith("SECTION") and len(parts) > 1 else parts[0]
    return f"{chunk.doc_id}::{head}"


def _pick_test_groups(chunks: list[Chunk], positive: set[str], gold: set[str],
                      rng: random.Random, frac: float, tries: int = 500) -> set[str]:
    """Whole groups holding about `frac` of the positives and a fair share of the gold clauses."""
    groups: dict[str, list[Chunk]] = {}
    for c in chunks:
        groups.setdefault(group_key(c), []).append(c)
    names = sorted(groups)
    want_pos, want_gold = frac * len(positive), 0.4 * len(gold)
    best: set[str] = set()
    best_score = float("inf")
    for _ in range(tries):
        rng.shuffle(names)
        pos_n = gold_n = 0
        chosen: set[str] = set()
        for name in names:
            if pos_n >= want_pos:
                break
            chosen.add(name)
            pos_n += sum(c.chunk_id in positive for c in groups[name])
            gold_n += sum(c.chunk_id in gold for c in groups[name])
        score = abs(pos_n - want_pos) + abs(gold_n - want_gold)
        if score < best_score:
            best, best_score = set(chosen), score
    return best


def make_splits(chunks: list[Chunk], positive: set[str], gold: set[str],
                test_policy: str | None = None, seed: int = SEED,
                test_frac: float = 0.2, val_frac: float = 0.2) -> dict[str, str]:
    """Map every clause to train, val or test. Gold clauses never land in train."""
    rng = random.Random(seed)
    if test_policy is not None:
        test = {c.chunk_id for c in chunks if c.doc_id == test_policy}
    else:
        groups = _pick_test_groups(chunks, positive, gold, rng, test_frac)
        test = {c.chunk_id for c in chunks if group_key(c) in groups}
    rest = sorted(c.chunk_id for c in chunks if c.chunk_id not in test)
    val = {i for i in rest if i in gold}
    extra_pos = sorted(i for i in rest if i in positive and i not in val)
    rng.shuffle(extra_pos)
    while extra_pos and len(val & positive) < val_frac * len(positive):
        val.add(extra_pos.pop())
    non_pos = sorted(i for i in rest if i not in positive and i not in val)
    val.update(i for i in non_pos if rng.random() < val_frac)
    return {c.chunk_id: "test" if c.chunk_id in test else "val" if c.chunk_id in val else "train"
            for c in chunks}


def hard_negative_in_split(pos: Chunk, reason: str, retriever: HybridRetriever,
                           splits: dict[str, str]) -> Chunk | None:
    """A BM25-similar clause from another section of the same split."""
    for r in retriever.retrieve(reason, k=len(splits), mode="bm25"):
        c = r.chunk
        if (c.chunk_id != pos.chunk_id and c.section_path != pos.section_path
                and splits[c.chunk_id] == splits[pos.chunk_id]):
            return c
    return None


def generate_pairs(llm: LLM, clauses: list[Chunk], retrievers: dict[str, HybridRetriever],
                   splits: dict[str, str]) -> list[Pair]:
    pairs: list[Pair] = []
    seen: set[str] = set()
    for c in clauses:
        got = _ask(llm, REASONS_PROMPT.format(clause=c.text), ("reason_1", "reason_2"))
        if not got:
            continue
        for reason in got.values():
            neg = hard_negative_in_split(c, reason, retrievers[c.doc_id], splits)
            if neg is None or reason.lower() in seen:
                continue
            seen.add(reason.lower())
            n = len(pairs) // 2 + 1
            for chunk, label, tag in ((c, 1, "pos"), (neg, 0, "neg")):
                pairs.append(Pair(pair_id=f"v2_{n:03d}_{tag}", policy_id=c.doc_id, reason=reason,
                                  clause_id=chunk.chunk_id, clause_text=chunk.text, label=label))
        log.info("clause %s done, %d pairs", c.chunk_id, len(pairs))
    return pairs


def generate_cases(llm: LLM, src: list[Chunk], prompt: str, verdict: Verdict, want: int,
                   start: int, rng: random.Random) -> list[Case]:
    cases: list[Case] = []
    for c in src:
        if len(cases) >= want:
            break
        got = _ask(llm, prompt.format(clause=c.text, ref=f"{rng.randint(10000, 99999)}"),
                   ("reason", "letter_text"))
        if got:
            cases.append(Case(case_id=f"case_{start + len(cases):03d}", policy_id=c.doc_id,
                              letter_text=got["letter_text"], reason=got["reason"],
                              gold_clause_ids=[c.chunk_id], gold_verdict=verdict))
    return cases


def load_pairs(path: Path) -> list[Pair]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Pair.model_validate_json(x) for x in lines if x.strip()]


def write_pairs(pairs: list[Pair], path: Path) -> None:
    text = "".join(p.model_dump_json() + "\n" for p in pairs)
    path.write_text(text, encoding="utf-8")


def load_all_cases(cases_dir: Path) -> list[Case]:
    return [Case.model_validate_json(f.read_text(encoding="utf-8"))
            for f in sorted(cases_dir.glob("*.json"))]


def leakage(pairs: list[Pair], splits: dict[str, str]) -> set[str]:
    """Clause ids reachable from more than one split through the pairs; must be empty."""
    seen: dict[str, set[str]] = {}
    for p in pairs:
        seen.setdefault(p.clause_id, set()).add(splits[p.clause_id])
    # A reason and both of its clauses must also stay together.
    by_reason: dict[str, set[str]] = {}
    for p in pairs:
        by_reason.setdefault(p.reason, set()).add(splits[p.clause_id])
    bad = {i for i, s in seen.items() if len(s) > 1}
    bad |= {p.clause_id for p in pairs if len(by_reason[p.reason]) > 1}
    return bad


def sample_for_verification(pairs: list[Pair], splits: dict[str, str], n: int = N_VERIFY,
                            seed: int = SEED) -> list[Pair]:
    """Balanced sample from the test split (then val), so P2 can score the verified subset."""
    rng = random.Random(seed)
    chosen: list[Pair] = []
    for label in (1, 0):
        pool: list[Pair] = []
        for split in ("test", "val"):
            part = [p for p in pairs if p.label == label and splits[p.clause_id] == split]
            rng.shuffle(part)
            pool += part
        chosen += pool[:n // 2]
    return sorted(chosen, key=lambda p: p.pair_id)


def write_verify_csv(pairs: list[Pair], splits: dict[str, str], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:  # BOM so Excel reads UTF-8
        w = csv.writer(f)
        w.writerow(["pair_id", "split", "reason", "clause_text", "generated_label",
                    "your_label", "note"])
        for p in pairs:
            w.writerow([p.pair_id, splits[p.clause_id], p.reason, p.clause_text, p.label, "", ""])


def apply_verification(pairs: list[Pair],
                       csv_path: Path) -> tuple[list[Pair], list[dict[str, str]]]:
    """Apply only rows the owner filled in. your_label 1 = clause justifies the rejection."""
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["your_label"].strip()]
    bad = [r["pair_id"] for r in rows if r["your_label"].strip() not in ("0", "1")]
    if bad:
        raise ValueError(f"your_label must be 0 or 1, got something else for: {bad}")
    by_id = {r["pair_id"]: r for r in rows}
    missing = set(by_id) - {p.pair_id for p in pairs}
    if missing:
        raise ValueError(f"unknown pair ids: {sorted(missing)}")
    out = []
    for p in pairs:
        row = by_id.get(p.pair_id)
        out.append(p.model_copy(update={"label": int(row["your_label"]), "verified": True})
                   if row else p)
    return out, rows


def verification_table(rows: list[dict[str, str]]) -> str:
    lines = [QA_START, "## Hand verification of pairs_v2 (done by the project owner)", "",
             "| Pair | Generated label | Owner label | Changed | Note |", "|---|---|---|---|---|"]
    for r in rows:
        changed = "yes" if r["generated_label"].strip() != r["your_label"].strip() else "no"
        note = r.get("note", "").replace("|", "/").strip()
        lines.append(f"| {r['pair_id']} | {r['generated_label']} | {r['your_label'].strip()} "
                     f"| {changed} | {note} |")
    return "\n".join(lines + ["", QA_END])


def replace_block(text: str, block: str) -> str:
    pattern = re.compile(re.escape(QA_START) + r".*?" + re.escape(QA_END), re.DOTALL)
    if pattern.search(text):
        return pattern.sub(lambda _: block, text)
    return text.rstrip() + "\n\n" + block + "\n"


def report_text(pairs: list[Pair], splits: dict[str, str], cases: list[Case],
                mode: str) -> str:
    lines = ["# Phase 2 data report", "", f"Split mode: {mode}", "",
             f"Pairs: {len(pairs)}. Verified: {sum(p.verified for p in pairs)}. "
             f"Clause leakage across splits: {len(leakage(pairs, splits))}.", "",
             "| Split | Pairs | Positive | Negative | Distinct clauses | Cases |",
             "|---|---:|---:|---:|---:|---:|"]
    for s in SPLITS:
        part = [p for p in pairs if splits[p.clause_id] == s]
        pos = sum(p.label for p in part)
        n_cases = sum(splits[c.gold_clause_ids[0]] == s for c in cases)
        lines.append(f"| {s} | {len(part)} | {pos} | {len(part) - pos} "
                     f"| {len({p.clause_id for p in part})} | {n_cases} |")
    verdicts = Counter(c.gold_verdict.value for c in cases)
    lines += ["", f"Cases: {len(cases)} ({verdicts['SUPPORTED']} SUPPORTED, "
              f"{verdicts['NOT_SUPPORTED']} NOT_SUPPORTED).", ""]
    return "\n".join(lines)


def generate(llm: LLM, data: Path, test_policy: str | None = None) -> None:
    rng = random.Random(SEED)
    docs = list_policies()
    by_doc = {d: load_chunks(d) for d in docs}
    chunks = [c for d in docs for c in by_doc[d]]
    cases = load_all_cases(data / "cases")
    used = {g for c in cases for g in c.gold_clause_ids}

    restrictive = [c for d in docs for c in select_restrictive(by_doc[d])]
    benefit = [c for d in docs for c in select_benefit(by_doc[d])]
    counts = Counter(c.gold_verdict for c in cases)
    want_sup = max(0, TARGET_SUPPORTED - counts[Verdict.SUPPORTED])
    want_not = max(0, TARGET_NOT_SUPPORTED - counts[Verdict.NOT_SUPPORTED])
    sup_src = [c for c in restrictive if c.chunk_id not in used]
    not_src = [c for c in benefit if c.chunk_id not in used]
    rng.shuffle(sup_src)
    rng.shuffle(not_src)
    sup_src, not_src = sup_src[:want_sup + 1], not_src[:want_not + 1]  # one spare each
    log.info("%d restrictive, %d benefit; reserved %d + %d for new cases", len(restrictive),
             len(benefit), len(sup_src), len(not_src))

    gold = used | {c.chunk_id for c in sup_src + not_src}
    splits = make_splits(chunks, {c.chunk_id for c in restrictive}, gold, test_policy)
    log.info("clauses per split: %s", Counter(splits.values()))

    retrievers = {d: HybridRetriever(by_doc[d], None) for d in docs}
    pairs = generate_pairs(llm, restrictive, retrievers, splits)

    new = generate_cases(llm, sup_src, SUPPORTED_PROMPT, Verdict.SUPPORTED, want_sup,
                         len(cases) + 1, rng)
    new += generate_cases(llm, not_src, NOT_SUPPORTED_PROMPT, Verdict.NOT_SUPPORTED, want_not,
                          len(cases) + len(new) + 1, rng)
    for case in new:
        (data / "cases" / f"{case.case_id}.json").write_text(
            json.dumps(case.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8")

    mode = f"third policy ({test_policy})" if test_policy else (
        "FALLBACK: held-out section groups (no third policy PDF)")
    splits_doc = {"mode": mode, "clauses": splits}
    (data / "splits.json").write_text(json.dumps(splits_doc, indent=1), encoding="utf-8")
    write_pairs(pairs, data / "pairs_v2.jsonl")
    write_verify_csv(sample_for_verification(pairs, splits), splits, data / "verify_pairs.csv")
    log.info("wrote %d pairs, %d new cases", len(pairs), len(new))


def load_splits(data: Path) -> tuple[dict[str, str], str]:
    doc = json.loads((data / "splits.json").read_text(encoding="utf-8"))
    return doc["clauses"], doc["mode"]


def write_report(data: Path, out: Path) -> None:
    splits, mode = load_splits(data)
    text = report_text(load_pairs(data / "pairs_v2.jsonl"), splits,
                       load_all_cases(data / "cases"), mode)
    out.write_text(text, encoding="utf-8")
    log.info("wrote %s", out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["generate", "apply-verify", "report"])
    ap.add_argument("--test-policy", help="doc id of the policy to hold out as the test set")
    args = ap.parse_args()
    data = ROOT / "data"
    if args.command == "generate":
        generate(OllamaClient(), data, args.test_policy)
    elif args.command == "apply-verify":
        pairs, rows = apply_verification(load_pairs(data / "pairs_v2.jsonl"),
                                         data / "verify_pairs.csv")
        write_pairs(pairs, data / "pairs_v2.jsonl")
        qa = data / "QA.md"
        qa.write_text(replace_block(qa.read_text(encoding="utf-8"), verification_table(rows)),
                      encoding="utf-8")
        log.info("applied %d verified rows", len(rows))
    write_report(data, ROOT / "eval" / "results" / "data_v2.md")


if __name__ == "__main__":
    main()
