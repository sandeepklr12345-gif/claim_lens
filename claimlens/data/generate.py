"""Generate synthetic cases and labeled pairs with Ollama (SPEC.md section 6).

Everything written here is labeled source="synthetic". Run:
    python -m claimlens.data.generate            (writes data/pairs.jsonl, data/cases/*.json)

NOTE: gold labels come from construction, not from a human: a SUPPORTED case is written
from a restrictive clause (exclusion, waiting period, limit); a NOT_SUPPORTED case is a
wrongful rejection written against a benefit clause that does cover the treatment. The
chunker must not change after this runs, because chunk ids are stored in the data.
"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path

from claimlens.llm.client import LLM, OllamaClient
from claimlens.rag.parse import list_policies, load_chunks
from claimlens.rag.retriever import HybridRetriever
from claimlens.schemas import Case, Chunk, Pair, Verdict
from claimlens.utils.config import ROOT

log = logging.getLogger(__name__)
SEED = 42

RESTRICTIVE = re.compile(
    r"\b(excluded|not covered|shall not|waiting period|not be liable|not payable|"
    r"sub-?limit|will not be|exclusion|not admissible|limited to|only after|not eligible|"
    r"disallow\w*|deductible)\b",
    re.IGNORECASE,
)
BENEFIT = re.compile(r"\b(shall indemnify|will indemnify|will pay|shall pay)\b", re.IGNORECASE)
SKIP_PATH = re.compile(r"definition|annexure|schedule of|contact|grievance|ombudsman", re.I)

SYSTEM = (
    "You write fictional, synthetic health-insurance documents for a research dataset. "
    "Never use real people, real claim numbers or real addresses. Reply with JSON only."
)

REASON_PROMPT = """Policy clause:
\"\"\"{clause}\"\"\"

Write ONE short reason (max 30 words) that an Indian health insurer could give to reject a
claim because of this clause. Describe the patient's situation in plain words. Do not quote
the clause, and do not mention clause numbers.
Reply as JSON: {{"reason": "..."}}"""

SUPPORTED_PROMPT = """Policy clause:
\"\"\"{clause}\"\"\"

Write a fictional claim rejection letter (80-120 words) from an insurer to a policyholder
called "Mr. A. Kumar" (reference CLM-{ref}). The rejection must be justified by this clause,
but the letter must not quote the clause or mention clause numbers.
Reply as JSON: {{"reason": "<the rejection reason in one sentence, max 30 words>",
"letter_text": "<the full letter>"}}"""

NOT_SUPPORTED_PROMPT = """Policy clause (this benefit IS covered by the policy):
\"\"\"{clause}\"\"\"

Write a fictional claim rejection letter (80-120 words) from an insurer to a policyholder
called "Ms. B. Rao" (reference CLM-{ref}) that WRONGLY rejects a claim for the treatment this
clause covers, claiming it is not covered. Do not quote the clause or mention clause numbers.
Reply as JSON: {{"reason": "<the rejection reason in one sentence, max 30 words>",
"letter_text": "<the full letter>"}}"""


def _words(c: Chunk) -> int:
    return len(c.text.split())


def select_restrictive(chunks: list[Chunk]) -> list[Chunk]:
    return [c for c in chunks if 40 <= _words(c) <= 250 and RESTRICTIVE.search(c.text)
            and not SKIP_PATH.search(c.section_path)]


def select_benefit(chunks: list[Chunk]) -> list[Chunk]:
    return [c for c in chunks if 50 <= _words(c) <= 250 and BENEFIT.search(c.text)
            and not RESTRICTIVE.search(c.text) and not SKIP_PATH.search(c.section_path)]


def hard_negative(pos: Chunk, retriever: HybridRetriever, reason: str) -> Chunk | None:
    """A lexically similar clause from a different section (not the gold clause)."""
    for r in retriever.retrieve(reason, k=15, mode="bm25"):
        c = r.chunk
        if c.chunk_id != pos.chunk_id and c.section_path != pos.section_path:
            return c
    return None


def _ask(llm: LLM, prompt: str, keys: tuple[str, ...]) -> dict[str, str] | None:
    try:
        out = llm.generate_json(prompt, SYSTEM)
    except Exception as e:  # noqa: BLE001 - log and skip this item
        log.warning("generation failed: %s", e)
        return None
    if all(isinstance(out.get(k), str) and out[k].strip() for k in keys):
        return {k: out[k].strip() for k in keys}
    log.warning("missing keys %s in %s", keys, list(out))
    return None


def generate(llm: LLM, out_root: Path = ROOT / "data", n_pairs: int = 40,
             n_sup: int = 12, n_not: int = 8) -> None:
    rng = random.Random(SEED)
    docs = list_policies()
    chunks_by_doc = {d: load_chunks(d) for d in docs}
    retrievers = {d: HybridRetriever(c, None) for d, c in chunks_by_doc.items()}

    restrictive = [c for d in docs for c in select_restrictive(chunks_by_doc[d])]
    benefit = [c for d in docs for c in select_benefit(chunks_by_doc[d])]
    rng.shuffle(restrictive)
    rng.shuffle(benefit)
    log.info("candidates: %d restrictive, %d benefit", len(restrictive), len(benefit))
    pair_src = restrictive[:n_pairs]
    case_src = restrictive[n_pairs:n_pairs + n_sup + 4]  # spare for failures
    ns_src = benefit[:n_not + 3]

    pairs: list[Pair] = []
    for c in pair_src:
        got = _ask(llm, REASON_PROMPT.format(clause=c.text), ("reason",))
        if not got:
            continue
        neg = hard_negative(c, retrievers[c.doc_id], got["reason"])
        if neg is None:
            continue
        n = len(pairs) // 2 + 1
        for chunk, label, tag in ((c, 1, "pos"), (neg, 0, "neg")):
            pairs.append(Pair(pair_id=f"pair_{n:03d}_{tag}", policy_id=c.doc_id,
                              reason=got["reason"], clause_id=chunk.chunk_id,
                              clause_text=chunk.text, label=label))
        log.info("pair %d done", n)

    cases: list[Case] = []

    def make_cases(src: list[Chunk], prompt: str, verdict: Verdict, want: int) -> None:
        made = 0
        for c in src:
            if made >= want:
                break
            got = _ask(llm, prompt.format(clause=c.text, ref=f"{rng.randint(10000, 99999)}"),
                       ("reason", "letter_text"))
            if not got:
                continue
            cases.append(Case(case_id=f"case_{len(cases) + 1:03d}", policy_id=c.doc_id,
                              letter_text=got["letter_text"], reason=got["reason"],
                              gold_clause_ids=[c.chunk_id], gold_verdict=verdict))
            made += 1
            log.info("case %d (%s) done", len(cases), verdict.value)

    make_cases(case_src, SUPPORTED_PROMPT, Verdict.SUPPORTED, n_sup)
    make_cases(ns_src, NOT_SUPPORTED_PROMPT, Verdict.NOT_SUPPORTED, n_not)

    (out_root / "cases").mkdir(parents=True, exist_ok=True)
    with (out_root / "pairs.jsonl").open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(p.model_dump_json() + "\n")
    for case in cases:
        (out_root / "cases" / f"{case.case_id}.json").write_text(
            json.dumps(case.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8")
    log.info("wrote %d pairs and %d cases", len(pairs), len(cases))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    generate(OllamaClient())


if __name__ == "__main__":
    main()
