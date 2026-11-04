"""Attribution Alignment Score (AAS), SPEC.md 7.5. Torch-free.

AAS = positive attribution mass on clause words inside cited spans / total positive mass.
Spans that are not found in the clause (not an exact substring, or not aligned to the
attribution words) contribute zero and are logged. Caveat (SPEC.md 7.5): AAS can be raised
just by quoting the top-attributed words, which is why Phase 2 adds comprehensiveness /
sufficiency and a random-span baseline.
"""

from __future__ import annotations

import logging
import re

from claimlens.schemas import AlignmentResult, Attribution, CitedSpan

log = logging.getLogger(__name__)
_WORD = re.compile(r"\w+|[^\w\s]")
STOP = frozenset(
    "a an the of to in on at by for or and is are was were be been as with that this it its "
    "any from shall will may not no under".split()
)


def words(text: str) -> list[str]:
    """Lower-case word/punctuation tokens, comparable to the merged attribution tokens."""
    return _WORD.findall(text.lower())


def _find(sub: list[str], seq: list[str]) -> int | None:
    n = len(sub)
    if n == 0:
        return None
    for i in range(len(seq) - n + 1):
        if seq[i : i + n] == sub:
            return i
    return None


def span_indices(
    attr: Attribution, spans: list[CitedSpan], clause_text: str
) -> tuple[set[int], list[str]]:
    """Indices of attribution words covered by cited spans (clause spans only) + unmatched."""
    idx: set[int] = set()
    unmatched: list[str] = []
    for sp in spans:
        text = sp.text.strip()
        if sp.source != "clause" or not text or text not in clause_text:
            unmatched.append(sp.text)
            continue
        toks = words(text)
        start = _find(toks, attr.tokens)
        if start is None:
            unmatched.append(sp.text)
            continue
        idx.update(range(start, start + len(toks)))
    if unmatched:
        log.info("AAS: %d unmatched span(s): %s", len(unmatched), unmatched)
    return idx, unmatched


def alignment_score(
    attr: Attribution, spans: list[CitedSpan], clause_text: str
) -> AlignmentResult:
    idx, unmatched = span_indices(attr, spans, clause_text)
    pos = [max(0.0, s) for s in attr.scores]
    total = sum(pos)
    aas = sum(pos[i] for i in idx) / total if total > 0 else 0.0
    return AlignmentResult(aas=min(1.0, aas), cited_idx=sorted(idx), unmatched=unmatched)


def feedback_words(
    attr: Attribution, cited_idx: set[int], k: int = 8, low_frac: float = 0.1
) -> tuple[list[str], list[str]]:
    """(influential words not cited, cited words with little influence) for reflection."""
    pos = [max(0.0, s) for s in attr.scores]
    top = max(pos) if pos else 0.0
    order = sorted(range(len(pos)), key=lambda i: -pos[i])
    missing: list[str] = []
    for i in order:
        w = attr.tokens[i]
        if pos[i] <= 0 or len(missing) >= k:
            break
        if i not in cited_idx and w not in STOP and re.search(r"\w", w) and w not in missing:
            missing.append(w)
    weak: list[str] = []
    for i in sorted(cited_idx):
        w = attr.tokens[i]
        if pos[i] <= low_frac * top and w not in STOP and re.search(r"\w", w) and w not in weak:
            weak.append(w)
    return missing, weak[:k]


def random_span_aas(
    attr: Attribution, span_lengths: list[int], trials: int = 200, seed: int = 42
) -> float:
    """Chance level for AAS: mean AAS of random contiguous spans of the same word lengths.

    Compare the real AAS with this baseline: AAS above chance means the cited spans really
    sit on high-attribution words (SPEC.md 2 'Should', 7.5).
    """
    import random

    pos = [max(0.0, s) for s in attr.scores]
    total, n = sum(pos), len(pos)
    lengths = [k for k in span_lengths if 0 < k <= n]
    if total <= 0 or not lengths:
        return 0.0
    rng = random.Random(seed)
    acc = 0.0
    for _ in range(trials):
        idx: set[int] = set()
        for k in lengths:
            start = rng.randrange(0, n - k + 1)
            idx.update(range(start, start + k))
        acc += sum(pos[i] for i in idx) / total
    return acc / trials
