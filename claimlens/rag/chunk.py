"""Split policy page text into clause-sized chunks that keep section_path and page.

Pure functions, no PDF or model dependency. Size is counted in words (a rough proxy for
tokens; 100-300 words is about 130-400 model tokens).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from claimlens.schemas import Chunk

# dotted numbers only: "4.2 Permanent Exclusions", "2.5.1 Secure Benefit". Bare "12. Foo" lines
# are list items in these policies, not headings.
_NUM_HEAD = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){1,3})\.?\s+(?=[A-Z])(.{2,90})$")
# a bare number on its own line ("2.5"), whose title is on the next line
_NUM_ONLY = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){1,3})\.?$")
# "(ii) ...", "(12) ...", "(a) ..." sub-clauses
_SUB_ITEM = re.compile(r"^\(([ivxlc]{1,5}|\d{1,2}|[a-z])\)\s*(.*)$", re.IGNORECASE)
_SECTION_HEAD = re.compile(r"^(SECTION|PART|ANNEXURE|SCHEDULE)\b.{0,80}$")  # upper-case only
_PAGE_NO = re.compile(r"^\d{1,3}$")
_SENT_SPLIT = re.compile(r"(?<=[.;:])\s+")


@dataclass
class _Block:
    path: list[str]
    page: int
    words: list[str]


def words_in(text: str) -> int:
    return len(text.split())


def clean_page_lines(pages: list[str]) -> list[list[str]]:
    """Strip blank lines, bare page numbers and headers/footers that repeat across pages."""
    raw = [[ln.strip() for ln in p.splitlines() if ln.strip()] for p in pages]
    counts: Counter[str] = Counter()
    for lines in raw:
        counts.update(set(lines))
    limit = max(3, int(0.3 * len(pages)))
    repeated = {ln for ln, c in counts.items() if c >= limit and len(ln) > 3}
    out = []
    for lines in raw:
        out.append([ln for ln in lines if ln not in repeated and not _PAGE_NO.match(ln)])
    return out


def _depth(num: str) -> int:
    return num.count(".") + 1


def _build_blocks(pages: list[list[str]]) -> list[_Block]:
    blocks: list[_Block] = []
    stack: list[tuple[int, str]] = []  # (depth, label)
    sub: str | None = None  # current "(ii)" label under the stack
    cur: _Block | None = None

    def path() -> list[str]:
        p = [lbl for _, lbl in stack]
        if sub:
            p.append(sub)
        return p

    def start(page: int, first: str = "") -> None:
        nonlocal cur
        cur = _Block(path(), page, first.split())
        blocks.append(cur)

    for pno, lines in enumerate(pages, start=1):
        i = 0
        while i < len(lines):
            ln = lines[i]
            i += 1
            num: str | None = None
            title = ""
            m = _NUM_HEAD.match(ln)
            if m:
                num, title = m.group(1).rstrip("."), m.group(2)
            elif _NUM_ONLY.match(ln) and i < len(lines):
                num, title = _NUM_ONLY.match(ln).group(1), lines[i]  # type: ignore[union-attr]
                i += 1
            if num is not None:
                d = _depth(num)
                while stack and stack[-1][0] >= d:
                    stack.pop()
                stack.append((d, f"{num} {title.strip(' :.')[:60]}"))
                sub = None
                start(pno, title)
                continue
            if _SECTION_HEAD.match(ln):
                stack.clear()
                stack.append((0, ln[:70]))
                sub = None
                start(pno, ln)
                continue
            s = _SUB_ITEM.match(ln)
            if s and stack:
                sub = f"({s.group(1)})"
                start(pno, s.group(2))
                continue
            if cur is None:
                start(pno)
            assert cur is not None
            cur.words.extend(ln.split())
    return [b for b in blocks if b.words]


def _split_long(text: str, max_words: int) -> list[str]:
    parts: list[str] = []
    cur: list[str] = []
    n = 0
    for sent in _SENT_SPLIT.split(text):
        w = words_in(sent)
        if cur and n + w > max_words:
            parts.append(" ".join(cur))
            cur, n = [], 0
        # a single sentence longer than max_words is hard-split on words
        while w > max_words:
            ws = sent.split()
            parts.append(" ".join(ws[:max_words]))
            sent = " ".join(ws[max_words:])
            w = words_in(sent)
        cur.append(sent)
        n += w
    if cur:
        parts.append(" ".join(cur))
    return parts


def chunk_pages(
    doc_id: str, pages: list[str], min_words: int = 100, max_words: int = 300
) -> list[Chunk]:
    """Chunk page texts along clause numbers/headings into min..max word chunks."""
    blocks = _build_blocks(clean_page_lines(pages))
    # 1) split oversize blocks, 2) pack small neighbouring blocks up to max_words
    units: list[tuple[str, int, str]] = []  # (text, page, section_path)
    for b in blocks:
        text = " ".join(b.words)
        sp = " > ".join(b.path) or "(preamble)"
        for part in _split_long(text, max_words) if words_in(text) > max_words else [text]:
            units.append((part, b.page, sp))

    chunks: list[Chunk] = []
    buf: list[tuple[str, int, str]] = []  # (text, page, section_path)

    def flush() -> None:
        nonlocal buf
        if buf:
            # page of the first unit; section_path of the largest unit (not a bare heading stub)
            path = max(buf, key=lambda u: words_in(u[0]))[2]
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}-{len(chunks):04d}",
                    doc_id=doc_id,
                    text=" ".join(u[0] for u in buf),
                    section_path=path,
                    page=buf[0][1],
                )
            )
            buf = []

    for text, page, sp in units:
        n_buf = sum(words_in(u[0]) for u in buf)
        if buf and (n_buf >= min_words or n_buf + words_in(text) > max_words):
            flush()
        buf.append((text, page, sp))
    flush()
    return chunks
