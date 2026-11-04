"""PDF text extraction (PyMuPDF) with a cleaned-.txt fallback (SPEC.md section 10)."""

from __future__ import annotations

import logging
from pathlib import Path

from claimlens.rag.chunk import chunk_pages
from claimlens.schemas import Chunk
from claimlens.utils.config import ROOT, load_config

log = logging.getLogger(__name__)
POLICY_DIR = ROOT / "data" / "policies"
FORM_FEED = "\f"


def extract_pages(path: Path) -> list[str]:
    """Return one string per page. A .txt file uses form feeds as page breaks."""
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8").split(FORM_FEED)
    import pymupdf  # imported lazily so unit tests on plain text need no PDF library

    with pymupdf.open(path) as doc:
        return [page.get_text() for page in doc]


def policy_path(doc_id: str) -> Path:
    """Prefer a cleaned .txt beside the PDF if one exists, else the PDF."""
    txt, pdf = POLICY_DIR / f"{doc_id}.txt", POLICY_DIR / f"{doc_id}.pdf"
    if txt.exists():
        return txt
    if pdf.exists():
        return pdf
    raise FileNotFoundError(f"no policy named {doc_id!r} in {POLICY_DIR}")


def list_policies() -> list[str]:
    return sorted({p.stem for p in POLICY_DIR.glob("*") if p.suffix in {".pdf", ".txt"}})


def load_chunks(doc_id: str) -> list[Chunk]:
    cfg = load_config()["rag"]
    pages = extract_pages(policy_path(doc_id))
    chunks = chunk_pages(doc_id, pages, cfg["chunk_min_words"], cfg["chunk_max_words"])
    log.info("%s: %d pages -> %d chunks", doc_id, len(pages), len(chunks))
    return chunks
