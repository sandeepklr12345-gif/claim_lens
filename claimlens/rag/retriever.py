"""Hybrid retrieval: BM25 + dense embeddings fused with reciprocal rank fusion (SPEC.md 7.2).

`Retriever` is a Protocol so the NumPy store can later be swapped for Chroma/Qdrant
(SPEC.md 12.1). The embedder is injected, so unit tests use a fake and need no torch.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from claimlens.schemas import Chunk, RetrievedClause
from claimlens.utils.config import ROOT

log = logging.getLogger(__name__)
CACHE_DIR = ROOT / ".cache"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge-small query prompt

# embed(texts, is_query) -> (n, d) L2-normalised float32 array
Embedder = Callable[[Sequence[str], bool], np.ndarray]


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 10) -> list[RetrievedClause]: ...


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal rank fusion over index rankings (best first)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return scores


def make_st_embedder(model_name: str, device: str | None = None) -> Embedder:
    """sentence-transformers embedder (needs torch; local only, not used in CI)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)

    def embed(texts: Sequence[str], is_query: bool) -> np.ndarray:
        items = [QUERY_PREFIX + t for t in texts] if is_query else list(texts)
        out = model.encode(items, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(out, dtype=np.float32)

    return embed


class HybridRetriever:
    """BM25 + dense (cosine on a NumPy matrix) fused by RRF. `embed=None` means BM25 only."""

    def __init__(
        self,
        chunks: list[Chunk],
        embed: Embedder | None = None,
        rrf_k: int = 60,
        cache_path: Path | None = None,
    ) -> None:
        self.chunks = chunks
        self.embed = embed
        self.rrf_k = rrf_k
        self.bm25 = BM25Okapi([tokenize(c.text + " " + c.section_path) for c in chunks])
        self.matrix: np.ndarray | None = None
        if embed is not None:
            self.matrix = self._load_or_embed(cache_path)

    def _load_or_embed(self, cache_path: Path | None) -> np.ndarray:
        assert self.embed is not None
        if cache_path is not None and cache_path.exists():
            m = np.load(cache_path)
            if m.shape[0] == len(self.chunks):
                return m
        m = self.embed([c.text for c in self.chunks], False)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, m)
        return m

    def _bm25_ranking(self, query: str) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return [int(i) for i in np.argsort(-scores) if scores[i] > 0]

    def _dense_ranking(self, query: str) -> list[int]:
        assert self.embed is not None and self.matrix is not None
        q = self.embed([query], True)[0]
        return [int(i) for i in np.argsort(-(self.matrix @ q))]

    def retrieve(self, query: str, k: int = 10, mode: str = "hybrid") -> list[RetrievedClause]:
        """mode: 'hybrid' (default), 'bm25' or 'dense'. Hybrid falls back to BM25 without embed."""
        bm = self._bm25_ranking(query) if mode in ("hybrid", "bm25") else []
        dn = self._dense_ranking(query) if mode in ("hybrid", "dense") and self.embed else []
        rankings = [r for r in (bm, dn) if r]
        fused = rrf_fuse(rankings, self.rrf_k)
        bm_rank = {i: r for r, i in enumerate(bm, 1)}
        dn_rank = {i: r for r, i in enumerate(dn, 1)}
        top = sorted(fused, key=lambda i: -fused[i])[:k]
        return [
            RetrievedClause(
                chunk=self.chunks[i],
                score=fused[i],
                bm25_rank=bm_rank.get(i),
                dense_rank=dn_rank.get(i),
            )
            for i in top
        ]


def build_retriever(doc_id: str, dense: bool = True) -> HybridRetriever:
    """Build the retriever for one policy from configs/app.yaml (needs torch if dense)."""
    from claimlens.rag.parse import load_chunks
    from claimlens.utils.config import load_config

    cfg = load_config()["rag"]
    chunks = load_chunks(doc_id)
    embed = None
    cache = None
    if dense:
        embed = make_st_embedder(cfg["embed_model"])
        safe = re.sub(r"[^A-Za-z0-9]+", "_", cfg["embed_model"])
        cache = CACHE_DIR / f"emb_{doc_id}_{safe}.npy"
    return HybridRetriever(chunks, embed, cfg["rrf_k"], cache)
