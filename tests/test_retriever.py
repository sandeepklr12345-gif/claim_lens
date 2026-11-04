import numpy as np
import pytest

from claimlens.rag.chunk import chunk_pages
from claimlens.rag.parse import POLICY_DIR, extract_pages
from claimlens.rag.retriever import HybridRetriever, rrf_fuse, tokenize

PAGES = [
    "4.1 Waiting Periods\nPre-existing diseases are covered after 36 months of cover.\n"
    "4.2 Room Rent\nRoom rent is limited to 1 percent of the sum insured per day.\n"
    "4.3 Maternity\nMaternity expenses are covered after a waiting period of 24 months.\n"
    "4.4 Dental\nDental treatment is excluded unless it follows an accident.\n"
]


def fake_embed(texts, is_query):
    """Deterministic bag-of-words hashing embedder (no model)."""
    out = np.zeros((len(texts), 64), dtype=np.float32)
    for r, t in enumerate(texts):
        for w in tokenize(t):
            out[r, hash(w) % 64] += 1.0
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norm, 1e-9)


def test_rrf_prefers_items_ranked_high_in_both():
    scores = rrf_fuse([[1, 2, 3], [2, 1, 4]], k=60)
    assert set(scores) == {1, 2, 3, 4}
    assert scores[1] == pytest.approx(scores[2])
    assert scores[1] > scores[3] > scores[4] or scores[1] > scores[4]


def test_bm25_and_hybrid_find_known_clause():
    chunks = chunk_pages("t", PAGES, min_words=1, max_words=300)
    for embed in (None, fake_embed):
        r = HybridRetriever(chunks, embed)
        top = r.retrieve("dental treatment excluded", k=3)
        assert "Dental" in top[0].chunk.text
        assert top[0].bm25_rank == 1


def test_bm25_only_has_no_dense_rank():
    chunks = chunk_pages("t", PAGES, min_words=1, max_words=300)
    top = HybridRetriever(chunks, None).retrieve("maternity", k=2)
    assert all(t.dense_rank is None for t in top)


@pytest.mark.parametrize("doc_id", ["care_supreme", "hdfc_optima_secure"])
def test_real_policy_bm25_waiting_period(doc_id):
    pages = extract_pages(POLICY_DIR / f"{doc_id}.pdf")
    chunks = chunk_pages(doc_id, pages)
    assert len(chunks) > 50
    top = HybridRetriever(chunks, None).retrieve("waiting period", k=3)
    assert top and all("waiting" in t.chunk.text.lower() or "wait" in t.chunk.section_path.lower()
                       for t in top)
    assert all(t.chunk.section_path for t in top)


@pytest.mark.model
def test_real_policy_dense_hybrid_waiting_period():
    from claimlens.rag.retriever import build_retriever

    top = build_retriever("hdfc_optima_secure").retrieve("waiting period for pre-existing", k=3)
    assert any(t.dense_rank is not None for t in top)
    assert "SECTION C" in top[0].chunk.section_path or "PED" in top[0].chunk.section_path
