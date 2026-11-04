# PROGRESS

## P0 (done 2026-11-04 ~22:20 IST)
- Private repo https://github.com/sandeepklr12345-gif/claim_lens created; initial commit `fbcd8cd` pushed and verified (local == `origin/main`).
- venv `C:\venvs\claim_lens` (Python 3.11.9); `torch 2.11.0+cu128`, `cuda.is_available() == True` (GPU is an RTX 4050 Laptop, not 4060).
- Ollama 0.40.14 running; `qwen2.5:7b` pulled and answers a prompt.
- Acceptance: PASS.

## M0 (2026-11-04, started ~22:20 IST, about 10 min behind target)
**Objective:** 2 policy PDFs parsed and chunked; hybrid retrieval; CI.

**Decision (SPEC 11 open item "which 2 policies"):** Care Health *Care Supreme* and HDFC ERGO *Optima Secure*, both public wordings (see `data/policies/SOURCES.md`). Star Health's own domain and Niva Bupa's site refused scripted downloads (redirect / HTTP 406), so those were not used and no workaround was attempted. Both PDFs are third-party mirrors (policybazaar static host); verify against the insurer sites before the report.

**Files:** `claimlens/schemas.py`, `claimlens/rag/{chunk,parse,retriever,search}.py`, `claimlens/utils/config.py`, `configs/app.yaml`, `tests/test_{chunk,schemas,retriever}.py`, `.github/workflows/ci.yml`, `requirements*.txt`, `pytest.ini`, `ruff.toml`, `data/policies/*`.

**Chunking:** headings = dotted numbers (`3.2.14`) and upper-case `SECTION/PART/ANNEXURE/SCHEDULE` lines; sub-items `(ii)` appended to the path. Bare `12. Foo` lines are treated as list items (they are exclusion lists in these PDFs). Size is counted in words (100-300), not model tokens. Result: care_supreme 124 chunks, hdfc_optima_secure 96 chunks.

**Known issues:** Care Supreme PDF has letter-spaced words in places ("t h e l i s t e d"); a few chunks are under 100 words (heading stubs at section ends). The cross-encoder re-rank (top 10 -> top 3) arrives with the matcher in M1; the M0 search CLI returns the fused top-k.

**Tests:** `ruff check .` pass; `pytest -q -m "not model"` 12 passed; `pytest -q -m model` 1 passed (bge-small dense + hybrid on HDFC).
**Search check:** `python -m claimlens.rag.search "waiting period" --doc care_supreme` returns the Named Ailment waiting-period clause first, with section path and page.
