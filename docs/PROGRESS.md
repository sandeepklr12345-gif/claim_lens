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

**CI note:** the owner added `.github/workflows/blank.yml` on GitHub (commit `3ea0db4`, same job as SPEC 7.10 plus concurrency, read-only permissions, timeout). It is the project's CI workflow; the duplicate `ci.yml` written by the build was removed. SPEC/CLAUDE.md say `ci.yml`: read that as `blank.yml`. The remote history was also rewritten once before M0 (initial commit amended to `e178315`); not done by the build.

**M0 result (2026-11-04 ~22:33 IST): PASS.** Commits `251ae7e` (M0 code), `72def7b` (CI de-dup), `57e879f` (fixed invalid YAML in `blank.yml`: unquoted `: ` in a step name made GitHub reject the workflow in 0s; first two CI runs were red for that reason). Push verified (`origin/main` == local). CI run 41873438920 on `57e879f`: success (ruff + 12 unit tests).

## M1 (2026-11-04 ~22:48 IST, ahead of the 23:45 target)
**Objective:** 20 synthetic cases + ~80 labeled pairs; matcher scoring; IG attributions.

**Files:** `claimlens/llm/client.py` (OllamaClient, FakeLLM), `claimlens/data/generate.py`, `claimlens/matcher/{scorer,metrics}.py`, `claimlens/explain/{ig,render,wordpieces}.py`, `eval/run_matcher.py`, `eval/run_ig_samples.py`, `data/pairs.jsonl` (80 pairs: 40 pos / 40 hard neg), `data/cases/case_001..020.json` (12 SUPPORTED, 8 NOT_SUPPORTED), `data/QA.md`, `eval/results/{matcher.json,ig_samples.html}`, tests.

**Matcher (pretrained `cross-encoder/ms-marco-MiniLM-L-6-v2`, no fine-tuning), from `eval/results/matcher.json`:** threshold picked on dev half; held-out half (n=40): accuracy 0.575, F1 0.653, ROC-AUC 0.6275. Weak, as expected for an off-the-shelf MS-MARCO model on this task; reported as is.

**IG:** Captum LayerIntegratedGradients on the word-embedding layer, pad baseline with special tokens kept, 50 steps, summed over the embedding dim, L2-normalised over clause tokens, word-pieces merged by summing. 3 sample pairs rendered to `eval/results/ig_samples.html`. Manual look: top words for a "12 months" reason are "12" and "months".

**Tests:** `ruff check .` pass; `pytest -q -m "not model"` 19 passed; `pytest -q -m model` 4 passed.

**Known issues / honesty notes:**
- Gold labels are by construction. Hand check of 5 cases (by Claude, not the owner): 3 sound, 1 questionable, 1 wrong (see `data/QA.md`). SUPPORTED labels are noisier. Owner should hand-check all 20.
- Letters are short (~40 words) and generic; some pair reasons only loosely match their clause.
- The chunker is frozen now: chunk ids in the data depend on it.
- NOT_SUPPORTED gold_clause_ids holds the benefit clause that contradicts the rejection (not an empty list).

**M1 CI fix:** first CI run for `4a15d0b` was red (`ModuleNotFoundError: requests`; the CI requirements are deliberately light). `requests` is now imported lazily inside `OllamaClient`. Verified by running the unit tests in a clean venv with only `requirements-ci.txt` (19 passed) before pushing.

## M2 (2026-11-04 ~23:25 IST, ahead of the 01:45 target)
**Objective:** explanation + reflection with AAS; 2 MCP servers; LangGraph flow.

**Files:** `claimlens/explain/{aas,explain}.py`, `claimlens/agent/{graph,tools,verdict,run}.py`, `claimlens/core.py` (shared tool functions), `claimlens/mcp_servers/{policy_server,explainer_server}.py`, `configs/app.yaml` (agent: tau_high 0.7, tau_low 0.3, tau_aas 0.6, max_rounds 2), `tests/test_{explain,agent}.py`.

**Acceptance (case_001, CLI):** `python -m claimlens.agent.run --case data\cases\case_001.json` prints verdict, initial and final explanation, AAS before and after. PASS. Same result through MCP (`--mcp`): both servers started, all 5 tools called over stdio, identical output (AAS initial 0.133, final 0.133; history 0.133 -> 0.133 -> 0.015, i.e. the 7B model's revisions did not raise AAS on this case). The MCP path took about 2 minutes (two model loads); direct tools are faster and are the default.

**Bugs found and fixed while testing:**
- The LLM wrote `source: "Policy clause care_supreme-0051"`, so every span was treated as a non-clause span and AAS was silently 0. `parse_explanation` now treats anything not mentioning "reason" as a clause span; regression test added.
- `mcp` 2.x removed `FastMCP`; pinned `mcp<2` (installed 1.30.0), as SPEC 5 names FastMCP.
- MCP servers hung on Windows when torch/the embedding model loaded inside a tool call; both servers now warm up (retrievers, matcher, Captum) in the main thread before `mcp.run()`. MCP wiring finished inside the 30-minute time-box, so no direct-call fallback was needed for the agent (DirectTools stays as the default and for tests/UI).
- Retrieval returning nothing crashed the graph with IndexError; it now abstains (INSUFFICIENT_EVIDENCE, no explanation).

**Design notes / deviations:** attributions cover clause words only, so only spans cited from the clause can earn AAS mass (reason-sourced spans count as unmatched, as SPEC 7.5 says unmatched count as zero). `langgraph` added to `requirements-ci.txt` (pure Python) so the FakeLLM end-to-end test runs in CI as SPEC 9 asks. Appeal draft is a template (no LLM), only for NOT_SUPPORTED. Reason extraction relies on the letter; the synthetic letters are short, so extracted reasons can be vague (case_001: "Not covered under your policy.").

**Tests:** `ruff check .` pass; `pytest -q -m "not model"` 30 passed (also in a clean venv with only `requirements-ci.txt` and no torch); `pytest -q -m model` 4 passed.
