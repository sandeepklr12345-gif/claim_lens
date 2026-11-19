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

## M3 (2026-11-04 ~23:45 IST, ahead of the 03:30 target)
**Objective:** Streamlit UI; evaluation on all 20 cases.

**Files:** `app/streamlit_app.py`, `eval/{common,run_retrieval,run_e2e,run_faithfulness,run_all,tune_tau}.py`, `eval/results/{summary.md,retrieval.json,tau.json,match_probs.json,e2e.json,faithfulness.json,e2e/*.json}`, `tests/test_ui.py` (local), `configs/app.yaml` (tau 0.8 / 0.4), `claimlens/explain/aas.py` (random-span baseline), `claimlens/agent/graph.py` (`extract_reason`, `retrieve_rerank` helpers).

**Acceptance:** `python -m eval.run_all` produced `eval/results/summary.md`. PASS. UI: `streamlit run app\streamlit_app.py` serves (health ok, HTTP 200); AppTest loads the page and runs a full case through the button (verdict banner, IG-highlighted clause, initial/final explanation with AAS).

**Results (all synthetic, n=20; see `eval/results/summary.md`):**
- Retrieval Recall@3: BM25 0.85, hybrid 0.80, hybrid+rerank 0.90 (n=20, differences are within noise; no claim that hybrid beats BM25).
- End-to-end verdict agreement 13/20 (65%); the always-SUPPORTED baseline is 12/20 (60%). The agent says SUPPORTED for 17/20 cases. Match probability separates SUPPORTED from NOT_SUPPORTED weakly (ROC-AUC 0.615): the pretrained cross-encoder scores relevance, not support vs contradiction, and NOT_SUPPORTED cases are written against highly relevant benefit clauses.
- Thresholds: SPEC says tune on the first 10 cases, but cases 1-12 are all SUPPORTED, so a stratified 10-case dev split was used (documented in `tune_tau.py`). Tuned 0.8/0.4 improved dev (0.60 -> 0.70) but not held-out (0.60 -> 0.50); all-20 stays 0.60. Adopted as the protocol's result; both reported.
- Faithfulness (core question): mean AAS 0.200 initial -> 0.233 after reflection (improved in 4/20 cases, unchanged in 16, worse in 0 by construction since the best is kept). Random spans of the same size score 0.171, so initial AAS is only slightly above chance and the reflection gain (+0.033) is small. No significance test; the 7B model often returns the same spans when asked to revise.

**Known issues:** end-to-end verdicts differ slightly from `tau.json` because the reason is re-extracted by the LLM each run. A `streamlit run` dev server on port 8599 was left running by the build (kill-by-pattern was denied); stop it with Task Manager or ignore. Comprehensiveness / sufficiency (SPEC "Should") not built.

## M4 (2026-11-04 ~23:55 IST, about 5 hours before the 05:00 hard stop)
**Objective:** README with real numbers, demo steps, final push.

**Files:** `README.md`. Every number in it was checked against `eval/results/` (retrieval.json, matcher.json, e2e.json, faithfulness.json, tau.json, summary.md).

**Acceptance:** a fresh `git clone` of GitHub `main` (commit `e7ce6b0`) ran `python -m claimlens.agent.run --case data\cases\case_001.json` and printed verdict, initial and final explanation, and AAS before and after; `claimlens.rag.search` worked; 31 unit tests and ruff passed in the clone. The clone reused the existing venv (torch CUDA is a 2.7 GB wheel, so a full reinstall was not repeated); `pip install --dry-run -r requirements.txt` resolved cleanly. PASS.

**Open items for the owner (not done by the build):** name and USN in the README header (`[TODO]`); hand-check all 20 case labels (`data/QA.md`); verify the two policy PDFs against the insurers' sites; `MASTER_PROMPT.md` has an uncommitted local edit that the build left alone; a `streamlit run` dev server on port 8599 may still be running.
**Cut / deferred:** comprehensiveness and sufficiency metrics (SPEC "Should"); everything in SPEC section 12 (Phase 2).

## Front-end design (2026-11-05 ~01:30 IST, after M4)
**Objective:** apply the canvas design: Intake screen, the audit-result screen in `app/result_view.py`, live run progress.

**Files:** new `app/{adapter,intake_view,result_view,preview_result}.py`, `.streamlit/config.toml`, `tests/{test_adapter,test_intake_view,test_result_view}.py`. Changed `app/streamlit_app.py` (rewritten), `claimlens/agent/graph.py` and `claimlens/explain/explain.py` (optional progress callbacks), `claimlens/llm/client.py` (`OllamaClient.health`), `tests/{test_ui,test_agent}.py`, `ruff.toml`, `README.md`. No new dependencies; `requirements-ci.txt` unchanged.

**Pipeline state:** the M2 pipeline runs end to end, so the UI calls the real `run_case`. The sample-data fallback was not needed; `preview_result.py` is the only place wireframe values appear and it is labelled.

**Live progress:** `run_case(..., on_progress=...)` streams the same linear graph (`graph.stream`, same nodes and edges; a test checks the result equals a run without the callback). It reports each node, plus attribution and each AAS score from inside `explain_reflect`. The UI runs the audit in a worker thread and redraws `RunProgress` twice a second, so elapsed time ticks during long LLM calls. Timings shown are measured; nothing is estimated.

**Schema gaps (no values invented):**
- STUB (marked in `app/adapter.py`): `Attribution` covers clause words only, so reason words have score 0.0 and the "Reason in the letter" card is unshaded. Smallest fix if wanted: also return the token_type 0 tokens from `claimlens/explain/ig.py` as `reason_tokens` and `reason_scores` on `Attribution`.
- Clause display: tokens are lower-cased and split on punctuation, so `clause_words` rebuilds the clause's own words and sums their tokens' positive mass.
- No per-clause summaries: "Other clauses" shows shortened clause text (an excerpt), and logits go through a sigmoid.
- "What the check changed" is built from `aas_history` and `feedback_words`; the agent stores no prose.
- `appeal_draft` exists only for NOT_SUPPORTED, so the mockup's clarification draft on SUPPORTED is not shown.

**Bugs found and fixed while testing:**
- `extract_reason` hides LLM failures, so a stopped Ollama only surfaced after retrieval and matching. `health()` now checks before the run.
- AAS needs an exact substring but the view matches loosely, so on case_020 a span scoring zero would have been underlined (share 0.035 vs AAS 0.014). `to_view` now tags such spans "unmatched": listed, not underlined.
- `preview_result.py` failed under `streamlit run` (`No module named 'app'`); it now sets `sys.path` as `streamlit_app.py` does.
- Streamlit drops widget state on the result screen, which would have reset the policy select. It is now restored from the selected case; `tests/test_ui.py` covers it.
- In a real browser the panel, spinner and errors sat below the fold, and Streamlit's heading rule overrode Source Serif 4 on `h1`. Both fixed (`!important` on `.cl-serif` and `.cl-hero`).

**Deviations:** `ruff.toml` exempts the three inline-HTML view files from E501 so the supplied markup stays as written. `app/result_view.py` differs from the supplied file by the Google Fonts `@import` (as asked) and that `!important`. `app/preview_result.py` differs by the `sys.path` lines.

**Tests:** `ruff check .` pass; `pytest -q -m "not model"` 63 passed (also in the clean CI venv, no torch or Streamlit); `pytest -q -m model` 8 passed (real Ollama and models, including a full audit through the button and the Ollama-down message).

**Ran in a real browser (headless Edge):** intake, selecting a case, Run audit, every live panel stage in order, result screen, back to intake; no errors in the server log. Real-data check on 3 sample cases and 1 pasted letter: `clause_words` kept 100% of the positive attribution mass and the view's cited share equals AAS on all 4.

**Measured here (8 runs, not evaluation results):** 25-44 s per audit once models are loaded, about 85% of it in Explain and check. The first run in a fresh process adds 39-149 s of model loading. Rerun scores vary (case_013 AAS 0.499 then 0.345), so the local LLM is not repeatable even with a fixed seed.

**Not checked:** phone-width layout; visual judgement beyond the desktop screenshots; the result screen's "After reflection" toggle and policy restore in the browser (both covered by AppTest only); `eval/results` were not regenerated, and the evaluation path through `run_case` now uses `graph.stream`.

**Known issue:** the page fetches fonts from Google Fonts, so "Nothing is sent to an outside service" is true of letters and claim data but not of that font request.

## Phase 0 (2026-11-05 ~20:00 IST)
Tag `phase1-presented` on `e631ec7` pushed and verified. Branch `phase2` created; Phase 2 docs committed as `0d5ad09`. Ollama lists `qwen2.5:7b`. Venv `C:\venvs\claim_lens` has torch 2.11.0+cu128, CUDA available.
**Corrections to CLAUDE.md:** the GPU is an RTX 4050 Laptop, not a 4060. The venv is outside the repo. CI (`blank.yml`) runs only on `main`, so `phase2` CI is started with `gh workflow run blank.yml --ref phase2`.

## P1 Data (2026-11-05 ~20:35 IST, ahead of the Fri 09:00 target; IN PROGRESS, needs the owner)
**Objective:** larger labeled set, 30 cases, leakage-free splits, verification log.

**Files:** new `claimlens/data/phase2.py`, `tests/test_phase2_data.py`, `data/pairs_v2.jsonl`, `data/splits.json`, `data/verify_pairs.csv`, `data/cases/case_021..030.json`, `eval/results/data_v2.md`. Changed `claimlens/schemas.py` (`Pair.verified`).

**Results (`eval/results/data_v2.md`):** 206 pairs, 103 pos / 103 neg (train 114, val 52, test 40); 30 cases (18 SUPPORTED, 12 NOT_SUPPORTED; val 17, test 13); clause leakage 0.

**Decisions:**
- FALLBACK (SPEC 13.2): no third policy PDF yet, so test = whole held-out section groups. `generate --test-policy <id>` rebuilds with a third policy.
- Only 55 restrictive clauses exist, so each gives two reasons; 3 clauses returned one. Positive clauses repeat across pairs inside one split.
- Beyond SPEC: case gold clauses never enter train, so no train-split cases exist. The Phase 1 `pairs.jsonl` is kept but not reused, because its clauses would cross the splits.

**Acceptance:** files exist PASS; no clause leakage PASS (tested on the committed files); label balance reported PASS; verified >= 20 **PENDING** (0 so far; `data/verify_pairs.csv` is ready for the owner). Hand-written cases **NOT DONE**: `data/cases_extra/` (t01 to t06, two realistic letters) is not in the repo, git history or the searched folders, so the 30 cases are all generated.

**Tests:** `ruff check .` pass; `pytest -q -m "not model"` 75 passed. Commit `b9226b6`, push verified. CI run 42050898056 on `phase2`: success.

**Known issues:** labels are by construction; hard negatives may also support the reason. Generated case labels 021-030 are not hand-checked. Phase 1 case_003 (questionable) and case_009 (wrong) keep their labels until the owner decides.

**Fix found in P2:** `tests/test_ui.py` hard-coded 20 cases and failed once cases 021-030 existed. P1 had run only the unit tests, so this was missed; the test now counts the files.

## P2 Matcher (2026-11-05 ~21:20 IST, ahead of the Fri 14:00 target)
**Objective:** fine-tuned cross-encoder and baseline table.

**Files:** new `eval/train_matcher.py`, `eval/run_matcher_v2.py`, `claimlens/matcher/baselines.py`, `tests/test_matcher_baselines.py`, `tests/test_matcher_resolve.py`, `eval/results/matcher_v2.md`, `eval/results/matcher_v2_scores.json`. Changed `claimlens/matcher/scorer.py`, `configs/app.yaml`, `eval/run_matcher.py`, `tests/test_ui.py`.

**Decisions:** the 13.7 model IDs were confirmed on Hugging Face. Plain torch loop, because `accelerate` is not an allowed dependency. FALLBACK: a checkpoint missing from `models/` (git-ignored, so every fresh clone) falls back to the pretrained model with a logged warning. All three bases were trained, and the agent's checkpoint is the best fine-tuned one on validation.

**Negative result (`eval/results/matcher_v2.md`):** the fine-tuned MiniLM does not beat the Phase 1 pretrained model on validation. Fine-tuned legal-bert and InLegalBERT are clearly worse, probably because 114 training pairs are too few for a fresh head. The zero-shot LLM's validation threshold labels almost everything positive, although its test ROC-AUC is the highest. Test has n=40, so small gaps are not reliable.

**Acceptance:** the fine-tuned model runs through `python -m claimlens.agent.run` PASS; table complete PASS; verified subset **PENDING** (0 pairs verified, the table says so and rebuilds with `--render`).

**Tests:** `ruff check .` pass; `pytest -q -m "not model"` 89 passed; `pytest -q -m model` 8 passed. Commit `1604bac`, push verified. CI run 42064951236: success.

## P3 Explainability (2026-11-05 ~22:15 IST, ahead of the Sat 09:00 target)
**Objective:** SHAP, attention rollout, comprehensiveness and sufficiency.

**Files:** new `claimlens/explain/{rollout,shap_text,faithfulness}.py`, `eval/run_faithfulness_v2.py`, `tests/test_faithfulness.py`, `eval/results/faithfulness_v2.{md,json}`. Changed `claimlens/explain/{ig,explain}.py`, `claimlens/matcher/scorer.py` (eager attention), `requirements.txt`, `tests/test_model_ig.py`.

**Results (`eval/results/faithfulness_v2.md`, 30 cases):** every method beats random words; SHAP is the most faithful and rollout the weakest at k=10 and 20. IG vs SHAP agree only weakly, and IG changes a lot with its baseline. AAS is built on IG, so it inherits that sensitivity. SPEC's cut order was not needed: SHAP ran on all 30 cases in 348 s, then 234 s in the final `run_all`, with `max_evals` 2000.

**Method notes:** comprehensiveness and sufficiency use the logit, because fine-tuned probabilities saturate near 1. Words are deleted, and scoring uses detokenised lower-case text on both sides of each difference.

**Tests:** `ruff check .` pass; `pytest -q -m model` 9 passed. Commit `81ddb71`, push verified. CI run 42069216642: success.

## P4 Reflection (2026-11-06 ~00:20 IST, ahead of the Sat 16:00 target)
**Objective:** tuned `tau`, manual check of 10 revised explanations.

**Files:** new `eval/tune_reflection.py`, `eval/results/reflection_v2.{md,json}`, `docs/reflection_check.md`. Changed `claimlens/agent/graph.py`, `claimlens/explain/explain.py`, `claimlens/schemas.py`, `configs/app.yaml`, `eval/{run_all,run_e2e,run_faithfulness,tune_tau}.py`, `tests/test_faithfulness.py`.

**Decisions:**
- One agent run at `tau_aas` 0.8 records each revision path; a lower tau stops earlier on the same path, so `replay_reflection` tests the grid with no extra LLM calls. A test compares it with real `reflect` runs.
- Beyond SPEC: `tau_high` and `tau_low` were tuned on the Phase 1 model, so they were retuned on validation cases. `tau.json` now uses the Phase 2 split and replaces the old dev split; the Phase 1 results stay at tag `phase1-presented`.
- Added `CaseResult.revisions`. The first run kept only the best text and gave 4 explanations for a sheet that needs 10, so the agent was rerun. The sheet includes revisions that did not raise AAS.

**Results (`eval/results/`):** every tau in the grid gives the same mean final AAS, because AAS rarely reaches 0.4, so the tie rule gives `tau_aas` = 0.4: the threshold has no practical effect on this data. Verdicts: tuned thresholds improve validation but not test, where agreement equals always answering SUPPORTED (0.46).

**Repeatability:** two agent runs with the same seed gave different reflection numbers: validation mean final AAS 0.166 and 0.230; test mean AAS 0.182 -> 0.191 (improved 2) and 0.176 -> 0.196 (improved 1). The second run is the reported one, because it is the run that stores revisions; tau was the same both times and nothing was selected on test. The LLM's run-to-run variation is as large as the reflection effect.

**Acceptance:** all result files produced by `python -m eval.run_all` PASS (exit 0). `docs/reflection_check.md` has 10 explanations PASS; owner's better / same / worse marks **PENDING**.

**Tests:** `ruff check .` pass; `pytest -q -m "not model"` 110 passed.

**Commit note:** `claimlens/schemas.py` holds the revisions field and the P5 API models, so it went in with P4. Commit `35ad07e`, push verified.

## P5 Packaging (2026-11-06, ahead of the Sun 12:00 target)
**Objective:** FastAPI, Docker Compose, CI, final README.

**Files:** new `api/__init__.py`, `api/main.py`, `claimlens/utils/storage.py`, `Dockerfile`, `.dockerignore`, `docker-compose.yml`, `tests/test_api.py`. Changed `.github/workflows/blank.yml`, `requirements-ci.txt`, `README.md`, `.gitignore`.

**Corrections to the plan:** SPEC's "existing storage module" did not exist, so `claimlens/utils/storage.py` is new. The CI workflow only runs on push or PR to `main`, so `phase2` runs are started with `gh workflow run`. The GPU here is an RTX 4050 Laptop 6 GB, not a 4060. Containers run the matcher on CPU.

**Run in this session:** `docker compose build` passed (image 3.42 GB). `docker compose up` served the API on 8000 and the UI on 8501 (health 200). A full audit of case_001 through the container and host Ollama returned status done, verdict SUPPORTED and the disclaimer in 130 s, while `run_all` shared Ollama; that run used the local `models/` folder. Coverage: `claimlens` and `api` 67% total, `api/main.py` 92%, `storage.py` 100%.

**Fresh clone (`8fd599c`, no `models/`):** `docker compose up -d --build` served the UI (health 200) and API. A real audit finished with status done in 120 s, and the API log shows `checkpoint models/matcher_v2 not found, using pretrained`. So a fresh clone runs the pretrained matcher, not the fine-tuned one. A first attempt failed the UI health check because it polled before Streamlit had started; the rerun waits and passes. Fine-tuned checkpoint through the agent was run from the local `models/` folder.

**CI:** run 42092831256 (`35ad07e`) success; run 42092965259 (`8fd599c`) success on both jobs, lint and unit tests, and the Docker build.

**Open issue, verification (needs the owner):** at 00:44-00:46 on 2026-11-06, outside this session, `data/verify_pairs.csv` was filled in and `apply-verify` was run. All 20 notes start `CLAUDE (...)`, and 5 labels were flipped. The records then read `verified: true` under "Owner-verified subset", but the files do not show that the owner checked them, and project rules require that. Those changes (`data/pairs_v2.jsonl`, `data/verify_pairs.csv`, `data/QA.md`, `eval/results/data_v2.md`, `eval/results/matcher_v2.md`) were left as they are and **not committed**, so the repo and README stay at 0 verified until the owner decides. If kept, the test scores change: fine-tuned MiniLM test acc/F1/AUC 0.625/0.706/0.772 becomes 0.550/0.625/0.668, and the README must be updated from `eval/results/`.

## Phase 2b (branch `phase2b`)
**T0 = 2026-11-13 02:43 IST** (R1a starts). Targets: R1a 03:28, R2 04:33, R1b 05:23, R3 05:58, R4 06:13 (stop new work 06:13), hard stop 06:43.
**Phase 0:** tag `phase2-3day` on `55588c4` pushed. Owner confirmed the 20 verification labels as-is on 2026-11-13 (AI-assisted review, owner-confirmed notes); applied with `apply-verify`, matcher table re-rendered, committed as `7b3027a`. 6 positive / 14 negative among the 20. Local `main` has one unpushed docs commit (`39ba4be`) diverged from `origin/main`; left alone, work is on `phase2b`.

### R1a code (audit module; outputs follow when the judge run finishes)
Files: `claimlens/data/audit.py` (judge, flag rule, review-queue builder, calibration), `eval/run_audit.py` (report), `tests/test_audit.py` (8 tests: flag rule, output cleaning, swapped prompt order, queue cap/exclusions). Two passes per pair (second with reason/clause order swapped); flagged if either pass disagrees with the generated label or is low confidence. Judge labels are `llm_audit`, never verified. Judge and generator are the same model family. Judge speed about 9 to 15 s per pair on the RTX 4050 (qwen2.5:7b partly on CPU), so the full run finishes after the R1a target (see R1a entry).

## R2 Appeal (code done 2026-11-13 ~03:10 IST)
**Objective:** grounded appeal drafter, appeal MCP server, API and UI hook.
**Files:** new `claimlens/appeal/{__init__,draft,deadlines,templates,tools}.py`, `claimlens/mcp_servers/appeal_server.py`, `eval/run_appeal.py`, `tests/test_appeal.py`, results `eval/results/appeal_v3.{md,json}` and `eval/results/appeal/`. Changed `claimlens/schemas.py` (AppealDraft, CitedClause, DeadlineInfo, `CaseResult.appeal`), `claimlens/agent/graph.py` (draft node), `api/main.py` (`GET /cases/{id}/appeal`), `app/adapter.py`, `app/result_view.py` (sections + download).
**Design:** every quote must be verbatim in the retrieved clause (whitespace-normalised, case-sensitive, at least 3 words); numbers in facts/reasoning/relief must appear in the letter, reason or clause; letter details kept only if present in the letter, else visible `[placeholders]`; one retry that names the failures, then the template fallback (`source: "template"`). INSUFFICIENT_EVIDENCE gets a clarification request; SUPPORTED gets none. `appeal_template` in `verdict.py` is kept as the last-resort fallback if the drafter itself raises.
**Ombudsman rules:** Rule 14(3)(a)-(b) read on 2026-11-13 from the Council for Insurance Ombudsmen's published copy "as amended till 09.11.2023" (cioins.co.in). Matches the logic (representation first, one month, complaint within one year). Amendments after 9 Nov 2023 NOT checked, so every output says to confirm with the Ombudsman office. The Rs 50 lakh cap and IRDAI caps were not checked and are not used. Date arithmetic is unit-tested (month ends, 29 Feb).
**Tests:** 135 passed (`pytest -q -m "not model"`), `ruff check .` clean. The appeal MCP server was started over stdio and its three tools called (list_templates, get_template, compute_deadlines).
**Evaluation (`appeal_v3.md`, synthetic):** cached Phase 2 e2e run predicts 27 SUPPORTED and 3 NOT_SUPPORTED, so only 3 drafts: 3/3 first-try, 3/3 complete. Stress test on all 12 gold NOT_SUPPORTED cases with the verdict forced (labelled as such): 10/12 first try, 2/12 after one retry, 0 template fallbacks, 12/12 complete. Mechanical checks only; no human rated the drafts. Rerun in R4 on the R1b verdict mix.
**Known issue:** the preview wireframe `app/preview_result.py` has no `appeal` key (UI shows the plain draft only there).

### R1a Label audit (done 2026-11-13 03:27 IST; target 03:28)
**Objective:** LLM-judge audit of all 206 pairs (two passes, second with order swapped), review queue for the owner.
**Files:** `claimlens/data/audit.py`, `eval/run_audit.py`, `tests/test_audit.py` (10 tests), `data/label_audit.csv`, `data/review_queue.csv` (40 rows), `eval/results/audit_v3.md`; `Pair` gained `original_label` and `hard_negative` fields; `apply-review` and `--judge-fallback` are implemented for R1b.
**Results (`audit_v3.md`):** 99/206 flagged (48%); 0 judge failures; the two passes agree on 177/206. 84 of 103 generated positives are flagged versus 15 of 103 negatives. Calibration on the owner-verified 20 (n = 20, sanity check only, all from the test split): pass 1 17/20, pass 2 16/20, combined 14/15 on the 15 decided, generator's own label 16/20; the audit flagged all 4 labels the owner overturned (and 9 of 20 verified pairs overall). Judge and generator are the same model family.
**Queue:** 30 flagged (of 90 flagged and unverified; ranked: both passes contradict the generated label first, then confidence) plus 10 random unflagged. Random rows are not marked; they are the rows not flagged in `label_audit.csv`. The other 60 flagged pairs are not in the queue: if the owner returns the queue, only reviewed pairs change in R1b.
**Acceptance:** both files exist, calibration number in `audit_v3.md`, queue 40 rows (cap 40), flagging rule tested: PASS.
**Known issues:** the review noted chunks cut mid-clause (care_supreme-0031, 0038, 0042, 0043, 0064) and a suspect `v2_008_pos`; re-chunking is outside SPEC 14 and was not done. Judge speed was about 9 s per pair, so the run took about 38 minutes.

### Prep for R1b and R3 while the review queue is with the owner (2026-11-13 ~03:50 IST)
Written ahead of the labels because it does not depend on them; nothing here has been run on retrained models yet.
- **Hard negatives (R1b):** `claimlens/data/hardneg.py`, `data/hard_negatives.jsonl` (60, train/val groups) and `data/hard_negatives_test.jsonl` (20, test groups). For each generated-positive reason the top 10 retrieved clauses are filtered to non-gold clauses in the reason's own split (first 5 considered, at most 2 kept per reason); kept only if both judge passes say 0 with high confidence. Deviation from the spec text ("top-5 retrieved"): same-split filtering so no reason text crosses splits. `source: llm_audit`, `verified: false`, `hard_negative: true`. 100 candidate clauses judged; 31 distinct reasons in the train/val file (41 HDFC / 19 Care), test file 19 Care / 1 HDFC. Leakage check including the new files passes.
- **Training / evaluation scripts:** `eval/train_matcher.py` gained `--seed` and `--hard-negatives`; `eval/run_matcher_v3.py` (old vs new labels side by side, three seeds, 1000-resample bootstrap with seed 42, paired difference vs the pretrained baseline, hard-negative test accuracy).
- **R3 code:** `claimlens/agent/replan.py`, nodes `check_retrieval` and `replan` plus a conditional edge in `claimlens/agent/graph.py` (abstain still runs `match`/`verdict`/`draft` as no-ops so progress events stay complete), `plan_trace` and `abstain_reason` on `CaseResult`, `tau_retr` read from `configs/app.yaml` (unset = rule off), `eval/tune_tau_retr.py`, `eval/run_replan.py`, five no-clause cases in `data/cases_noclause/` (synthetic, gold INSUFFICIENT_EVIDENCE by construction; kept outside `data/cases/` so the 30-case runs are unaffected). Tests: `tests/test_replan.py` (10), `tests/test_hardneg.py` (6). One existing assertion changed: `tests/test_agent.py` expected node order now includes `check_retrieval` after `retrieve`. 152+ non-model tests and ruff pass.

## R1b Retrain and re-evaluate (done 2026-11-13 ~04:50 IST; target 05:23)
**Objective:** apply the owner's review, retrain the matcher on cleaned labels plus hard negatives (3 seeds), re-tune thresholds, rerun the 30-case pipeline.
**Labels:** `python -m claimlens.data.audit apply-review` applied 37 of the 40 queue rows (your_label filled); 13 of them differ from the generated label. Those pairs are `verified: true` with `original_label` kept; the 3 blank rows were left as generated. The judge fallback was NOT used, so no `llm_audit` label replaced a generated one. 57 of 206 pairs are verified, 149 are not. Splits are frozen. Old and new labels are shown side by side in `eval/results/matcher_v3.md`; the review table is in `data/QA.md`.
**Training:** `eval.train_matcher --seed {42,43,44} --hard-negatives data/hard_negatives.jsonl` (60 hard negatives, train/val groups). The agent checkpoint is the seed-42 run, fixed in advance (`models/matcher_v3_s42`, git-ignored, set in `configs/app.yaml`).
**Matcher (`matcher_v3.md`, test n = 40, new labels):** v3 accuracy 0.600 / 0.650 / 0.675 for seeds 42 / 43 / 44 (mean 0.642, SD 0.038, 95% bootstrap CI [0.500, 0.775]); AUC mean 0.647. Pretrained baseline accuracy 0.450, Phase 2 matcher 0.475. Paired difference v3 minus pretrained: accuracy +0.192 (CI [+0.050, +0.342], excludes 0), F1 +0.066 (CI includes 0), AUC +0.015 (CI includes 0). AUC does not improve over the Phase 2 model (0.648 vs 0.647). On the 20 hard negatives (test groups, `llm_audit`, not verified) the v3 seeds score 0.65 to 0.70, the pretrained model 0.40 and BM25 0.75: BM25 is better there. The test set is small; read the intervals.
**Thresholds (`tau.json`, validation only, n = 17):** tau_high 0.46, tau_low 0.2 (were 0.96 / 0.82 for the Phase 2 matcher); `tau_aas` unchanged at 0.4 (`tune_reflection`). Put into `configs/app.yaml`.
**End-to-end (`e2e.json`, 30 cases, synthetic):** verdict agreement 19/30 (val 14/17, test 5/13). Test agreement (0.38) is no better than the default 0.7/0.3 thresholds (0.38) and below always-SUPPORTED (0.46). Gold NOT_SUPPORTED cases are mostly missed: 9 of 12 predicted SUPPORTED, 2 correct, 1 INSUFFICIENT_EVIDENCE. Gold clause in the top 3 for 80% of cases; abstain rate 6.7%. Negative result, reported as it is.
**Faithfulness (`faithfulness.json`, all 30 cases; replayed from the e2e run, so not a 10-case subset):** mean AAS 0.180 initial, 0.212 after reflection, random-span baseline 0.145; 8 improved, 22 unchanged, 0 worse; reflection triggered on 28.
**Note:** `eval/results/summary.md` and the other Phase 2 reports were not regenerated here; `eval.run_all` in R4 does that. The Phase 2 e2e run was backed up in `logs/` (git-ignored).

## R3 Re-plan on low retrieval confidence (done 2026-11-13 ~05:15 IST; target 05:58)
**Objective:** one re-plan when the best clause is weak, then abstain (`abstain_reason: "no clause above confidence threshold"`), recorded in `plan_trace`.
**Files:** `claimlens/agent/replan.py`, graph nodes `check_retrieval` / `replan` with a conditional edge, `eval/tune_tau_retr.py`, `eval/run_replan.py`, `data/cases_noclause/` (5 synthetic no-clause cases, gold INSUFFICIENT_EVIDENCE by construction), `tests/test_replan.py`. Results: `tau_retr.json`, `retrieval_v3.json`, `replan_v3.md/json`.
**Calibration (validation only, n = 17, v3 seed-42 matcher):** `tau_retr` = 0.936 (Youden J 0.33; 15 of 17 validation cases have the gold clause in the top 3). The separation is weak.
**Result (negative):** the rule re-plans 25 of 35 cases and rescues none (gold clause in top 3: val 15/17 before and after, test 8/13 before and after). It abstains on all 5 no-clause cases (0/5 without the rule), but also on 20 of the 30 real cases, and verdict agreement falls from 19/30 to 7/30. The best-clause probability overlaps between the two groups: the 23 real cases with the gold clause in the top 3 score 0.04 to 1.00 (14 of them below 0.936), the 7 without it score 0.44 to 0.96, so one threshold cannot separate them. The 5 no-clause cases score 0.00 to 0.51.
**Decision:** the rule stays in the code and tests but is **off by default** (`tau_retr` is commented out in `configs/app.yaml`, with the tuned value in the comment). This decision was made after seeing `replan_v3.md`, which includes the test cases; the tuned value itself used validation only. The README reports this as a negative result.
**Acceptance (SPEC 14.5):** node and abstain behaviour tested (10 tests); calibration on validation only; no-clause abstention measured; the effect on the 30 cases measured and reported. The rule is not enabled because it hurts overall.

## R4 Finish (done 2026-11-13 ~05:40 IST; target 06:13)
**Done:** `eval.run_all` now also runs the audit report, v3 matcher table, appeal evaluation and (when `replan_v3.json` is missing) the re-plan evaluation; it ran end to end and rebuilt `summary.md`. Two stale report sentences fixed (`appeal_v3.md` verdict-mix wording, `summary.md` reflection trigger). README rewritten from `eval/results/` only: Phase 2b results (audit, v3 matcher with CIs, faithfulness, verdicts, re-plan negative result, appeal), Ombudsman verification status (Rule 14(3)(a)-(b) checked against the copy as amended till 09.11.2023, later amendments and IRDAI figures unchecked), limitations, CI scope. `ruff check .` clean; `pytest -q -m "not model"` 153 passed; `pytest -q -m model` 9 passed (local Ollama and models).
**Appeal rerun on the R1b verdict mix:** 4 drafts (2 NOT_SUPPORTED, 2 INSUFFICIENT_EVIDENCE), 4/4 first-try quote pass; forced stress test on the 12 gold NOT_SUPPORTED cases: 10 first try, 1 retry, 1 template, 12/12 complete.
**Not done / owner-only:** `[TODO: name, USN]` in the README; `docs/reflection_check.md` marks; `case_003` / `case_009` label decisions; the 3 blank review-queue rows.
**Known issues:** the Phase 2 pretrained-baseline row in `matcher_v2.md` is re-rendered under the current (owner-reviewed) labels, so its numbers differ from the Phase 2 README; `matcher_v3.md` shows old and new labels side by side. Re-chunking and `v2_008_pos` untouched. Local `main` has an unpushed docs commit `39ba4be` that diverges from `origin/main`.

**Update (2026-11-13):** README author and team members filled in by the owner (no USN, as requested).
