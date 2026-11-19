# SPEC.md: ClaimLens (v1)

**Two phases.** Phase 1 (sections 1-11) is the shorter version built overnight for the 5 Nov presentation. Phase 2 (section 12) is the full project scope, built after the presentation. Phase 1 must be designed so Phase 2 extends it without rewrites (section 12.1).

Repo: `claim_lens`. Owner: [TODO name]. Build window: Wed 4 Nov, about 20:30, to Thu 5 Nov, 05:00 IST (hard stop).
Machine: Windows laptop with NVIDIA RTX 4060. App LLM: local Ollama.

## 1. Goal

An explainable agent that audits health-insurance claim rejections (India).

- **Input:** rejection letter text + a policy document.
- **Output:** verdict, best-matching clause with token-level attributions, an LLM explanation checked against those attributions (faithfulness), and a short appeal draft.
- **Core research question:** does checking an LLM's explanation against Integrated Gradients attributions, and revising it, make the explanation more faithful?

## 2. Scope for tonight

| Priority | Features |
|---|---|
| Must | Policy PDF to clause chunks; hybrid retrieval (BM25 + dense); pretrained cross-encoder as clause matcher (no fine-tuning); Integrated Gradients attributions; Ollama explanation with cited spans; reflection loop with Attribution Alignment Score (AAS); verdict with abstain option; 2 MCP servers; linear LangGraph flow; Streamlit UI; evaluation scripts on 20 cases; GitHub Actions CI (lint + unit tests, section 7.10) |
| Should (only if ahead of schedule) | Template appeal draft; comprehensiveness and sufficiency metrics; random-span AAS baseline |
| Deferred to Phase 2 | Everything in section 12 |

**Cut line:** if behind schedule, drop "Should" items first. Never drop IG attributions, the reflection loop with AAS, or the evaluation script.

## 3. Milestones (times are targets in IST)

| ID | Target | Deliverable | Acceptance criteria |
|---|---|---|---|
| P0 | 20:30-21:00 | GitHub repo `claim_lens` created and initial push verified; venv; CUDA PyTorch; Ollama model pulled | `python -c "import torch; print(torch.cuda.is_available())"` prints True; `ollama run <model> "hi"` responds |
| M0 | 21:00-22:15 | 2 policy PDFs parsed and chunked; hybrid retrieval; GitHub Actions CI | `python -m claimlens.rag.search "waiting period" --doc <id>` returns clauses with section paths; tests pass; the CI run for the pushed commit shows success (`gh run list --limit 1`) |
| M1 | 22:15-23:45 | 20 synthetic cases + ~80 labeled pairs (generated with Ollama, labeled `synthetic`); matcher scoring; IG attributions | IG output rendered to HTML for 3 sample pairs; matcher metrics saved to `eval/results/matcher.json` |
| M2 | 23:45-01:45 | Explanation + reflection with AAS; 2 MCP servers; LangGraph flow | One case runs end to end from the CLI and prints verdict, initial and final explanation, AAS before and after |
| M3 | 01:45-03:30 | Streamlit UI; evaluation on all 20 cases | `eval/results/summary.md` produced by `python -m eval.run_all` |
| M4 | 03:30-04:30 | README with real numbers from `eval/results/`, demo steps, final push | Fresh clone runs the demo case using README steps |
| Buffer | 04:30-05:00 | Fixes only | |

## 4. User flow

1. Pick one of the sample cases (or paste a letter) and select a policy.
2. System extracts the rejection reason, retrieves candidate clauses, scores each pair.
3. UI shows: verdict banner, best clause with attribution highlighting, explanation before and after reflection with AAS values, and the appeal draft if built.

## 5. Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11 on Windows (PowerShell) |
| LLM | Ollama, default model `qwen2.5:7b` [TODO verify tag; fallback `llama3.1:8b`], JSON output mode, called through `claimlens/llm/client.py` |
| Transformers | PyTorch (CUDA build), Hugging Face Transformers, sentence-transformers |
| Matcher | `cross-encoder/ms-marco-MiniLM-L-6-v2` [verify ID] |
| Embeddings | `BAAI/bge-small-en-v1.5` [verify ID], vectors held in NumPy (no vector DB tonight) |
| Keyword search | rank_bm25 |
| PDF | PyMuPDF |
| Explainability | Captum (LayerIntegratedGradients) |
| Agent and MCP | LangGraph; MCP Python SDK (FastMCP, stdio); MCP Inspector to test servers |
| Storage | JSON files under `data/` and `runs/` |
| UI | Streamlit |
| CI | GitHub Actions (`ubuntu-latest`, Python 3.11, ruff + pytest on CPU) |

## 6. Data

- **Policies:** 2 public individual health-policy wordings (text-based PDFs) in `data/policies/`. If a PDF parses badly, keep a cleaned `.txt` beside it.
- **Cases:** `data/cases/*.json` with `{case_id, policy_id, letter_text, reason, gold_clause_ids, gold_verdict, source: "synthetic"}`. Generate 20 with Ollama from chosen clauses (about 12 SUPPORTED, 8 NOT_SUPPORTED), then hand-check gold labels for at least 5 and note it in `data/QA.md`.
- **Pairs:** `data/pairs.jsonl` with `{pair_id, policy_id, reason, clause_id, clause_text, label, source}`; about 40 positive and 40 hard negatives (similar clause from another section). Used only to evaluate the pretrained matcher tonight.
- Everything synthetic is labeled synthetic. No real personal data.

## 7. Components

### 7.1 Schemas (`claimlens/schemas.py`)

`Chunk`, `RetrievedClause`, `MatchResult`, `Attribution`, `CitedSpan`, `Explanation`, `Verdict` (SUPPORTED, NOT_SUPPORTED, INSUFFICIENT_EVIDENCE), `CaseResult` (case_id, reason, top_clauses, match, attribution, explanation_initial, explanation_final, aas_initial, aas_final, verdict, appeal_draft optional).

### 7.2 Parsing and retrieval (`claimlens/rag/`)

Chunk along clause numbers and headings, keeping `section_path` and `page`, 100-300 tokens per chunk. BM25 + dense fused with reciprocal rank fusion, top 10, re-ranked by the matcher, top 3 returned.

### 7.3 Matcher (`claimlens/matcher/`)

Score `(reason, clause)` with the pretrained cross-encoder; `support_score = sigmoid(logit)`. Evaluate on `pairs.jsonl` (accuracy, F1, ROC-AUC); pick the decision threshold on half the pairs and report on the other half.

### 7.4 Integrated Gradients (`claimlens/explain/`)

Captum `LayerIntegratedGradients` on the cross-encoder's embedding layer, pad-token baseline (keep special tokens), `n_steps=50`. Sum over the embedding dimension, L2-normalise, merge sub-words into words. `render.py` writes HTML with background shading proportional to positive attribution.

### 7.5 Explanation, reflection and AAS

1. Ollama returns JSON `{explanation, cited_spans: [{source, text}]}`; spans must be exact substrings of the reason or clause.
2. **AAS** = positive attribution mass on tokens inside cited spans divided by total positive attribution mass, in [0, 1]. Unmatched spans count as zero and are logged.
3. If AAS < 0.6: send feedback (high-attribution words not cited; cited words with low attribution) and ask for a revision. The verdict must not change. Up to 2 rounds; keep the best.
4. Report AAS initial vs final for every case. If time allows (Should), also report a random-span AAS baseline and comprehensiveness/sufficiency, since AAS alone can be raised just by quoting top tokens.

### 7.6 Verdict

SUPPORTED if best support score >= `tau_high`; NOT_SUPPORTED if < `tau_low`; otherwise INSUFFICIENT_EVIDENCE. Start with 0.7 and 0.3 and tune on the first 10 cases; report on all 20 and say so.

### 7.7 MCP servers (`claimlens/mcp_servers/`)

- **policy-search:** `search_clauses(query, doc_id, top_k=5)`, `get_clause(doc_id, clause_id)`.
- **explainer:** `score_pair(reason, clause)`, `attribute_ig(reason, clause)`, `alignment_score(reason, clause, cited_spans)`.

Thin wrappers over the core functions; test each with MCP Inspector. The LangGraph nodes call tools through an MCP client. **Fallback:** if MCP client wiring takes more than 30 minutes, nodes call the same core functions directly; note it in `docs/PROGRESS.md` and the README.

### 7.8 Agent (`claimlens/agent/`)

Linear LangGraph: `parse -> retrieve -> match -> explain_reflect -> verdict -> draft(optional)`. State is the `CaseResult` being built. CLI: `python -m claimlens.agent.run --case data/cases/case_001.json`.

### 7.9 UI (`app/streamlit_app.py`)

Single page: case picker or pasted letter, run button, verdict, clause with IG highlighting, initial vs final explanation with AAS, appeal draft. Show "decision support, not legal advice" on every result.

### 7.10 CI (`.github/workflows/ci.yml`)

- **Triggers:** push and pull request on `main`.
- **Job (ubuntu-latest, Python 3.11, pip cache enabled):** install `requirements-ci.txt`, run `ruff check .`, then `pytest -q -m "not model"`.
- **`requirements-ci.txt` stays light:** pytest, ruff, pydantic, numpy, rank_bm25, PyMuPDF, pyyaml. No torch, Captum, sentence-transformers, Streamlit or Ollama client in CI.
- **Tests marked `@pytest.mark.model`** (they need torch, a Hugging Face model, Ollama or a GPU) are deselected in CI and run locally with `pytest -q -m model`. Register the marker in `pytest.ini`.
- **No secrets in CI.** CI cannot run the Ollama or GPU parts, so the README must say CI covers lint and unit tests only.
- **Pushing the workflow file needs the `workflow` scope:** run `gh auth refresh -s workflow` once before the first push of `.github/`.
- Add a CI status badge to the README after the first green run.
- If CI install or lint fails for environment reasons, spend at most 15 minutes fixing it, otherwise log it in `docs/PROGRESS.md` and continue.

## 8. Evaluation (`eval/`)

| Script | Metrics |
|---|---|
| `run_retrieval.py` | Recall@3, MRR against `gold_clause_ids`; BM25-only vs hybrid |
| `run_matcher.py` | Accuracy, F1, ROC-AUC on pairs |
| `run_faithfulness.py` | AAS initial vs final per case and averaged |
| `run_e2e.py` | Verdict agreement with `gold_verdict` |
| `run_all.py` | Runs all and writes `eval/results/summary.md` |

All numbers in the README come from these outputs. The README states the data is synthetic and small (20 cases).

## 9. Tests

pytest: chunking on a sample text, retrieval finds a known clause, AAS on a hand-made example, schema validation, end-to-end run with a `FakeLLM`. Tests that need torch, a Hugging Face model, Ollama or a GPU get `@pytest.mark.model`; CI deselects them (`-m "not model"`).

## 10. Risks and fallbacks

| Risk | Fallback |
|---|---|
| CUDA PyTorch install fails | Run on CPU (cross-encoder is small; IG will be slower but works) |
| Ollama JSON is unreliable | Retry once; use Ollama's JSON format option; shorten prompts |
| Ollama model too slow | Switch to a smaller model, e.g. `qwen2.5:3b` [verify tag] |
| PDF parsing is messy | Use a cleaned `.txt` per policy |
| MCP client wiring stalls | Direct function calls behind the same interface (7.7) |
| Claude Code usage limit reached | Enable usage credits or continue the remaining steps manually from `docs/PROGRESS.md` |
| CI fails on push (workflow scope, dependencies, lint) | `gh auth refresh -s workflow`; trim `requirements-ci.txt`; mark heavy tests `model`; 15-minute limit, then log and continue |
| Running late | Apply the cut line in section 2 |

## 11. Open items

- [ ] Name and USN for the README
- [ ] Which 2 insurer policy PDFs to use
- [ ] Confirm Ollama and Hugging Face model tags

## 12. Phase 2: full scope (after the presentation; not part of the overnight run)

### 12.1 Design rules for Phase 1 so Phase 2 is an extension, not a rewrite

- Retrieval sits behind a `Retriever` interface; the NumPy store can be swapped for Chroma or Qdrant.
- The matcher loader takes a model name or local checkpoint path, so a fine-tuned model drops in.
- The explainer exposes one `attribute(method=...)` function, so SHAP and attention rollout plug in beside IG.
- The LLM client is provider-agnostic (Ollama now; a hosted API later via env vars).
- The agent graph is built from named nodes, so a planner, re-plan edge and new workers can be added.
- Storage goes through one small module, so JSON files can be replaced by PostgreSQL.

### 12.2 Phase 2 features

| Area | Phase 2 work |
|---|---|
| Data | ~300 labeled pairs (10% manually verified), 30+ end-to-end cases, 3-5 policies, cases adapted from anonymised public ombudsman awards; splits by policy document |
| Matcher | Fine-tune LEGAL-BERT or InLegalBERT cross-encoder [verify IDs]; baselines: BM25, pretrained cross-encoder, zero-shot LLM; early stopping on validation F1 |
| Explainability | SHAP cross-check with IG (Spearman); attention rollout as a contrast; comprehensiveness and sufficiency at k = 10/20/50; random-span AAS baseline; IG baseline sensitivity |
| Reflection | Tune `tau` on validation data; manual sanity check of 10 revised explanations |
| Agents | Planner node with plan-and-execute; re-plan on low retrieval confidence, then abstain; orchestrator with Parser, Retriever, Matcher and Drafter workers |
| Memory | Case state and long-term store (policy index, circular excerpts, past outcomes) in PostgreSQL |
| MCP | Add a regulations server (IRDAI circular search) and an appeal server (templates, Ombudsman deadline calculator) |
| Appeal | Full drafter: facts, cited clauses, reasoning, relief, document checklist, filing-window reminder (verify against Insurance Ombudsman Rules, 2017) |
| Backend and deploy | FastAPI endpoints (`/cases`, `/cases/{id}/run`, `/cases/{id}`); Docker Compose; CI extended with Docker build and coverage (basic CI already exists from Phase 1) |
| Evaluation | All hypotheses H1-H3 from the report; RAGAS for retrieval-grounded answers; MLflow or W&B tracking |
| Stretch | OCR for scanned letters; Hindi and Kannada; counterfactual explanations of what would change the outcome |

## 13. Phase 2 cut for 3 days (Fri 6 to Sun 8 Nov 2026)

This is the part of section 12 that is built now. Everything in 12.2 not listed here is deferred (see 13.6). Same integrity rules as Phase 1: every number in the README comes from `eval/results/`, negative results are reported as they are.

### 13.1 Time plan (IST). Hard stop: Sun 8 Nov 22:00. Cut line: if more than 30 min behind a target, apply 13.6 without asking.

| Milestone | Target window | Deliverable |
|---|---|---|
| P1 Data | Fri 6 Nov 09:00-13:00 | Larger labeled set, 30 cases, splits, verification log |
| P2 Matcher | Fri 6 Nov 14:00-20:00 | Fine-tuned cross-encoder and baseline table |
| P3 Explainability | Sat 7 Nov 09:00-15:00 | SHAP, attention rollout, comprehensiveness and sufficiency, random-span AAS |
| P4 Reflection | Sat 7 Nov 16:00-19:00 | Tuned `tau`, manual check of 10 revised explanations |
| P5 Serve | Sun 8 Nov 09:00-18:00 | FastAPI, Docker Compose, extended CI, README with new results |
| Buffer | Sun 8 Nov 18:00-22:00 | Fixes, final report |

### 13.2 P1 Data
- `data/pairs_v2.jsonl`: about 200 labeled `(reason, clause)` pairs. Fields: `pair_id, policy_id, reason, clause_id, clause_text, label, source: "synthetic", verified: bool`. Generate with Ollama from chosen clauses, as in Phase 1.
- Add a third policy PDF (the user supplies it; see 13.7). Split: train and validation from policies 1 and 2, split by `clause_id` so no clause appears in both; test = the third policy. If no third PDF is available, hold out whole clause groups and say so in the README (fallback, logged in `docs/PROGRESS.md`).
- Cases: grow `data/cases/` to 30. Include the hand-written cases from `data/cases_extra/` (t01 to t06 and the two realistic letters) with real `gold_clause_ids`.
- The user hand-verifies at least 20 pairs (10%). Record IDs and corrections in `data/QA.md`. Claude Code prepares a CSV of the 20 pairs for the user to mark; do not mark verified yourself.
- Acceptance: files exist, splits have no clause leakage (test it), verified count >= 20, label balance reported.

### 13.3 P2 Matcher
- Script `python -m eval.train_matcher --base <model-id-or-path> --out models/matcher_v2` (models/ is git-ignored). Seed 42, fp16, batch 16, max length 256, at most 4 epochs, early stopping on validation F1.
- Base models to try, in order: `nlpaueb/legal-bert-base-uncased`, `law-ai/InLegalBERT` [verify IDs]. Fallback if downloads fail or both do worse than the Phase 1 model: fine-tune the existing `cross-encoder/ms-marco-MiniLM-L-6-v2`. If the GPU runs out of memory: batch 8, max length 192, gradient accumulation 2.
- Baselines on the same test set: BM25 score threshold, the Phase 1 pretrained cross-encoder, a zero-shot Ollama LLM.
- Report accuracy, F1 and ROC-AUC on validation and test, and on the verified subset. Output `eval/results/matcher_v2.md`.
- The matcher loader (`claimlens/matcher`) already accepts a checkpoint path; the agent switches model through `configs/app.yaml`, not code changes.
- Acceptance: the fine-tuned model runs through `python -m claimlens.agent.run`; the table is complete even if the fine-tuned model does not win.

### 13.4 P3 Explainability and P4 Reflection
- `attribute(method="ig" | "shap" | "rollout")` returns the same word-level structure for all methods.
- SHAP with the `shap` text explainer at word level. Time-box: if it takes more than 60 minutes per run, use 10 cases and state it.
- Report Spearman correlation between IG and SHAP over the cases, with IG baseline sensitivity (pad vs zero-embedding baseline).
- Comprehensiveness and sufficiency at k = 10, 20, 50 tokens, for IG, SHAP, attention rollout and a random-token baseline. Output `eval/results/faithfulness_v2.md`.
- AAS with a random-span baseline (already in Phase 1; keep it).
- P4: tune the reflection threshold `tau` on validation cases only (grid 0.4 to 0.8), then report once on test. Write `docs/reflection_check.md` with 10 revised explanations, each with the original, the revision and the AAS change; the user marks each as better, same or worse.
- Acceptance: all result files exist and are produced by `python -m eval.run_all`.

### 13.5 P5 Serve
- FastAPI app `api/main.py`: `POST /cases` (create from letter text and policy id), `POST /cases/{id}/run`, `GET /cases/{id}`. Pydantic schemas from `claimlens/schemas.py`. Case state stored through the existing storage module (JSON files; no PostgreSQL).
- Docker Compose: `api` and `ui` services. Ollama stays on the host (`OLLAMA_HOST=http://host.docker.internal:11434`); do not containerise the GPU model.
- CI: add `docker build` (build only, no push) and `pytest --cov` to `.github/workflows`; keep `pytest -m "not model"` and `ruff check .`.
- README: update with the new tables, copied from `eval/results/` only. Add the name and USN open item.
- Acceptance: `docker compose up` serves the API and UI from a fresh clone; API tests pass with the LLM and models mocked; `gh run list --limit 1` shows success.

### 13.6 Deferred (do not build)
Planner and orchestrator agents and re-planning, PostgreSQL and long-term memory, regulations and appeal MCP servers, the full appeal drafter, RAGAS, MLflow or W&B, OCR, Hindi and Kannada, counterfactual explanations, a vector-database swap.

Cut order if behind: (1) IG baseline sensitivity, (2) attention rollout, (3) Docker Compose, (4) CI extension, (5) SHAP down to 10 cases. Never cut: P1, P2, comprehensiveness and sufficiency, the README.

### 13.7 Open items before starting
- [ ] Which third policy PDF (the user supplies it)
- [ ] Name and USN for the README
- [ ] The 20 pairs to hand-verify (done by the user in P1)
- [ ] Confirm Hugging Face model IDs in 13.3
- [ ] Fix `gold_clause_ids` placeholders in the t01 to t06 cases

## 14. Phase 2b: remaining work, 4-hour cut

This is the part of section 12.2 and 13.6 built next, in about 4 hours of Claude Code time. The order is fixed: data quality first, because the label audit showed that a large share of the generated "positive" labels are wrong, and that limits every model result. Everything not listed here stays deferred (14.6). Same integrity rules as before: every number comes from `eval/results/`, negative results are reported as they are, nothing is tuned on test.

Branch: `phase2b`, created from `main` after the Phase 0 check. Milestones: R1a, R2, R1b, R3, R4.

### 14.1 Time plan

T0 is the moment R1a starts. Write T0 into `docs/PROGRESS.md` and compute the targets from it.

| Milestone | Target window | Deliverable |
|---|---|---|
| R1a Label audit | T0 to T0+0:45 | LLM-judge audit of all pairs, review queue for the owner |
| R2 Appeal | T0+0:45 to T0+1:50 | Full appeal drafter, appeal MCP server, API and UI hook (owner reviews the queue in parallel) |
| R1b Retrain | T0+1:50 to T0+2:40 | Cleaned labels, hard negatives, retrained matcher, re-evaluation |
| R3 Re-plan | T0+2:40 to T0+3:15 | Low-confidence re-plan, then abstain |
| R4 Packaging | T0+3:15 to T0+3:50 | README, tests, CI, merge, final report |
| Buffer | T0+3:50 to T0+4:00 | Fixes |

Hard stop: T0+4:00. At T0+3:30 stop new work and finish R4. If more than 20 minutes behind at any target, apply the cut order in 14.6 without asking.

### 14.2 Owner steps before T0 (Claude Code does not do these)

1. Decide on the 20 pair-verification labels. Only the owner's statement counts. If the owner says they reviewed all 20 and agree, Claude Code commits the five pending files and rewrites every note prefix from `CLAUDE (conf)` to `owner-confirmed (Claude note, conf)`. If the owner changed labels, rerun `python -m claimlens.data.phase2 apply-verify`. If not reviewed, stop and wait.
2. Mark the 10 explanations in `docs/reflection_check.md`.
3. Add name and USN to `README.md` (replace `[TODO: name, USN]`).
4. Decide the case_003 (questionable) and case_009 (wrong) labels.
5. Have Ollama running with the model in `.env` pulled.

Claude Code never fills `your_label` or any `verified` field itself.

### 14.3 R1 Data quality

**R1a: audit (T0 to T0+0:45)**
- Add `claimlens/data/audit.py` and `eval/run_audit.py`. Judge every pair in `data/pairs_v2.jsonl` with the local LLM (`qwen2.5:7b`), using this rubric: label 1 only if the clause contains the operative rule the reason relies on, so that if the facts in the reason are true the rejection is justified; label 0 if the clause is on another topic, shares only keywords, or contradicts or does not establish the reason's ground. The judge returns JSON `{label, confidence: high|medium|low, rationale}` with a rationale of at most 30 words.
- Two passes per pair (the second with the clause and reason order changed). A pair is flagged if either pass disagrees with the generated label or reports low confidence.
- Write `data/label_audit.csv` (`pair_id, generated_label, judge_label_1, judge_label_2, confidence, rationale, flagged`). Judge labels are stored with `source: "llm_audit"` and never `verified: true`.
- Calibration: report the judge's agreement with the owner-verified 20 pairs. With n = 20 this is only a sanity check, and the report says so. Note the limitation that the judge and the generator are the same model family, so their errors may be correlated.
- Write `data/review_queue.csv` for the owner: all flagged pairs (cap 30) plus 10 random unflagged pairs, with columns `pair_id, reason, clause_text, generated_label, judge_label, rationale, your_label, note`. The 10 random ones estimate how many errors the judge misses.
- Acceptance: both files exist, the calibration number is in `eval/results/audit_v3.md`, the queue has at most 40 rows, tests cover the flagging rule.

**R1b: retrain and re-evaluate (T0+1:50 to T0+2:40)**
- Apply the owner's `your_label` values from `data/review_queue.csv` (`python -m claimlens.data.audit apply-review`), recorded as `verified: true`. If the owner has not returned the queue by T0+1:50, use the judge labels for flagged pairs, record them as `source: "llm_audit", verified: false`, and say so in every table that uses them. Never change a label silently.
- Splits are frozen: do not move any pair between train, validation and test. Corrected labels apply inside the same splits. Report old-label and new-label results side by side.
- Add up to 60 hard negatives: for each positive reason take top-5 retrieved clauses that are not gold, keep those the judge labels 0 with high confidence. They are `hard_negative: true`, split by clause group, added to train and validation only. Evaluate them on a separate hard-negative test file built from test-group clauses.
- Retrain the cross-encoder with `eval/train_matcher.py` (same settings as 13.3) on seeds 42, 43 and 44. Report mean and standard deviation, and 95% bootstrap confidence intervals (1000 resamples, seed 42) for test accuracy, F1 and ROC-AUC, against the pretrained baseline.
- Re-tune `tau_high` and `tau_low` on validation only (`eval.tune_tau`), rerun `eval.run_e2e` on the 30 cases, and rerun faithfulness on a 10-case subset (state it as a subset). Write `eval/results/matcher_v3.md`.
- Acceptance: tables show old vs new labels, confidence intervals are present, the leakage test still passes, nothing was tuned on test, and the README text is unchanged until R4.

### 14.4 R2 Appeal (T0+0:45 to T0+1:50)

- New package `claimlens/appeal/`. The current template in `claimlens/agent/verdict.py` (`appeal_template`) stays as the fallback.
- `draft_appeal(case_result)` returns an `AppealDraft` (pydantic, in `claimlens/schemas.py`) with: facts, cited clauses (`clause_id` and quote), reasoning, relief sought, document checklist, deadline note, disclaimer. It is generated for NOT_SUPPORTED, and for INSUFFICIENT_EVIDENCE it produces a request-for-clarification letter instead. It is not generated for SUPPORTED.
- Grounding rules: every quote must appear verbatim (after whitespace normalisation) in the retrieved clause text, or the draft is retried once, then replaced by the template with `source: "template"`. Facts come only from the letter; anything missing becomes a visible placeholder like `[date of rejection]`. No invented policy numbers, dates or amounts.
- New MCP server `claimlens/mcp_servers/appeal_server.py` (FastMCP, stdio, thin wrapper) with tools `list_templates`, `get_template(name)` and `compute_deadlines(rejection_date, grievance_filed_date=None, insurer_reply_date=None)`. Deadline logic uses stdlib `datetime` only. The filing-window rules must be checked against the Insurance Ombudsman Rules, 2017 text from an official source; if it cannot be reached, ship with `verified: false` on those rules and show "verify with the Ombudsman office" in the UI and README. Do not state the rules as fact without that check.
- API: `GET /cases/{id}/appeal`. UI: an appeal section with the structured draft and a text download. Every draft shows "decision support, not legal advice".
- Tests (no model needed): verbatim-quote check and fallback with a mocked LLM, schema, deadline arithmetic at month ends and on 29 February, the three tool outputs.
- Evaluation: over the 30 cases, report the share of drafts whose quotes pass on the first try, the share that fall back to the template, and the share of NOT_SUPPORTED cases with a complete draft. Write `eval/results/appeal_v3.md`.
- Acceptance: tests pass, the MCP server starts, `appeal_v3.md` exists, CI green.

### 14.5 R3 Re-plan (T0+2:40 to T0+3:15)

- Add two named nodes to the LangGraph agent in `claimlens/agent/graph.py`: `check_retrieval` (a conditional edge after `retrieve`) and `replan`.
- Rule: if the top re-ranked clause score is below `tau_retr`, run `replan` once: the LLM rewrites the reason into two alternative search queries, retrieval runs on each, results are merged by reciprocal rank fusion. If the score is still below `tau_retr`, stop and return INSUFFICIENT_EVIDENCE with `abstain_reason: "no clause above confidence threshold"`. At most one re-plan per case.
- Calibrate `tau_retr` on validation cases only, choosing the value that best separates cases whose gold clause is in the top 3 from those whose gold clause is not. Add 5 synthetic "no matching clause" cases (reasons with no basis in the policy, `source: "synthetic"`, gold verdict INSUFFICIENT_EVIDENCE, labelled by construction) so abstention can be tested.
- Add `plan_trace` to `CaseResult` recording each decision. Put `tau_retr` in `configs/app.yaml`.
- Tests: the conditional edge with a mocked retriever and LLM, abstain path, at-most-once rule.
- Evaluation: gold-clause-in-top-3 before and after re-plan on validation and test, the abstain rate on the 5 no-clause cases, and any regression on the 30-case run. Write `eval/results/replan_v3.md`.
- Acceptance: the graph has the new nodes, tests pass, results file exists, and no previous test breaks.

### 14.6 R4 Packaging, cut order, deferred

R4: update `README.md` with a new results section copied from `eval/results/` only, append `docs/PROGRESS.md`, run `ruff check .`, `pytest -q -m "not model"`, `pytest -q -m model`, `python -m eval.run_all`, push `phase2b`, run CI with `gh workflow run` and wait for success, merge to `main` only after green, tag `phase2b`, and write the final report in the MASTER_PROMPT format.

Cut order if behind: (1) R3 re-plan, (2) the appeal MCP server (keep the drafter), (3) hard negatives (keep the audit and retrain), (4) faithfulness rerun on the 10-case subset, (5) bootstrap intervals down to 500 resamples. Never cut: R1a, R1b retrain and re-evaluation, the appeal drafter's quote check, R4.

Deferred, do not build: orchestrator and worker refactor of the agent, PostgreSQL memory and long-term store, the regulations (IRDAI circular) MCP server, RAGAS, MLflow or W&B, OCR, Hindi and Kannada, counterfactual explanations, a vector-database swap, a third policy PDF and more policies (optional if the owner supplies one). No new dependencies.

### 14.7 Open items

- [ ] Owner steps in 14.2
- [ ] Insurance Ombudsman Rules, 2017: confirm the filing-window rules from an official source (R2)
- [ ] README name and USN
- [ ] CLAUDE.md says RTX 4060, the machine reports RTX 4050 6 GB: correct it
