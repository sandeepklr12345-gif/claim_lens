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
