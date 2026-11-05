# ClaimLens

![CI](https://github.com/sandeepklr12345-gif/claim_lens/actions/workflows/blank.yml/badge.svg)

An explainable agent that audits **health-insurance claim rejections (India)**.
Input: a rejection letter and a policy document. Output: a verdict (with an abstain option), the
best-matching policy clause with word-level **Integrated Gradients** attributions, an LLM
explanation checked against those attributions (**faithfulness**), and a template appeal draft.

> **Decision support, not legal advice.** Explainable AI mini project, Phase 2 (Phase 1 was
> presented on 5 Nov 2026, tag `phase1-presented`). Author: [TODO: name, USN].

**Research question:** does checking an LLM's explanation against Integrated Gradients attributions,
and asking it to revise, make the explanation more faithful?
**Short answer so far:** only a little (mean AAS 0.165 → 0.215 on 30 synthetic cases, random spans
0.166), and the gain varies between runs. See [Results](#results) and [Limitations](#limitations).

## How it works

```
letter ─ parse ─ retrieve ─ match ─ explain_reflect ─ verdict ─ draft
         (LLM)   BM25+dense  cross-    LLM explanation     tau_high/   template
                 RRF, top 10 encoder   cites clause spans; tau_low,     appeal
                 → re-rank   score     IG attributions →   abstains     (if NOT_
                 → top 3                AAS; revise < tau                SUPPORTED)
```

- **Policies:** two public wordings (Care Supreme, HDFC ERGO Optima Secure), chunked along clause
  numbers (`data/policies/`, sources in `data/policies/SOURCES.md`).
- **Matcher:** `cross-encoder/ms-marco-MiniLM-L-6-v2` fine-tuned on synthetic pairs. The checkpoint
  lives in `models/`, which is not in git. Without it the agent logs a warning and uses the
  pretrained model. Build it with:
  `python -m eval.train_matcher --base cross-encoder/ms-marco-MiniLM-L-6-v2 --out models/matcher_v2`
- **Attribution:** `attribute(method=...)` gives word scores from Integrated Gradients (Captum, 50
  steps, pad baseline), SHAP or attention rollout. The agent uses IG.
- **AAS (Attribution Alignment Score):** share of positive IG mass that falls inside the spans the
  explanation cites. Below `tau_aas` the agent sends feedback (influential words not cited, cited
  words that did not matter) and asks for a revision, up to 2 rounds, keeping the best.
- **LLM:** local Ollama (`qwen2.5:7b`). No paid API calls.
- **Agent:** linear LangGraph. Tools are served by two **MCP servers** (`policy-search`,
  `explainer`) over stdio; `--mcp` makes the agent call them, the default calls the same functions
  in-process.

## Run it

### With Docker

Needs Docker and [Ollama](https://ollama.com) running on the host. Ollama stays outside the
containers, and the containers run the matcher on the CPU.

```powershell
git clone https://github.com/sandeepklr12345-gif/claim_lens.git ; cd claim_lens
ollama pull qwen2.5:7b
docker compose up --build
```

UI: <http://localhost:8501>. API docs: <http://localhost:8000/docs>. The first audit downloads the
Hugging Face models. Put a checkpoint in `models/matcher_v2` to use the fine-tuned matcher.

### API

```powershell
$body = @{ letter_text = "Your claim is rejected because of the waiting period."; policy_id = "care_supreme" } | ConvertTo-Json
$case = Invoke-RestMethod -Method Post http://localhost:8000/cases -ContentType "application/json" -Body $body
Invoke-RestMethod -Method Post "http://localhost:8000/cases/$($case.case_id)/run" -TimeoutSec 1500
Invoke-RestMethod "http://localhost:8000/cases/$($case.case_id)"
```

### Without Docker (Windows, PowerShell)

Needs Python 3.11; an NVIDIA GPU is optional.

```powershell
git clone https://github.com/sandeepklr12345-gif/claim_lens.git ; cd claim_lens
py -3.11 -m venv .venv ; .\.venv\Scripts\Activate.ps1      # any venv location works
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA build; CPU also works
pip install -r requirements.txt
ollama pull qwen2.5:7b
python -c "import torch; print(torch.cuda.is_available())"
```

If `Activate.ps1` is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.
The first run downloads Hugging Face models (`BAAI/bge-small-en-v1.5`,
`cross-encoder/ms-marco-MiniLM-L-6-v2`).

**Demo (one case, prints verdict, initial and final explanation, AAS before and after):**

```powershell
python -m claimlens.agent.run --case data\cases\case_001.json          # in-process tools
python -m claimlens.agent.run --case data\cases\case_001.json --mcp    # through the MCP servers
```

**UI:** `streamlit run app\streamlit_app.py` with Ollama running. Pick a sample case or paste a letter
and press *Run audit*. A live panel shows each step as it runs, then the verdict and evidence appear.
The first run in a session also loads the models, so it takes longest. The page loads its fonts from
Google Fonts when online and falls back to system fonts otherwise; letters stay on this computer.
`streamlit run app\preview_result.py` shows the result screen with hand-made sample values.

**API:** `uvicorn api.main:app --reload`

**Search a policy:** `python -m claimlens.rag.search "waiting period" --doc care_supreme`

**MCP servers (stdio):** `python -m claimlens.mcp_servers.policy_server` and
`python -m claimlens.mcp_servers.explainer_server` (test with MCP Inspector).

**Evaluation:** `python -m eval.run_all` writes everything in `eval/results/`. Per-case agent
results are cached in `eval/results/e2e/`; with the cache it took about 5 minutes, and the agent run
on all 30 cases took about 14 minutes. `python -m eval.run_all --summary` only rebuilds the tables.
**Tests:** `pytest -q -m "not model"` (no GPU) and `pytest -q -m model` (needs models and Ollama).

## Results

All numbers come from `eval/results/`. **All data is synthetic and small: 206 pairs and 30 cases,
generated by the local LLM, with labels by construction.** No pair is owner-verified yet, and only 5
cases were hand-checked, not by the owner (3 sound, 1 questionable, 1 wrong; see `data/QA.md`).
Phase 1 results are kept at tag `phase1-presented`.

**Split.** No third policy PDF was available, so test is whole held-out section groups of the same
two policies, not an unseen policy. Train, validation and test share no clause, and case gold
clauses stay out of train (`eval/results/data_v2.md`).

| Split | Pairs | Positive / negative | Cases |
|---|---:|---:|---:|
| Train | 114 | 57 / 57 | 0 |
| Validation | 52 | 26 / 26 | 17 |
| Test | 40 | 20 / 20 | 13 |

### Matcher

From `eval/results/matcher_v2.md`. Thresholds were picked on validation.

| System | Val F1 | Val AUC | Test acc | Test F1 | Test AUC |
|---|---:|---:|---:|---:|---:|
| BM25 score (baseline) | 0.741 | 0.744 | 0.600 | 0.619 | 0.618 |
| Phase 1 pretrained cross-encoder (baseline) | 0.862 | 0.834 | 0.600 | 0.680 | 0.755 |
| Zero-shot Ollama LLM (baseline) | 0.667 | 0.627 | 0.500 | 0.667 | 0.799 |
| Fine-tuned legal-bert | 0.667 | 0.620 | 0.525 | 0.537 | 0.412 |
| Fine-tuned InLegalBERT | 0.711 | 0.664 | 0.525 | 0.486 | 0.560 |
| Fine-tuned ms-marco-MiniLM (the agent's model) | 0.833 | 0.833 | 0.625 | 0.706 | 0.772 |

**Fine-tuning did not beat the best baseline on validation:** the pretrained cross-encoder has the
higher validation F1. The fine-tuned MiniLM is slightly ahead on test, but with n = 40 that gap is
not reliable. The two legal models did worse. The LLM's threshold labels nearly everything positive
(accuracy 0.500), although its test ROC-AUC is the highest. Scores on an owner-verified subset are
pending.

### Attribution faithfulness

From `eval/results/faithfulness_v2.md`: change in the matcher's logit on 30 cases. Higher
comprehensiveness is better; lower sufficiency is better.

| Method | Compr. k=10 | k=20 | k=50 | Suff. k=10 | k=20 | k=50 |
|---|---:|---:|---:|---:|---:|---:|
| Integrated Gradients | 4.482 | 5.880 | 8.347 | 1.186 | 0.032 | -0.663 |
| SHAP | 6.705 | 9.180 | 12.392 | -0.507 | -1.350 | -1.588 |
| Attention rollout | 1.977 | 3.259 | 8.377 | 6.822 | 3.885 | 0.151 |
| Random words (baseline) | 0.098 | 0.197 | 1.119 | 11.966 | 9.970 | 4.803 |

All three methods beat random words, and SHAP scores best. They do not agree closely: word-level
Spearman between IG and SHAP is 0.247 (SD 0.150). IG also depends on its baseline: Spearman between
the pad and zero-embedding baselines is 0.394 (SD 0.132).

### Reflection and verdicts

From `eval/results/summary.md`, `reflection_v2.md` and `tau.json`.

| Component | Result |
|---|---|
| Mean AAS, initial → after reflection (30 cases) | 0.165 → 0.215 (improved in 7, unchanged in 23) |
| Mean AAS of random spans of the same size | 0.166 |
| Test only, 13 cases, `tau_aas` = 0.4 | 0.176 → 0.196 (improved in 1); random spans 0.130 |
| Verdict agreement on test, tuned thresholds 0.96 / 0.82 | 0.46 (6 of 13); always SUPPORTED also gives 0.46 |
| Verdict agreement on validation, tuned / always SUPPORTED | 0.76 / 0.71 |
| Verdict agreement on all 30 cases | 19 of 30 (63.3%); always SUPPORTED gives 18 of 30 |
| Retrieval Recall@3 (30 cases) | BM25 0.900, hybrid 0.867, hybrid + re-rank 0.867 |
| Gold clause in top 3, agent run | 80.0% |

The agent's owner check of 10 revised explanations is in `docs/reflection_check.md`, not yet marked.

## Limitations

- **The agent does no better than a trivial baseline on test.** It answers SUPPORTED for most cases:
  27 of 30, including 10 of the 12 NOT_SUPPORTED ones. The matcher's training pairs teach whether a
  clause suits a reason. A NOT_SUPPORTED case is a wrong rejection against a closely related benefit
  clause, so a high match score does not separate the two verdicts (test ROC-AUC 0.631).
- **The reflection gain is small and not stable.** Initial AAS equals random spans (0.165 vs 0.166).
  Two runs with the same seed gave validation mean final AAS of 0.166 and 0.230, so the local LLM's
  run-to-run variation is as large as the effect. There is no significance test. AAS can also be
  raised just by quoting top-attributed words.
- **The `tau_aas` threshold has no effect here.** All values from 0.4 to 0.8 give the same mean
  final AAS, because AAS rarely reaches 0.4; ties go to the lowest value.
- **AAS rests on IG, and IG depends on its baseline** and agrees only weakly with SHAP.
- **Re-ranking does not improve retrieval here,** and BM25 alone has the highest Recall@3 (n = 30).
- **Synthetic, noisy data.** Letters are short and generic, and gold labels are not verified. The
  hand-written cases and third policy planned in `SPEC.md` are not included. Train has 114 pairs
  over 57 clauses, so pairs are not independent.
- **CI covers lint, unit tests with coverage, and the Docker build** (no GPU, Ollama or models).
  Model-dependent tests run locally.
- The two policy PDFs come from third-party mirrors; check the UIN and version against the insurer.
- The appeal draft is a fixed template, not legal advice and not verified against the Insurance
  Ombudsman Rules.

## Repository layout

```
claimlens/   schemas.py  core.py  rag/  matcher/  explain/  llm/  agent/  mcp_servers/  data/  utils/
api/main.py  app/ (streamlit_app.py and views)   configs/app.yaml   .streamlit/config.toml
data/        policies/  cases/  pairs_v2.jsonl  splits.json  verify_pairs.csv  QA.md
eval/        run_*.py  train_matcher.py  tune_*.py  results/
docs/        PROGRESS.md (build log, bugs, deviations)  reflection_check.md
tests/       .github/workflows/blank.yml   Dockerfile   docker-compose.yml   SPEC.md   CLAUDE.md
```

Deferred work (planner agent, PostgreSQL, more MCP servers and others) is listed in `SPEC.md` 13.6.
