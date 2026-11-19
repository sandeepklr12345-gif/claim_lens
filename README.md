# ClaimLens

![CI](https://github.com/sandeepklr12345-gif/claim_lens/actions/workflows/blank.yml/badge.svg)

An explainable agent that audits **health-insurance claim rejections (India)**.
Input: a rejection letter and a policy document. Output: a verdict (with an abstain option), the
best-matching policy clause with word-level **Integrated Gradients** attributions, an LLM
explanation checked against those attributions (**faithfulness**), and a grounded appeal draft.

> **Decision support, not legal advice.** Explainable AI mini project, Phase 2b (Phase 1 was
> presented on 5 Nov 2026, tag `phase1-presented`; Phase 2 3-day cut, tag `phase2-3day`). Author: [TODO: name, USN].

**Research question:** does checking an LLM's explanation against Integrated Gradients attributions,
and asking it to revise, make the explanation more faithful?
**Short answer so far:** only a little (mean AAS 0.180 → 0.212 on 30 synthetic cases, random spans
0.145), and the gain varies between runs. See [Results](#results) and [Limitations](#limitations).

## How it works

```
letter ─ parse ─ retrieve ─ match ─ explain_reflect ─ verdict ─ draft
         (LLM)   BM25+dense  cross-    LLM explanation     tau_high/   grounded
                 RRF, top 10 encoder   cites clause spans; tau_low,     appeal
                 → re-rank   score     IG attributions →   abstains     (if NOT_
                 → top 3                AAS; revise < tau                SUPPORTED);
                                                                         INSUFFICIENT:
                                                                         clarification)
```

- **Policies:** two public wordings (Care Supreme, HDFC ERGO Optima Secure), chunked along clause
  numbers (`data/policies/`, sources in `data/policies/SOURCES.md`).
- **Matcher:** `cross-encoder/ms-marco-MiniLM-L-6-v2` fine-tuned on synthetic pairs. The checkpoint
  lives in `models/`, which is not in git. Without it the agent logs a warning and uses the
  pretrained model. Build it with:
  `python -m eval.train_matcher --base cross-encoder/ms-marco-MiniLM-L-6-v2 --out models/matcher_v3_s42 --seed 42 --hard-negatives data/hard_negatives.jsonl`
- **Attribution:** `attribute(method=...)` gives word scores from Integrated Gradients (Captum, 50
  steps, pad baseline), SHAP or attention rollout. The agent uses IG.
- **AAS (Attribution Alignment Score):** share of positive IG mass that falls inside the spans the
  explanation cites. Below `tau_aas` the agent sends feedback (influential words not cited, cited
  words that did not matter) and asks for a revision, up to 2 rounds, keeping the best.
- **LLM:** local Ollama (`qwen2.5:7b`). No paid API calls.
- **Agent:** LangGraph. The `check_retrieval` node can send a weak retrieval through one `replan`
  (alternative queries) and then abstain; it is **off by default** because it hurt (see Results).
  Tools are served by **MCP servers** (`policy-search`, `explainer`, and `appeal` for templates and
  Ombudsman deadline arithmetic) over stdio; `--mcp` makes the agent call them, the default calls
  the same functions in-process.
- **Appeal draft:** the LLM writes facts, reasoning and relief as JSON. Every quoted clause text must
  appear verbatim in the retrieved clause, numbers must come from the letter, reason or clause,
  missing letter details become `[placeholders]`, one retry, then a fixed template. It is never
  sent anywhere and always carries "decision support, not legal advice".

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
Hugging Face models. Put a checkpoint in `models/matcher_v3_s42` to use the fine-tuned matcher.

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
`python -m claimlens.mcp_servers.explainer_server`, `python -m claimlens.mcp_servers.appeal_server` (test with MCP Inspector).

**Evaluation:** `python -m eval.run_all` writes everything in `eval/results/`. Per-case agent
results are cached in `eval/results/e2e/`; with the cache it took about 5 minutes, and the agent run
on all 30 cases took about 14 minutes. `python -m eval.run_all --summary` only rebuilds the tables.
**Tests:** `pytest -q -m "not model"` (no GPU) and `pytest -q -m model` (needs models and Ollama).

## Results

All numbers come from `eval/results/`. **All data is synthetic and small: 206 pairs and 30 cases,
generated by the local LLM.** Labels were audited in Phase 2b (below): **57 of the 206 pairs are
owner-verified** (`verified: true`; the owner confirmed an AI-assisted review of the labels, see
`data/QA.md`), and the rest are as generated. Only 5 cases were hand-checked, not by the owner
(3 sound, 1 questionable, 1 wrong; see `data/QA.md`). Phase 1 results are at tag `phase1-presented`.

**Split.** No third policy PDF was available, so test is whole held-out section groups of the same
two policies, not an unseen policy. Splits are frozen: corrected labels stay in their split.

| Split | Pairs | Cases |
|---|---:|---:|
| Train | 114 | 0 |
| Validation | 52 (+10 hard negatives) | 17 |
| Test | 40 | 13 |

### Label audit (R1a, `audit_v3.md`)

A local LLM judge (`qwen2.5:7b`, the same family as the pair generator) re-labelled all 206 pairs
twice. It flagged 99 (48%): 84 of the 103 generated positives and 15 of the 103 negatives. The owner
reviewed 37 queued pairs; 13 of those 37 labels changed. Judge labels are `source: "llm_audit"` and
never `verified`; no judge label replaced a generated label. Agreement with the owner's earlier 20
verified pairs (all test split, not random) is only a sanity check at n = 20.

### Matcher (R1b, `matcher_v3.md`)

Test, n = 40. Old labels = as generated; new labels = after the owner review. Thresholds are re-picked
on validation. The Phase 2 model was trained on the old labels; the v3 models on the new labels plus 60
hard negatives (judge-confirmed, not verified). 95% bootstrap CIs, 1000 resamples.

| System | Old acc | Old AUC | New acc | New F1 | New AUC |
|---|---:|---:|---:|---:|---:|
| BM25 score (baseline) | 0.575 | 0.618 | 0.525 | 0.424 | 0.552 |
| Pretrained cross-encoder (baseline) | 0.600 | 0.755 | 0.450 | 0.500 | 0.632 |
| Phase 2 fine-tuned MiniLM | 0.650 | 0.772 | 0.475 | 0.533 | 0.648 |
| v3 MiniLM, seed 42 (the agent's model) | 0.650 | 0.760 | 0.600 | 0.529 | 0.640 |
| v3 MiniLM, seed 43 | 0.650 | 0.762 | 0.650 | 0.562 | 0.646 |
| v3 MiniLM, seed 44 | 0.625 | 0.767 | 0.675 | 0.606 | 0.654 |

Mean of the three v3 seeds on new labels: accuracy 0.642 (SD 0.038, 95% CI [0.500, 0.775]), AUC
0.647 (CI [0.460, 0.822]). Paired difference to the pretrained baseline: accuracy +0.192 (CI
[+0.050, +0.342], excludes 0), F1 +0.066 (CI [-0.060, +0.182]) and AUC +0.015 (CI [-0.033, +0.063]),
both including 0. **Only accuracy improves; ranking quality (AUC) does not.** The drop of the
baselines from old to new labels shows how much the test result depends on 6 changed test labels.
On the 20 judge-confirmed hard negatives in the test groups the v3 seeds score 0.65 to 0.70 and the
pretrained model 0.40, but **BM25 scores 0.75**. The legal-bert, InLegalBERT and zero-shot LLM
baselines are in `matcher_v2.md`.

### Attribution faithfulness

From `eval/results/faithfulness_v2.md` (the v3 seed-42 matcher, 30 cases): change in the matcher's
logit. Higher comprehensiveness is better; lower sufficiency is better.

| Method | Compr. k=10 | k=20 | k=50 | Suff. k=10 | k=20 | k=50 |
|---|---:|---:|---:|---:|---:|---:|
| Integrated Gradients | 4.050 | 5.215 | 6.730 | -1.343 | -1.688 | -1.716 |
| SHAP | 5.550 | 7.282 | 9.384 | -1.233 | -1.977 | -2.027 |
| Attention rollout | 1.534 | 2.740 | 6.521 | 4.799 | 2.530 | -0.257 |
| Random words (baseline) | 0.051 | 0.118 | 0.803 | 8.436 | 7.220 | 3.600 |

All three methods beat random words, and SHAP scores best on comprehensiveness. They agree only
weakly: word-level Spearman between IG and SHAP is 0.268 (SD 0.136), and between IG's pad and
zero-embedding baselines 0.336 (SD 0.128).

### Reflection and verdicts (30 cases, `summary.md`, `tau.json`, `e2e.json`)

| Component | Result |
|---|---|
| Mean AAS, initial → after reflection (30 cases) | 0.180 → 0.212 (improved in 8, unchanged in 22, worse in 0) |
| Mean AAS of random spans of the same size | 0.145 |
| Test only, 13 cases, `tau_aas` = 0.4 | 0.199 → 0.238 (improved in 4); random spans 0.161 |
| Verdict thresholds (tuned on 17 validation cases) | `tau_high` 0.46, `tau_low` 0.2 |
| Verdict agreement on test, tuned / default 0.7-0.3 / always SUPPORTED | 0.38 (5 of 13) / 0.38 / 0.46 |
| Verdict agreement on validation, tuned / always SUPPORTED | 0.82 / 0.71 |
| End-to-end verdict agreement on all 30 cases | 19 of 30 (63.3%); validation 14 of 17, test 5 of 13 |
| Gold NOT_SUPPORTED cases (12): predicted SUPPORTED / INSUFFICIENT / NOT_SUPPORTED | 9 / 1 / 2 |
| ROC-AUC of the match probability, SUPPORTED vs NOT_SUPPORTED, test | 0.655 |
| Retrieval Recall@3 (30 cases): BM25 / hybrid / hybrid + re-rank | 0.900 / 0.867 / 0.900 |
| Gold clause in top 3, agent run | 80.0% |

The owner check of 10 revised explanations is in `docs/reflection_check.md`, not yet marked.

### Re-plan on weak retrieval (R3, `replan_v3.md`): negative result

`tau_retr` = 0.936 (validation only; Youden J 0.33, so a weak separator). With it the agent re-plans
25 of 35 cases and **rescues none** (gold clause in top 3: validation 15/17 and test 8/13, before and
after). It abstains on all 5 synthetic no-clause cases (0/5 without it), but also on 20 of the 30
real cases, and verdict agreement falls from 19/30 to 7/30. The rule is implemented and tested but
off by default (`tau_retr` is commented out in `configs/app.yaml`). That decision was made after
seeing results that include the test cases.

### Appeal drafts (R2, `appeal_v3.md`)

Mechanical checks only; no human rated the drafts. The agent predicted 4 non-SUPPORTED cases (2
NOT_SUPPORTED, 2 INSUFFICIENT_EVIDENCE): 4/4 passed the quote check on the first try, none used the
template. Because so few cases were NOT_SUPPORTED, an appeal was also drafted for all 12 gold
NOT_SUPPORTED cases with the verdict forced: 10/12 first try, 1/12 after one retry, 1/12 template
fallback, 12/12 complete. This shows quotes are grounded and no figure is invented, not that the
legal argument is right.

**Ombudsman deadlines.** Rule 14(3)(a) and (b) of the Insurance Ombudsman Rules (representation to the
insurer first; complaint within one year of the insurer's final reply or after one month without a
reply) were read on 2026-11-13 from the Council for Insurance Ombudsmen's copy "as amended till
09.11.2023". **Later amendments were not checked**, the Rs 50 lakh limit and IRDAI turnaround figures
were not verified and are not used, and every draft tells the user to confirm with the Ombudsman office.

## Limitations

- **The agent does no better than a trivial baseline on test.** It answers SUPPORTED for 26 of 30
  cases, including 9 of the 12 NOT_SUPPORTED ones, and test agreement (0.38) is below always-SUPPORTED
  (0.46). A NOT_SUPPORTED case is a wrong rejection against a closely related benefit clause, so a
  high match score does not separate the two verdicts (test ROC-AUC 0.655).
- **Fine-tuning gives a small, uncertain gain.** Accuracy improves over the pretrained baseline
  (CI excludes 0), AUC and F1 do not, and BM25 beats the matcher on the hard negatives. The test
  set has 40 pairs; 6 of them changed label in the audit.
- **Label audit limits.** The judge and generator are the same model family. The owner reviewed 37
  queue pairs (13 changed); the 3 blank queue rows and about 60 other flagged pairs keep their
  generated labels. The owner confirmed an AI-assisted review of the first 20 pairs, 17 of which use
  Care Supreme (48-month pre-existing-disease wording), and some chunks are cut mid-clause (for
  example care_supreme-0031, 0038, 0042, 0043, 0064). `v2_008_pos` looks mislabelled. Re-chunking
  was out of scope.
- **The re-plan rule hurts and is off by default** (see Results).
- **The reflection gain is small and not stable.** There is no significance test, and two runs with
  the same seed in Phase 2 differed by about as much as the effect. AAS can be raised just by quoting
  top-attributed words, and it rests on IG, which depends on its baseline.
- **The `tau_aas` threshold has no effect here.** All values from 0.4 to 0.8 give the same mean
  final AAS; ties go to the lowest value.
- **Synthetic, noisy data.** Letters are short and generic. Hand-written cases and a third policy
  are not included. Train has 114 pairs over 57 clauses, so pairs are not independent.
- **CI covers lint, unit tests and the Docker build** (no GPU, Ollama or models). Model-dependent
  tests run locally.
- The two policy PDFs come from third-party mirrors; check the UIN and version against the insurer.
- Appeal drafts are decision support, not legal advice. The Ombudsman rule text is checked only as
  described above.

## Repository layout

```
claimlens/   schemas.py  core.py  rag/  matcher/  explain/  llm/  agent/  appeal/  mcp_servers/  data/  utils/
api/main.py  app/ (streamlit_app.py and views)   configs/app.yaml   .streamlit/config.toml
data/        policies/  cases/  cases_noclause/  pairs_v2.jsonl  hard_negatives*.jsonl  splits.json
             label_audit.csv  review_queue.csv  verify_pairs.csv  QA.md
eval/        run_*.py  train_matcher.py  tune_*.py  results/
docs/        PROGRESS.md (build log, bugs, deviations)  reflection_check.md
tests/       .github/workflows/blank.yml   Dockerfile   docker-compose.yml   SPEC.md   CLAUDE.md
```

Deferred work (planner agent, PostgreSQL, more MCP servers and others) is listed in `SPEC.md` 13.6.
