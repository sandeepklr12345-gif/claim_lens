# CLAUDE.md: ClaimLens

> v2 for the Phase 2 build. Details live in SPEC.md.

## Project

ClaimLens is an explainable agent that audits health-insurance claim rejections (India): rejection letter + policy in; verdict, attributed evidence, a faithfulness-checked explanation and an appeal draft out. Explainable AI mini project.

**Phase 1 is complete and was presented on 5 Nov 2026.** **Current phase: Phase 2, the 3-day cut in SPEC.md section 13** (milestones P1 to P5). Do not build anything listed in SPEC.md 13.6. Keep following the design rules in SPEC.md 12.1.

**Hard stop: Sun 8 Nov 2026, 22:00 IST.** Choose the simplest thing that meets the acceptance criteria in SPEC.md section 13. Apply the cut order in SPEC.md 13.6 without asking.

## Environment

- Windows, PowerShell. NVIDIA RTX 4060 (CUDA). Python 3.11.
- App LLM: local Ollama at `http://localhost:11434`. No paid API calls.
- Repo: `claim_lens` on GitHub (already exists; do not recreate).

## How to work

1. Read this file, then only the SPEC.md section for the current milestone (`Select-String -Path SPEC.md -Pattern "^## "` lists sections).
2. One milestone at a time, in order. Core logic in `claimlens/` as plain tested functions; MCP servers, LangGraph nodes, the API and the UI are thin wrappers.
3. After each milestone: tests pass, append to `docs/PROGRESS.md`, commit, push, verify, check the GitHub Actions run with `gh run list --limit 1` (never report CI green unless it shows success; fix red CI within 15 minutes or log it), continue.
4. If something blocks for more than 30 minutes, take the fallback in SPEC.md 13.3 or 13.6, log it, move on.
5. Anything only the user can do (supply the third policy PDF, verify 20 pairs, fill gold clause ids) is prepared by you, then you stop and ask for it. Never fill human verification yourself.

## Commands (PowerShell)

```
setup:    py -3.11 -m venv .venv ; .\.venv\Scripts\Activate.ps1 ; pip install -r requirements.txt
torch:    install the CUDA build using the command from pytorch.org's selector (Windows, pip, CUDA)
gpu check: python -c "import torch; print(torch.cuda.is_available())"
ollama:   ollama pull qwen2.5:7b      (confirm the model in .env)
tests:    pytest -q -m "not model"      (local model tests: pytest -q -m model)
lint:     ruff check .
ci:       gh run list --limit 1
search:   python -m claimlens.rag.search "waiting period" --doc <id>
case:     python -m claimlens.agent.run --case data\cases\case_001.json
mcp:      python -m claimlens.mcp_servers.policy_server   |   python -m claimlens.mcp_servers.explainer_server
ui:       streamlit run app\streamlit_app.py
eval:     python -m eval.run_all
train:    python -m eval.train_matcher --base <model-id-or-path> --out models\matcher_v2
api:      uvicorn api.main:app --reload
docker:   docker compose up --build
```

If `Activate.ps1` is blocked, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Conventions

- Type hints; pydantic models in `claimlens/schemas.py` for every tool and API input/output; `logging`, not `print`.
- Use `pathlib` for paths (Windows-safe). Read and write files as UTF-8.
- Config in `configs/app.yaml` and `.env` (git-ignored): `OLLAMA_HOST`, `OLLAMA_MODEL`.
- Seeds fixed at 42.
- CI is GitHub Actions. Mark tests that need torch, models, Ollama or a GPU with `@pytest.mark.model`. Keep `requirements-ci.txt` light.
- New dependencies allowed for Phase 2 only: `shap`, `fastapi`, `uvicorn`, `httpx`, `pytest-cov`. Anything else needs asking.
- Model checkpoints go in `models/` (git-ignored). Never commit them.

## Integrity rules (non-negotiable)

- Never invent data, metrics, results or citations. Every number in the README comes from `eval/results/`.
- Report negative results as they are. If the fine-tuned matcher does not beat a baseline, say so.
- Label all synthetic data `source: "synthetic"`. A pair is `verified: true` only when the user checked it.
- Mark stubs and fallbacks in code comments and `docs/PROGRESS.md`.
- Never say something works unless it was run in this session.
- Tune thresholds on validation data only; report once on test.

## Usage hygiene (Claude Code Pro limits)

- Use Sonnet unless a task is stuck.
- Never print large files, PDFs or logs; use `Get-Content -Tail 20` or small scripts.
- Long jobs (data generation, fine-tuning, SHAP, evaluation) write to `logs\` and show only the tail.
- Keep messages short; code over prose. Don't re-read files already in context.

## Safety

- No real personal data in the repo. `.env`, `logs/`, `runs/`, `models/` and `data/private/` are git-ignored.
- Every verdict and appeal draft shows "decision support, not legal advice".

## Repo layout

```
claimlens/  schemas.py  rag/  matcher/  explain/  llm/  agent/  mcp_servers/  utils/
api/main.py
app/streamlit_app.py
configs/app.yaml
data/  policies/  cases/  cases_extra/  pairs.jsonl  pairs_v2.jsonl  QA.md
eval/  run_*.py  train_matcher.py  results/
docs/PROGRESS.md  docs/reflection_check.md
tests/
.github/workflows/
Dockerfile  docker-compose.yml
requirements.txt  requirements-ci.txt  pytest.ini
```