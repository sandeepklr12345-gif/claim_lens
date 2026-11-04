# CLAUDE.md: ClaimLens

> v1 for the overnight build. Details live in SPEC.md.

## Project

ClaimLens is an explainable agent that audits health-insurance claim rejections (India): rejection letter + policy in; verdict, attributed evidence, a faithfulness-checked explanation and an appeal draft out. Explainable AI mini project.

**Current phase: Phase 1**, a shorter version for the 5 Nov presentation. The full scope is Phase 2 (SPEC.md section 12), built afterwards. Follow the design rules in SPEC.md 12.1 so Phase 2 extends Phase 1, but do not build Phase 2 features now.

**Hard stop: Thu 5 Nov 2026, 05:00 IST.** Always choose the simplest thing that meets the acceptance criteria in SPEC.md section 3. Apply the cut line in SPEC.md section 2 without asking.

## Environment

- Windows, PowerShell. NVIDIA RTX 4060 (CUDA). Python 3.11.
- App LLM: local Ollama at `http://localhost:11434`. No paid API calls.
- Repo: `claim_lens` on GitHub.

## How to work

1. Read this file, then only the SPEC.md section for the current milestone (`Select-String -Path SPEC.md -Pattern "^## "` lists sections).
2. One milestone at a time, in order. Core logic in `claimlens/` as plain tested functions; MCP servers, LangGraph nodes and the UI are thin wrappers.
3. After each milestone: tests pass, append to `docs/PROGRESS.md`, commit, push, verify, check the GitHub Actions run with `gh run list --limit 1` (never report CI green unless it shows success; fix red CI within 15 minutes or log it), continue.
4. If something blocks for more than 30 minutes, take the SPEC.md section 10 fallback, log it, move on.

## Commands (PowerShell)

```
setup:    py -3.11 -m venv .venv ; .\.venv\Scripts\Activate.ps1 ; pip install -r requirements.txt
torch:    install the CUDA build using the command from pytorch.org's selector (Windows, pip, CUDA)
gpu check: python -c "import torch; print(torch.cuda.is_available())"
ollama:   ollama pull qwen2.5:7b
tests:    pytest -q -m "not model"      (local model tests: pytest -q -m model)
lint:     ruff check .
ci:       gh run list --limit 1
search:   python -m claimlens.rag.search "waiting period" --doc <id>
case:     python -m claimlens.agent.run --case data\cases\case_001.json
mcp:      python -m claimlens.mcp_servers.policy_server   |   python -m claimlens.mcp_servers.explainer_server
ui:       streamlit run app\streamlit_app.py
eval:     python -m eval.run_all
```

If `Activate.ps1` is blocked, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Conventions

- Type hints; pydantic models in `claimlens/schemas.py` for every tool input/output; `logging`, not `print`.
- Use `pathlib` for paths (Windows-safe). Read and write files as UTF-8.
- Config in `configs/app.yaml` and `.env` (git-ignored): `OLLAMA_HOST`, `OLLAMA_MODEL`.
- Seeds fixed at 42.
- CI is GitHub Actions (SPEC.md 7.10). Mark tests that need torch, models, Ollama or a GPU with `@pytest.mark.model`. Keep `requirements-ci.txt` light. Run `gh auth refresh -s workflow` once before pushing `.github/`.
- No dependencies outside SPEC.md section 5 without asking.

## Integrity rules (non-negotiable)

- Never invent data, metrics, results or citations. Every number in the README comes from `eval/results/`.
- Label all synthetic data `source: "synthetic"`.
- Mark stubs and fallbacks in code comments and `docs/PROGRESS.md`.
- Never say something works unless it was run in this session.

## Usage hygiene (Claude Code Pro limits)

- Use Sonnet unless a task is stuck.
- Never print large files, PDFs or logs; use `Get-Content -Tail 20` or small scripts.
- Long jobs (data generation, evaluation) write to `logs\` and show only the tail.
- Keep messages short; code over prose. Don't re-read files already in context.

## Safety

- No real personal data in the repo. `.env`, `logs/`, `runs/` and `data/private/` are git-ignored.
- Every verdict and appeal draft shows "decision support, not legal advice".

## Repo layout

```
claimlens/  schemas.py  rag/  matcher/  explain/  llm/  agent/  mcp_servers/  utils/
app/streamlit_app.py
configs/app.yaml
data/  policies/  cases/  pairs.jsonl  QA.md
eval/  run_*.py  results/
docs/PROGRESS.md
tests/
.github/workflows/ci.yml
requirements.txt  requirements-ci.txt  pytest.ini
```
