# Appeal drafter (R2)

Synthetic cases (30), local Ollama `qwen2.5:7b`. Agent results are the cached `eval/results/e2e/` run (the 30-case run with the thresholds in `configs/app.yaml`; delete `eval/results/appeal/` and rerun if the verdict mix changes).

Verdicts in the cached run: INSUFFICIENT_EVIDENCE 2, NOT_SUPPORTED 2, SUPPORTED 26. Drafts are made for NOT_SUPPORTED (appeal) and INSUFFICIENT_EVIDENCE (clarification request) only.

| Measure | Value |
|---|---:|
| Drafts produced | 4 |
| Quotes pass on the first try (LLM draft, 1 attempt) | 4/4 (100%) |
| Pass after one retry | 0/4 (0%) |
| Fall back to the template (`source: template`) | 0/4 (0%) |
| NOT_SUPPORTED cases with a complete draft | 2/2 (100%) |
| SUPPORTED cases that got a draft (should be 0) | 0 |

A draft is complete when facts, at least one quote, reasoning, relief, checklist, deadline note and disclaimer are all present and every quote is re-checked verbatim against the clause text (whitespace-normalised, case-sensitive). Template fallbacks are counted as complete only if they carry a verified quote.

## Stress test on gold NOT_SUPPORTED cases (verdict forced)

With only a few predicted NOT_SUPPORTED cases above, an appeal was also drafted for all 12 cases whose gold verdict is NOT_SUPPORTED, using the cached retrieval and explanation but forcing the verdict. This tests the drafter, not the agent's verdicts.

| Measure | Value |
|---|---:|
| Quotes pass on the first try | 10/12 (83%) |
| Pass after one retry | 1/12 (8%) |
| Fall back to the template | 1/12 (8%) |
| Complete draft | 12/12 (100%) |

The check is mechanical: it shows quotes are grounded and no figure is invented, not that the legal argument is right. No human has rated these drafts.
