# Appeal drafter (R2)

Synthetic cases (30), local Ollama `qwen2.5:7b`. Agent results are the cached `eval/results/e2e/` run (the thresholds in force when it was produced), so the verdict mix below is not the retrained R1b mix.

Verdicts in the cached run: NOT_SUPPORTED 3, SUPPORTED 27. Drafts are made for NOT_SUPPORTED (appeal) and INSUFFICIENT_EVIDENCE (clarification request) only.

| Measure | Value |
|---|---:|
| Drafts produced | 3 |
| Quotes pass on the first try (LLM draft, 1 attempt) | 3/3 (100%) |
| Pass after one retry | 0/3 (0%) |
| Fall back to the template (`source: template`) | 0/3 (0%) |
| NOT_SUPPORTED cases with a complete draft | 3/3 (100%) |
| SUPPORTED cases that got a draft (should be 0) | 0 |

A draft is complete when facts, at least one quote, reasoning, relief, checklist, deadline note and disclaimer are all present and every quote is re-checked verbatim against the clause text (whitespace-normalised, case-sensitive). Template fallbacks are counted as complete only if they carry a verified quote.

## Stress test on gold NOT_SUPPORTED cases (verdict forced)

With only a few predicted NOT_SUPPORTED cases above, an appeal was also drafted for all 12 cases whose gold verdict is NOT_SUPPORTED, using the cached retrieval and explanation but forcing the verdict. This tests the drafter, not the agent's verdicts.

| Measure | Value |
|---|---:|
| Quotes pass on the first try | 10/12 (83%) |
| Pass after one retry | 2/12 (17%) |
| Fall back to the template | 0/12 (0%) |
| Complete draft | 12/12 (100%) |

The check is mechanical: it shows quotes are grounded and no figure is invented, not that the legal argument is right. No human has rated these drafts.
