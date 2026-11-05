# Evaluation summary

All data is **synthetic** and small (30 cases); labels are by construction and only 5 were hand-checked (see `data/QA.md`). Treat numbers as indicative. Phase 2 tables: `data_v2.md`, `matcher_v2.md`, `faithfulness_v2.md`, `reflection_v2.md`.

## Retrieval (query = case reason, 30 cases)

| Method | Recall@3 | MRR@10 |
|---|---|---|
| bm25 | 0.900 | 0.739 |
| hybrid | 0.867 | 0.763 |
| hybrid+rerank | 0.867 | 0.707 |

## Phase 1 matcher (pretrained cross-encoder, no fine-tuning, 80 pairs)

Model `cross-encoder/ms-marco-MiniLM-L-6-v2`. Threshold picked on the dev half (0.226), reported on the held-out half.

| Split | n | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|
| dev | 40 | 0.600 | 0.680 | 0.632 |
| test | 40 | 0.575 | 0.653 | 0.627 |

## Verdict thresholds and end-to-end agreement

Tuned on 17 validation cases: tau_high = 0.96, tau_low = 0.82. Reported once on 13 test cases.

| Thresholds | validation | test |
|---|---:|---:|
| default 0.7 / 0.3 | 0.65 | 0.46 |
| tuned 0.96 / 0.82 | 0.76 | 0.46 |
| always SUPPORTED (baseline) | 0.71 | 0.46 |

End-to-end (letter in, LLM-extracted reason, agent verdict) vs `gold_verdict`: **test 6/13, validation 13/17**, all 30 cases 19/30 (63.3%). Abstain rate 0.0%, gold clause in top-3 80.0%. ROC-AUC of the match probability for SUPPORTED vs NOT_SUPPORTED on test: 0.631. Baseline: always answering SUPPORTED agrees on 60.0% of all cases (18/30). The reason is re-extracted by the LLM on every run, so these verdicts can differ slightly from the `tau.json` run.

Confusion (gold -> predicted):

- SUPPORTED: SUPPORTED 17, NOT_SUPPORTED 1
- NOT_SUPPORTED: SUPPORTED 10, NOT_SUPPORTED 2

## Faithfulness: AAS before vs after reflection

| | Mean AAS |
|---|---|
| initial explanation | 0.165 |
| final (after reflection, best kept) | 0.215 |
| random spans of the same size (chance) | 0.166 |

Over 30 cases: improved 7, unchanged 23, worse 0 (best kept, so worse should be 0). Reflection was triggered (initial AAS < 0.6) in 28 cases.
On those: mean AAS 0.138 -> 0.191.

Caveat: AAS can be raised by quoting top-attributed words; compare with the random-span baseline. Small synthetic set; no significance test.

Raw results: `matcher.json`, `retrieval.json`, `tau.json`, `e2e.json`, `faithfulness.json`, `e2e/` (per case).
