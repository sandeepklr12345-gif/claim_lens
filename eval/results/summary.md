# Evaluation summary

All data is **synthetic** and small (20 cases, 80 pairs); labels are by construction and only 5 were hand-checked (see `data/QA.md`). Treat numbers as indicative.

## Retrieval (query = case reason, 20 cases)

| Method | Recall@3 | MRR@10 |
|---|---|---|
| bm25 | 0.850 | 0.658 |
| hybrid | 0.800 | 0.694 |
| hybrid+rerank | 0.900 | 0.691 |

## Matcher (pretrained cross-encoder, no fine-tuning, 80 pairs)

Model `cross-encoder/ms-marco-MiniLM-L-6-v2`. Threshold picked on the dev half (0.226), reported on the held-out half.

| Split | n | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|
| dev | 40 | 0.600 | 0.680 | 0.632 |
| test | 40 | 0.575 | 0.653 | 0.627 |

## Verdict thresholds and end-to-end agreement

Tuned on a stratified dev split of 10 cases (SPEC asked for the first 10, but those are all SUPPORTED): tau_high = 0.8, tau_low = 0.4.

| Thresholds | dev (10) | held-out (10) | all 20 |
|---|---|---|---|
| default 0.7 / 0.3 | 0.60 | 0.60 | 0.60 |
| tuned 0.8 / 0.4 | 0.70 | 0.50 | 0.60 |

End-to-end (letter in, LLM-extracted reason, agent verdict) vs `gold_verdict` on all 20 cases: **agreement 13/20 (65.0%)**, abstain rate 5.0%, gold clause in top-3 70.0%. ROC-AUC of the match probability for SUPPORTED vs NOT_SUPPORTED: 0.615. Baseline: always answering SUPPORTED agrees on 60.0% of cases (12/20); compare the agreement above with this. Thresholds were tuned on 10 of these 20 cases, so the all-20 number is optimistic. The reason is re-extracted by the LLM on every run, so these verdicts can differ slightly from the `tau.json` run.

Confusion (gold -> predicted):

- SUPPORTED: SUPPORTED 11, INSUFFICIENT_EVIDENCE 1
- NOT_SUPPORTED: SUPPORTED 6, NOT_SUPPORTED 2

## Faithfulness: AAS before vs after reflection

| | Mean AAS |
|---|---|
| initial explanation | 0.200 |
| final (after reflection, best kept) | 0.233 |
| random spans of the same size (chance) | 0.171 |

Over 20 cases: improved 4, unchanged 16, worse 0 (best kept, so worse should be 0). Reflection was triggered (initial AAS < 0.6) in 20 cases.
On those: mean AAS 0.200 -> 0.233.

Caveat: AAS can be raised by quoting top-attributed words; compare with the random-span baseline. Small synthetic set (20 cases); no significance test.

Raw results: `matcher.json`, `retrieval.json`, `tau.json`, `e2e.json`, `faithfulness.json`, `e2e/` (per case).
