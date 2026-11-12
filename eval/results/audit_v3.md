# Label audit (R1a)

An LLM judge (local Ollama `qwen2.5:7b`) re-labelled every pair twice (second pass with the reason and clause order swapped). Judge labels are `source: "llm_audit"`, never `verified`. The judge and the pair generator are the same model family, so their errors may be correlated; agreement with the generated label is not independent evidence.

Pairs judged: 206. Judge failures (unusable output): 0.

## Flag rates

A pair is flagged if either pass disagrees with the generated label or reports low confidence.

- Flagged overall: 99/206 (48%)
- Both passes agree with each other: 177/206 (86%)
- Both passes agree with each other and contradict the generated label: 63/206 (31%)

| Group | Pairs | Flagged |
|---|---:|---:|
| split = test | 40 | 19/40 (48%) |
| split = train | 114 | 57/114 (50%) |
| split = val | 52 | 23/52 (44%) |
| policy_id = care_supreme | 98 | 43/98 (44%) |
| policy_id = hdfc_optima_secure | 108 | 56/108 (52%) |
| generated_label = 0 | 103 | 15/103 (15%) |
| generated_label = 1 | 103 | 84/103 (82%) |

## Calibration against the owner-verified pairs

n = 20. With so few pairs this is only a sanity check, not an accuracy estimate. The 20 pairs are all from the test split and were chosen for verification, not at random. The owner labels came from an AI-assisted review that the owner confirmed.

| Measure | Agreement with owner label |
|---|---:|
| Judge pass 1 | 17/20 (85%) |
| Judge pass 2 (swapped order) | 16/20 (80%) |
| Judge combined (both passes agree; others undecided) | 14/15 (93%) decided |
| Generator's own label (for comparison) | 16/20 (80%) |

Of 4 generated labels the owner overturned, the audit flagged 4. It flagged 9 of the 20 verified pairs in total.

## Review queue for the owner

`data/review_queue.csv`: 40 rows (cap 40) = flagged pairs (cap 30) plus 10 random unflagged pairs. Owner-verified pairs are excluded. The random rows are not marked in the file; they are the rows whose `pair_id` is not flagged in `data/label_audit.csv`. Their error rate estimates how many errors the judge misses.
- Flagged pairs available to review (excluding verified): 90
- Queue rows that are flagged: 30
- Queue rows that are random unflagged: 10
- Queue rows where the two passes disagreed (blank judge_label): 0

## Judge confidence

| Confidence (lower of the two passes) | Pairs |
|---|---:|
| high | 197 |
| medium | 2 |
| low | 7 |
