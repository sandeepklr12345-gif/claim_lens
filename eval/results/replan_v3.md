# Re-plan on low retrieval confidence (R3)

`tau_retr` = 0.936 (match probability of the best re-ranked clause), calibrated on validation cases only (`eval/results/tau_retr.json`). A case below it gets one re-plan (two LLM-written alternative queries, reciprocal-rank-fused with the original search, re-ranked by the matcher against the original reason); still below it, the agent returns INSUFFICIENT_EVIDENCE with `abstain_reason: "no clause above confidence threshold"`. Synthetic data, 30 cases + 5 no-clause cases; small samples.

## Gold clause in the top 3, before and after the re-plan

| Split | Cases | Re-planned | Before | After | Rescued | Lost |
|---|---:|---:|---:|---:|---:|---:|
| val | 17 | 12 | 15/17 (88%) | 15/17 (88%) | 0 | 0 |
| test | 13 | 8 | 8/13 (62%) | 8/13 (62%) | 0 | 0 |

## Abstention on the 5 no-clause cases

Reasons with no basis in the policy (`data/cases_noclause/`, synthetic, gold verdict INSUFFICIENT_EVIDENCE by construction).

| Measure | Value |
|---|---:|
| Abstained by the re-plan rule | 5/5 (100%) |
| Verdict INSUFFICIENT_EVIDENCE without the rule | 0/5 (0%) |
| Verdict INSUFFICIENT_EVIDENCE with the rule | 5/5 (100%) |

## Regression check on the 30 cases

Verdict agreement with the gold verdict: 19/30 (63%) without the rule, 7/30 (23%) with it. Cases whose verdict changed: 18. Real cases that abstained on the retrieval rule: 20.

| Case | Gold | Before | After | Final decision |
|---|---|---|---|---|
| case_002 | SUPPORTED | SUPPORTED | INSUFFICIENT_EVIDENCE | abstain |
| case_004 | SUPPORTED | SUPPORTED | INSUFFICIENT_EVIDENCE | abstain |
| case_005 | SUPPORTED | SUPPORTED | INSUFFICIENT_EVIDENCE | abstain |
| case_007 | SUPPORTED | SUPPORTED | INSUFFICIENT_EVIDENCE | abstain |
| case_008 | SUPPORTED | SUPPORTED | INSUFFICIENT_EVIDENCE | abstain |
| case_009 | SUPPORTED | SUPPORTED | INSUFFICIENT_EVIDENCE | abstain |
| case_010 | SUPPORTED | SUPPORTED | INSUFFICIENT_EVIDENCE | abstain |
| case_012 | SUPPORTED | SUPPORTED | INSUFFICIENT_EVIDENCE | abstain |
| case_015 | NOT_SUPPORTED | NOT_SUPPORTED | INSUFFICIENT_EVIDENCE | abstain |
| case_016 | NOT_SUPPORTED | SUPPORTED | INSUFFICIENT_EVIDENCE | abstain |
| case_017 | NOT_SUPPORTED | SUPPORTED | INSUFFICIENT_EVIDENCE | abstain |
| case_018 | NOT_SUPPORTED | NOT_SUPPORTED | INSUFFICIENT_EVIDENCE | abstain |
| case_019 | NOT_SUPPORTED | SUPPORTED | INSUFFICIENT_EVIDENCE | abstain |
| case_020 | NOT_SUPPORTED | SUPPORTED | INSUFFICIENT_EVIDENCE | abstain |
| case_025 | SUPPORTED | SUPPORTED | INSUFFICIENT_EVIDENCE | abstain |
| case_026 | SUPPORTED | SUPPORTED | INSUFFICIENT_EVIDENCE | abstain |
| case_027 | NOT_SUPPORTED | SUPPORTED | INSUFFICIENT_EVIDENCE | abstain |
| case_029 | NOT_SUPPORTED | SUPPORTED | INSUFFICIENT_EVIDENCE | abstain |

First-check decisions over all 35 cases: ok 10, replan 25.

Only validation cases were used to set `tau_retr`; the test rows above are reported once. The cached 30-case run is reused for cases whose first retrieval was confident (same code path), and rerun for cases that re-plan.
