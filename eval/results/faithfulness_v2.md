# Attribution faithfulness (Phase 2)

Model `matcher_v3_s42`, 30 synthetic cases, each on its gold reason and gold clause. Change in logit after deleting words; higher comprehensiveness is better, lower sufficiency is better. No significance test.

| Method | Compr. k=10 | Compr. k=20 | Compr. k=50 | Suff. k=10 | Suff. k=20 | Suff. k=50 |
|---|---:|---:|---:|---:|---:|---:|
| Integrated Gradients | 4.050 | 5.215 | 6.730 | -1.343 | -1.688 | -1.716 |
| SHAP | 5.550 | 7.282 | 9.384 | -1.233 | -1.977 | -2.027 |
| Attention rollout | 1.534 | 2.740 | 6.521 | 4.799 | 2.530 | -0.257 |
| Random words (baseline) | 0.051 | 0.118 | 0.803 | 8.436 | 7.220 | 3.600 |

## Agreement between methods (word-level Spearman, per case)

| Comparison | Mean | SD | Cases |
|---|---:|---:|---:|
| IG vs SHAP | 0.268 | 0.136 | 30 |
| IG pad baseline vs zero-embedding baseline | 0.336 | 0.128 | 30 |

Time over all cases (s): Integrated Gradients 7.5, SHAP 224.9, Attention rollout 1.0.
