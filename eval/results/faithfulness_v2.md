# Attribution faithfulness (Phase 2)

Model `matcher_v2`, 30 synthetic cases, each on its gold reason and gold clause. Change in logit after deleting words; higher comprehensiveness is better, lower sufficiency is better. No significance test.

| Method | Compr. k=10 | Compr. k=20 | Compr. k=50 | Suff. k=10 | Suff. k=20 | Suff. k=50 |
|---|---:|---:|---:|---:|---:|---:|
| Integrated Gradients | 4.482 | 5.880 | 8.347 | 1.186 | 0.032 | -0.663 |
| SHAP | 6.705 | 9.180 | 12.392 | -0.507 | -1.350 | -1.588 |
| Attention rollout | 1.977 | 3.259 | 8.377 | 6.822 | 3.885 | 0.151 |
| Random words (baseline) | 0.098 | 0.197 | 1.119 | 11.966 | 9.970 | 4.803 |

## Agreement between methods (word-level Spearman, per case)

| Comparison | Mean | SD | Cases |
|---|---:|---:|---:|
| IG vs SHAP | 0.247 | 0.150 | 30 |
| IG pad baseline vs zero-embedding baseline | 0.394 | 0.132 | 30 |

Time over all cases (s): Integrated Gradients 12.7, SHAP 348.1, Attention rollout 2.2.
