# Matcher re-evaluation with cleaned labels (R1b)

All data is synthetic. Split: frozen from Phase 2 (held-out section groups, no third policy). Thresholds are picked on validation only (validation includes the train/val-group hard negatives); the test set is never used to choose anything. The agent's checkpoint is the seed-42 run, fixed in advance.

Test pairs: 40. Labels that differ between old and new in the test split: 6. Old labels = what the generator produced; new labels = owner-reviewed where `verified`, otherwise as listed in `docs/PROGRESS.md` (judge labels are marked `llm_audit`, not verified).

## Test results, old labels and new labels side by side

| System | Old: acc | Old: F1 | Old: AUC | New: acc | New: F1 | New: AUC |
|---|---:|---:|---:|---:|---:|---:|
| BM25 score (baseline) | 0.575 | 0.564 | 0.618 | 0.525 | 0.424 | 0.552 |
| Phase 1 pretrained cross-encoder (baseline) | 0.600 | 0.680 | 0.755 | 0.450 | 0.500 | 0.632 |
| Phase 2 fine-tuned MiniLM (original labels) | 0.650 | 0.696 | 0.772 | 0.475 | 0.533 | 0.648 |
| v3 fine-tuned MiniLM seed 42 | 0.650 | 0.708 | 0.760 | 0.600 | 0.529 | 0.640 |
| v3 fine-tuned MiniLM seed 43 | 0.650 | 0.731 | 0.762 | 0.650 | 0.562 | 0.646 |
| v3 fine-tuned MiniLM seed 44 | 0.625 | 0.681 | 0.767 | 0.675 | 0.606 | 0.654 |

Thresholds are re-picked on the validation split under each label set. The Phase 2 model was trained on the original labels; the v3 models on the cleaned labels plus hard negatives.

## Seeds 42, 43, 44: mean, standard deviation, 95% bootstrap intervals (new labels)

| System | Metric | Mean | SD across seeds | 95% bootstrap CI |
|---|---|---:|---:|---|
| v3 fine-tuned MiniLM (mean of 3 seeds) | accuracy | 0.642 | 0.038 | [0.500, 0.775] |
| v3 fine-tuned MiniLM (mean of 3 seeds) | f1 | 0.566 | 0.038 | [0.348, 0.739] |
| v3 fine-tuned MiniLM (mean of 3 seeds) | roc_auc | 0.647 | 0.007 | [0.460, 0.822] |
| Phase 1 pretrained cross-encoder (baseline) | accuracy | 0.450 | n/a | [0.300, 0.600] |
| Phase 1 pretrained cross-encoder (baseline) | f1 | 0.500 | n/a | [0.300, 0.655] |
| Phase 1 pretrained cross-encoder (baseline) | roc_auc | 0.632 | n/a | [0.443, 0.809] |
| Phase 2 fine-tuned MiniLM (original labels) | accuracy | 0.475 | n/a | [0.325, 0.625] |
| Phase 2 fine-tuned MiniLM (original labels) | f1 | 0.533 | n/a | [0.341, 0.681] |
| Phase 2 fine-tuned MiniLM (original labels) | roc_auc | 0.648 | n/a | [0.457, 0.824] |

## Paired bootstrap: v3 (mean of seeds) minus the pretrained baseline

Same resamples for both systems; the interval is for the difference.

| Metric | Difference | 95% CI | CI excludes 0 |
|---|---:|---|---|
| accuracy | +0.192 | [+0.050, +0.342] | yes |
| f1 | +0.066 | [-0.060, +0.182] | no |
| roc_auc | +0.015 | [-0.033, +0.063] | no |

## Validation (new labels, includes val-group hard negatives)

| System | Threshold | Acc | F1 | AUC |
|---|---:|---:|---:|---:|
| BM25 score (baseline) | 16.861 | 0.758 | 0.706 | 0.774 |
| Phase 1 pretrained cross-encoder (baseline) | 0.024 | 0.758 | 0.754 | 0.806 |
| Phase 2 fine-tuned MiniLM (original labels) | 0.132 | 0.710 | 0.719 | 0.809 |
| v3 fine-tuned MiniLM seed 42 | 0.534 | 0.823 | 0.756 | 0.827 |
| v3 fine-tuned MiniLM seed 43 | 0.563 | 0.806 | 0.727 | 0.826 |
| v3 fine-tuned MiniLM seed 44 | 0.593 | 0.806 | 0.727 | 0.827 |

n: validation 52 pairs + 10 hard negatives.

## Hard-negative test file (n = 20, all label 0)

Built from test-group clauses, judge-confirmed (`llm_audit`, not verified). Accuracy = share scored below the validation threshold.

| System | Accuracy on hard negatives |
|---|---:|
| BM25 score (baseline) | 0.750 |
| Phase 1 pretrained cross-encoder (baseline) | 0.400 |
| Phase 2 fine-tuned MiniLM (original labels) | 0.400 |
| v3 fine-tuned MiniLM seed 42 | 0.650 |
| v3 fine-tuned MiniLM seed 43 | 0.700 |
| v3 fine-tuned MiniLM seed 44 | 0.650 |

Test sets this small make differences unreliable: read the intervals, not the point estimates.
