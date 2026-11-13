# Matcher comparison (Phase 2)

All data is synthetic. Split: FALLBACK: held-out section groups (no third policy PDF). Thresholds are picked on validation only. Zero-shot LLM unusable answers: 0 (scored 0.5).

| System | Threshold | Val acc | Val F1 | Val AUC | Test acc | Test F1 | Test AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 score (baseline) | 15.359 | 0.731 | 0.731 | 0.740 | 0.550 | 0.500 | 0.552 |
| Phase 1 pretrained cross-encoder (baseline) | 0.024 | 0.808 | 0.821 | 0.826 | 0.450 | 0.500 | 0.632 |
| zero-shot Ollama LLM (baseline) | 0.050 | 0.462 | 0.632 | 0.617 | 0.350 | 0.519 | 0.897 |
| fine-tuned legal-bert | 0.357 | 0.462 | 0.632 | 0.579 | 0.300 | 0.462 | 0.429 |
| fine-tuned InLegalBERT | 0.519 | 0.788 | 0.744 | 0.731 | 0.575 | 0.452 | 0.574 |
| fine-tuned ms-marco-MiniLM | 0.132 | 0.769 | 0.793 | 0.823 | 0.475 | 0.533 | 0.648 |

n: validation 52, test 40. Test sets this small make small differences unreliable.

Best fine-tuned model on validation (the agent's checkpoint): fine-tuned ms-marco-MiniLM.
Best system overall on validation F1: Phase 1 pretrained cross-encoder (baseline).
The fine-tuned model does NOT beat the best baseline on validation.

## Owner-verified subset

n = 37.

| System | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| BM25 score (baseline) | 0.676 | 0.700 | 0.652 |
| Phase 1 pretrained cross-encoder (baseline) | 0.568 | 0.652 | 0.673 |
| zero-shot Ollama LLM (baseline) | 0.432 | 0.604 | 0.714 |
| fine-tuned legal-bert | 0.432 | 0.604 | 0.509 |
| fine-tuned InLegalBERT | 0.676 | 0.600 | 0.655 |
| fine-tuned ms-marco-MiniLM | 0.568 | 0.667 | 0.693 |
