# Matcher comparison (Phase 2)

All data is synthetic. Split: FALLBACK: held-out section groups (no third policy PDF). Thresholds are picked on validation only. Zero-shot LLM unusable answers: 0 (scored 0.5).

| System | Threshold | Val acc | Val F1 | Val AUC | Test acc | Test F1 | Test AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 score (baseline) | 15.359 | 0.731 | 0.741 | 0.744 | 0.550 | 0.526 | 0.547 |
| Phase 1 pretrained cross-encoder (baseline) | 0.024 | 0.846 | 0.862 | 0.834 | 0.500 | 0.565 | 0.685 |
| zero-shot Ollama LLM (baseline) | 0.050 | 0.500 | 0.667 | 0.627 | 0.400 | 0.571 | 0.824 |
| fine-tuned legal-bert | 0.496 | 0.692 | 0.667 | 0.620 | 0.525 | 0.486 | 0.451 |
| fine-tuned InLegalBERT | 0.519 | 0.750 | 0.711 | 0.664 | 0.575 | 0.485 | 0.617 |
| fine-tuned ms-marco-MiniLM | 0.132 | 0.808 | 0.833 | 0.833 | 0.525 | 0.596 | 0.698 |

n: validation 52, test 40. Test sets this small make small differences unreliable.

Best fine-tuned model on validation (the agent's checkpoint): fine-tuned ms-marco-MiniLM.
Best system overall on validation F1: Phase 1 pretrained cross-encoder (baseline).
The fine-tuned model does NOT beat the best baseline on validation.

## Owner-verified subset

n = 20.

| System | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| BM25 score (baseline) | 0.550 | 0.526 | 0.667 |
| Phase 1 pretrained cross-encoder (baseline) | 0.450 | 0.476 | 0.679 |
| zero-shot Ollama LLM (baseline) | 0.300 | 0.462 | 0.881 |
| fine-tuned legal-bert | 0.500 | 0.375 | 0.405 |
| fine-tuned InLegalBERT | 0.600 | 0.333 | 0.488 |
| fine-tuned ms-marco-MiniLM | 0.450 | 0.522 | 0.679 |
