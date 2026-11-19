# Matcher comparison (Phase 2)

All data is synthetic. Split: FALLBACK: held-out section groups (no third policy PDF). Thresholds are picked on validation only. Zero-shot LLM unusable answers: 0 (scored 0.5).

| System | Threshold | Val acc | Val F1 | Val AUC | Test acc | Test F1 | Test AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 score (baseline) | 15.359 | 0.731 | 0.741 | 0.744 | 0.600 | 0.619 | 0.618 |
| Phase 1 pretrained cross-encoder (baseline) | 0.024 | 0.846 | 0.862 | 0.834 | 0.600 | 0.680 | 0.755 |
| zero-shot Ollama LLM (baseline) | 0.050 | 0.500 | 0.667 | 0.627 | 0.500 | 0.667 | 0.799 |
| fine-tuned legal-bert | 0.496 | 0.692 | 0.667 | 0.620 | 0.525 | 0.537 | 0.412 |
| fine-tuned InLegalBERT | 0.519 | 0.750 | 0.711 | 0.664 | 0.525 | 0.486 | 0.560 |
| fine-tuned ms-marco-MiniLM | 0.132 | 0.808 | 0.833 | 0.833 | 0.625 | 0.706 | 0.772 |

n: validation 52, test 40. Test sets this small make small differences unreliable.

Best fine-tuned model on validation (the agent's checkpoint): fine-tuned ms-marco-MiniLM.
Best system overall on validation F1: Phase 1 pretrained cross-encoder (baseline).
The fine-tuned model does NOT beat the best baseline on validation.

## Owner-verified subset

PENDING: 0 pairs verified so far.
