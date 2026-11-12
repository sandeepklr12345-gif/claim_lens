# Reflection threshold (Phase 2)

Chosen on validation only: **tau_aas = 0.4**. Objective: highest mean final AAS on validation; ties go low. Replayed from one run recorded at tau_aas = 0.8, so every tau sees the same LLM outputs. All data is synthetic.

| tau (validation) | n | Mean AAS initial | Mean AAS final | Improved | Mean revisions |
|---:|---:|---:|---:|---:|---:|
| 0.4 | 17 | 0.165 | 0.192 | 4 | 1.94 |
| 0.5 | 17 | 0.165 | 0.192 | 4 | 1.94 |
| 0.6 | 17 | 0.165 | 0.192 | 4 | 1.94 |
| 0.7 | 17 | 0.165 | 0.192 | 4 | 1.94 |
| 0.8 | 17 | 0.165 | 0.192 | 4 | 1.94 |

Test, reported once at tau_aas = 0.4 (n = 13): mean AAS 0.199 -> 0.238, improved 4, mean revisions 1.69. Random spans of the same lengths score 0.161. Best of the revisions is kept, so final cannot fall below initial. No significance test.
