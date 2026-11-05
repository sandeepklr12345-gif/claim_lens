# Reflection threshold (Phase 2)

Chosen on validation only: **tau_aas = 0.4**. Objective: highest mean final AAS on validation; ties go low. Replayed from one run recorded at tau_aas = 0.8, so every tau sees the same LLM outputs. All data is synthetic.

| tau (validation) | n | Mean AAS initial | Mean AAS final | Improved | Mean revisions |
|---:|---:|---:|---:|---:|---:|
| 0.4 | 17 | 0.157 | 0.230 | 6 | 1.94 |
| 0.5 | 17 | 0.157 | 0.230 | 6 | 1.94 |
| 0.6 | 17 | 0.157 | 0.230 | 6 | 2.00 |
| 0.7 | 17 | 0.157 | 0.230 | 6 | 2.00 |
| 0.8 | 17 | 0.157 | 0.230 | 6 | 2.00 |

Test, reported once at tau_aas = 0.4 (n = 13): mean AAS 0.176 -> 0.196, improved 1, mean revisions 1.69. Random spans of the same lengths score 0.130. Best of the revisions is kept, so final cannot fall below initial. No significance test.
