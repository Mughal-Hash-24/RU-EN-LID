# Evaluation report

Data: `data/demo/seed_labels.csv` (51 scored tokens)

## Overall

| Metric | Value |
|---|---|
| Token accuracy | 0.8235 |
| Macro F1 | 0.7984 |
| CMI (gold labels, mean per sentence) | 22.36 |
| Errors | 9 |

## Per category

| Category | Tokens | Accuracy | Macro F1 |
|---|---|---|---|
| ambiguous | 8 | 0.0000 | 0.0000 |
| both | 3 | 0.6667 | 0.4000 |
| standard | 34 | 1.0000 | 1.0000 |
| variant | 6 | 1.0000 | 1.0000 |

## Error share by cause

| Cause | Share of all errors |
|---|---|
| ambiguous | 88.89% |
| both | 11.11% |

## Confidence gate and Stage 3

| Measure | Value |
|---|---|
| Coverage (accepted by Stage 2) | 0.6667 |
| Accuracy on accepted words | 1.0000 |
| Deferred words | 17 |
| Accuracy on deferred words, Stage 2 label | 0.4706 |
| Accuracy on deferred words, after Stage 3 | 0.4706 |
| Stage 3 fixes / breaks | 0 / 0 |

| Stage-3 change | Changed | Fixed | Broke |
|---|---|---|---|

| Stage-1 category | Tokens | Deferred | Stage-2 acc. | Final acc. |
|---|---|---|---|---|
| ambiguous | 8 | 1.0000 | 0.0000 | 0.0000 |
| both | 3 | 1.0000 | 0.6667 | 0.6667 |
| standard | 34 | 0.0000 | 1.0000 | 1.0000 |
| variant | 6 | 1.0000 | 1.0000 | 1.0000 |
