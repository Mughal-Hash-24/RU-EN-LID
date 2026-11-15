# Evaluation report

Data: `data/cache/comi-lingua-lid/roman/test.csv` (15207 scored tokens)

## Overall

| Metric | Value |
|---|---|
| Token accuracy | 0.9651 |
| Macro F1 | 0.9405 |
| CMI (gold labels, mean per sentence) | 25.54 |
| Errors | 531 |
| Ignored (no gold label) | 252 |

## Per category

| Category | Tokens | Accuracy | Macro F1 |
|---|---|---|---|
| ambiguous | 687 | 0.8981 | 0.8557 |
| both | 27 | 0.9259 | 0.9206 |
| standard | 10670 | 0.9814 | 0.9222 |
| unknown | 3671 | 0.9292 | 0.9281 |
| variant | 152 | 0.9934 | 0.4983 |

## Error share by cause

| Cause | Share of all errors |
|---|---|
| ambiguous | 13.18% |
| both | 0.38% |
| standard | 37.29% |
| unknown | 48.96% |
| variant | 0.19% |

## Confidence gate and Stage 3

| Measure | Value |
|---|---|
| Coverage (accepted by Stage 2) | 1.0000 |
| Accuracy on accepted words | 0.9651 |
| Deferred words | 0 |
| Accuracy on deferred words, Stage 2 label | n/a |
| Accuracy on deferred words, after Stage 3 | n/a |
| Stage 3 fixes / breaks | 0 / 0 |

| Stage-3 change | Changed | Fixed | Broke |
|---|---|---|---|

| Stage-1 category | Tokens | Deferred | Stage-2 acc. | Final acc. |
|---|---|---|---|---|
| ambiguous | 687 | 0.0000 | 0.8981 | 0.8981 |
| both | 27 | 0.0000 | 0.9259 | 0.9259 |
| standard | 10670 | 0.0000 | 0.9814 | 0.9814 |
| unknown | 3671 | 0.0000 | 0.9292 | 0.9292 |
| variant | 152 | 0.0000 | 0.9934 | 0.9934 |
