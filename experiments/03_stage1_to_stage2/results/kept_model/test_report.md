# Evaluation report

Data: `data/cache/comi-lingua-lid/roman/test.csv` (15207 scored tokens)

## Overall

| Metric | Value |
|---|---|
| Token accuracy | 0.9651 |
| Macro F1 | 0.9405 |
| CMI (gold labels, mean per sentence) | 25.54 |
| Errors | 530 |
| Ignored (no gold label) | 252 |

## Per category

| Category | Tokens | Accuracy | Macro F1 |
|---|---|---|---|
| ambiguous | 687 | 0.8981 | 0.8557 |
| both | 27 | 0.9259 | 0.9206 |
| standard | 10670 | 0.9815 | 0.9223 |
| unknown | 3671 | 0.9292 | 0.9281 |
| variant | 152 | 0.9934 | 0.4983 |

## Error share by cause

| Cause | Share of all errors |
|---|---|
| ambiguous | 13.21% |
| both | 0.38% |
| standard | 37.17% |
| unknown | 49.06% |
| variant | 0.19% |

## Confidence gate and Stage 3

| Measure | Value |
|---|---|
| Coverage (accepted by Stage 2) | 0.9893 |
| Accuracy on accepted words | 0.9699 |
| Deferred words | 163 |
| Accuracy on deferred words, Stage 2 label | 0.5215 |
| Accuracy on deferred words, after Stage 3 | 0.5276 |
| Stage 3 fixes / breaks | 2 / 1 |

| Stage-3 change | Changed | Fixed | Broke |
|---|---|---|---|
| urdu-lexicon-prior | 3 | 2 | 1 |

| Stage-1 category | Tokens | Deferred | Stage-2 acc. | Final acc. |
|---|---|---|---|---|
| ambiguous | 687 | 0.0378 | 0.8981 | 0.8981 |
| both | 27 | 0.0000 | 0.9259 | 0.9259 |
| standard | 10670 | 0.0051 | 0.9814 | 0.9815 |
| unknown | 3671 | 0.0226 | 0.9292 | 0.9292 |
| variant | 152 | 0.0000 | 0.9934 | 0.9934 |
