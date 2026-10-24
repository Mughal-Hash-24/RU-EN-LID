# Evaluation report

Data: `/home/ibtasaam/Sir Shehreyar/ruen-lid-pipeline/data/demo/seed_labels.csv` (51 tokens, demo slice)

## Overall

| Metric | Value |
|---|---|
| Token accuracy | 0.8235 |
| Macro F1 | 0.7984 |
| CMI (gold labels) | 38.75 |
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
