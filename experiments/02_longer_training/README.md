# Experiment 02: train for longer

**Question (supervisor's suggestion).** Iteration 1 stopped every run at 5 epochs, and the best validation epoch
was the last one in 10 of 12 runs. The curves were cut off at the limit we chose, so they don't show whether
the scores were still rising, levelling off or dropping. What happens with more epochs?

## Design

- The same 12 learning-curve runs as iteration 1 (`scripts/learning_curve.py`): 1,000 / 2,000 / 5,000 / all 6,540
  Roman-script training sentences × seeds 1, 2, 3. The same validation (727) and test (750) files.
- The same settings (XLM-R base, lr 3e-5, warm-up 0.1, weight decay 0.01, batch 16, bf16, max 256 pieces), except
  **up to 15 epochs** instead of 5. Validation after every epoch; the best epoch by validation macro-F1 is kept and
  scored on the test set.
- The learning rate still decays linearly to 0 at the last step, so in a 15-epoch run it falls more slowly. Epoch 5
  of a 15-epoch run is therefore not the same model as the end of a 5-epoch run.
- Recorded per run: validation loss, macro-F1 and accuracy after every epoch; the training loss every 50 steps;
  test scores (`../score.py`). The best full-data model is kept in the git-ignored
  `checkpoints/experiments/02_longer_training/best_full_data/`; the others are deleted.

## How to run

```bash
python -X utf8 experiments/02_longer_training/run.py
```

About 80 minutes on an RTX 4070 laptop GPU (a full-data run takes about 11 minutes). `results/summary.md` and
`results/runs.json` are rewritten after every run.

## Results (2026-11-09, 12 runs, 79 minutes)

Full tables: `results/summary.md`; everything per run (validation history, training loss, test reports):
`results/runs.json`.

**Validation after every epoch** (all 6,540 sentences, mean of 3 seeds):

| Epoch | 1 | 3 | 5 | 7 | 10 | 15 |
|---|---|---|---|---|---|---|
| Validation macro-F1 | 0.920 | 0.941 | 0.942 | **0.943** | 0.941 | 0.942 |
| Validation loss | 0.155 | **0.121** | 0.135 | 0.160 | 0.200 | 0.246 |

- Macro-F1 levels off from epoch 3 and stays within ±0.002 up to epoch 15. The best epoch per seed is 9, 7 and 6.
- Validation loss is lowest at epoch 3 and doubles by epoch 15, while the training loss keeps falling: the model
  grows over-confident on the words it gets wrong, without getting more of them wrong.

**Test accuracy at the best validation epoch** (Stage 2 alone, mean ± s.d. over 3 seeds):

| Train sentences | Up to 5 epochs (iteration 1) | Up to 15 epochs | Best epochs |
|---|---|---|---|
| 1,000 | 0.9501 ± 0.0016 | 0.9527 ± 0.0003 | 8, 13, 15 |
| 2,000 | 0.9547 ± 0.0010 | 0.9550 ± 0.0005 | 5, 4, 9 |
| 5,000 | 0.9631 ± 0.0006 | 0.9618 ± 0.0012 | 7, 7, 14 |
| 6,540 | 0.9631 ± 0.0015 | 0.9623 ± 0.0010 | 9, 7, 6 |

**What it means.**
- More epochs help only the smallest subset (1,000 sentences: +0.003). From 2,000 sentences on the difference is
  within about one standard deviation, and at 5,000 and 6,540 it is slightly lower.
- The learning curve is flat after 5,000 sentences with either limit.
- So 5 epochs did not cut off a real gain. The released 5-epoch model stays.
- The best epoch jumps around (4 to 15) because validation macro-F1 is flat: tiny differences decide it.

The best 15-epoch full-data model (seed 2, epoch 7, validation macro-F1 0.9444, test 0.9635) is kept in
`checkpoints/experiments/02_longer_training/best_full_data/` (1.1 GB). It is not released, and can be deleted.
