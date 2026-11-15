# Experiment 03: Stage 1 feeds Stage 2

Test = the Roman-only COMI-LINGUA test split (750 sentences, 15,207 scored words), the same for every version. Stage 2 alone (tau 0) unless marked. Mean ± standard deviation over seeds.

Settings: `{"model": "xlm-roberta-base", "epochs": 5.0, "batch_size": 16, "learning_rate": 3e-05, "warmup_ratio": 0.1, "weight_decay": 0.01, "max_length": 256, "cand_dropout": 0.15, "cand_lr": 0.001, "bf16": true, "resources": "data/resources_induced"}`

Stage 1 lexicon (all versions): `data/resources_induced`, 1399 English / 2102 Roman Urdu words, 170 in both; it covers 75.6% of validation words (seeds alone: 34.1%).

## Overall

| Version | Runs | Valid macro-F1 | Test acc. | Test macro-F1 | UR recall | EN recall | OTH F1 | Test acc. + Stage 3 | Running ex. /25 | Demo /51 |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 3 | 0.9447 ± 0.0012 | 0.9633 ± 0.0004 | 0.9376 ± 0.0008 | 0.9848 ± 0.0006 | 0.9485 ± 0.0030 | 0.8797 ± 0.0025 | 0.9634 ± 0.0004 | 25, 25, 25 | 51, 51, 51 |
| tag | 3 | 0.9455 ± 0.0007 | 0.9643 ± 0.0010 | 0.9394 ± 0.0013 | 0.9858 ± 0.0011 | 0.9487 ± 0.0038 | 0.8836 ± 0.0020 | 0.9643 ± 0.0010 | 25, 25, 25 | 51, 50, 50 |
| tag+reading | 3 | 0.9445 ± 0.0023 | 0.9642 ± 0.0006 | 0.9392 ± 0.0007 | 0.9859 ± 0.0022 | 0.9504 ± 0.0033 | 0.8834 ± 0.0007 | 0.9643 ± 0.0006 | 25, 25, 25 | 51, 51, 50 |

## Paired by seed: version minus `none` (test accuracy, Stage 2 alone)

| Version | Seed 1 | Seed 2 | Seed 3 | Mean ± s.d. | Seeds better |
|---|---|---|---|---|---|
| tag | +0.0014 | +0.0002 | +0.0014 | 0.0010 ± 0.0007 | 3 of 3 |
| tag+reading | +0.0006 | +0.0017 | +0.0004 | 0.0009 ± 0.0007 | 3 of 3 |

## By Stage-1 category (test accuracy, Stage 2 alone; categories from the induced lexicon)

| Version | standard (10670) | ambiguous (687) | variant (152) | both (27) | unknown (3671) |
|---|---|---|---|---|---|
| none | 0.9805 ± 0.0002 | 0.8894 ± 0.0015 | 0.9934 ± 0.0000 | 0.9630 ± 0.0371 | 0.9258 ± 0.0008 |
| tag | 0.9820 ± 0.0007 | 0.8860 ± 0.0111 | 0.9934 ± 0.0000 | 0.9383 ± 0.0214 | 0.9266 ± 0.0024 |
| tag+reading | 0.9820 ± 0.0002 | 0.8884 ± 0.0030 | 0.9934 ± 0.0000 | 0.9506 ± 0.0214 | 0.9256 ± 0.0026 |

## Two-language words on the test split (correct / total, summed over runs)

| Word | Gold | none | tag | tag+reading |
|---|---|---|---|---|
| *band* | UR | 6/6 | 5/6 | 6/6 |
| *bar* | UR | 18/18 | 18/18 | 18/18 |
| *bat* | EN | 0/3 | 0/3 | 0/3 |
| *bat* | UR | 6/6 | 6/6 | 6/6 |
| *do* | EN | 15/15 | 15/15 | 15/15 |
| *do* | UR | 39/39 | 39/39 | 39/39 |
| *he* | EN | 29/30 | 27/30 | 27/30 |
| *he* | UR | 31/33 | 31/33 | 32/33 |
| *hi* | EN | 3/6 | 3/6 | 3/6 |
| *hi* | OTH | 0/3 | 0/3 | 0/3 |
| *hi* | UR | 258/258 | 258/258 | 258/258 |
| *is* | EN | 107/108 | 108/108 | 108/108 |
| *is* | UR | 31/33 | 32/33 | 32/33 |
| *main* | UR | 96/96 | 96/96 | 96/96 |
| *me* | EN | 19/24 | 17/24 | 20/24 |
| *me* | UR | 290/294 | 289/294 | 289/294 |
| *or* | EN | 1/3 | 2/3 | 2/3 |
| *or* | UR | 25/27 | 26/27 | 23/27 |
| *to* | EN | 115/123 | 115/123 | 113/123 |
| *to* | UR | 249/249 | 246/249 | 246/249 |
| *use* | EN | 37/39 | 36/39 | 36/39 |
| *use* | UR | 9/27 | 9/27 | 8/27 |

## Runs

| Version | Seed | Best epoch | Valid macro-F1 | Test acc. / macro-F1 | Reading gate | Stage 3 changes (fixed / broke) | Minutes |
|---|---|---|---|---|---|---|---|
| none | 1 | 5 | 0.9447 | 0.9637 / 0.9385 | – | 2 (1 / 0) | 12.2 |
| tag | 1 | 4 | 0.9463 | 0.9651 / 0.9405 | 0.000 | 2 (2 / 0) | 11.7 |
| tag+reading | 1 | 4 | 0.9419 | 0.9643 / 0.9393 | -0.008 | 3 (3 / 0) | 12.4 |
| none | 2 | 5 | 0.9436 | 0.9630 / 0.9374 | – | 3 (2 / 1) | 11.8 |
| tag | 2 | 5 | 0.9453 | 0.9632 / 0.9380 | 0.000 | 1 (0 / 1) | 8.6 |
| tag+reading | 2 | 4 | 0.9459 | 0.9647 / 0.9398 | 0.030 | 5 (2 / 1) | 9.5 |
| none | 3 | 5 | 0.9459 | 0.9632 / 0.9369 | – | 0 (0 / 0) | 7.7 |
| tag | 3 | 5 | 0.9449 | 0.9646 / 0.9398 | 0.000 | 0 (0 / 0) | 8.3 |
| tag+reading | 3 | 5 | 0.9459 | 0.9636 / 0.9384 | 0.014 | 3 (2 / 1) | 8.9 |
