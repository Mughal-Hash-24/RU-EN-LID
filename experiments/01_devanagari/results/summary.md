# Experiment 01: keep the Devanagari sentences

Test = the Roman-only COMI-LINGUA test split (750 sentences, 15,207 scored words) for every variant.
Mean ± standard deviation over seeds. Stage 2 alone = tau 0; + Stage 3 = default tau.

Settings: `{"model": "xlm-roberta-base", "epochs": 5.0, "max_steps": -1, "batch_size": 16, "learning_rate": 3e-05, "warmup_ratio": 0.1, "weight_decay": 0.01, "max_length": 256, "bf16": true}`

## Overall

| Variant | Runs | Train sentences | Valid macro-F1 | Test acc. | Test macro-F1 | UR recall | EN recall | OTH F1 | Test acc. + Stage 3 | Running ex. /25 | Demo /51 | Minutes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| roman | 3 | 6540 | 0.9445 ± 0.0018 | 0.9626 ± 0.0009 | 0.9368 ± 0.0019 | 0.9844 ± 0.0009 | 0.9478 ± 0.0049 | 0.8793 ± 0.0042 | 0.9627 ± 0.0009 | 25, 25, 25 | 51, 51, 51 | 4.3 |
| deva_oth | 3 | 25078 | 0.9438 ± 0.0006 | 0.9618 ± 0.0026 | 0.9358 ± 0.0046 | 0.9846 ± 0.0011 | 0.9381 ± 0.0096 | 0.8779 ± 0.0088 | 0.9618 ± 0.0027 | 25, 25, 25 | 50, 50, 51 | 12.8 |
| deva_ign | 3 | 25078 | 0.9447 ± 0.0002 | 0.9624 ± 0.0005 | 0.9365 ± 0.0010 | 0.9839 ± 0.0004 | 0.9454 ± 0.0037 | 0.8786 ± 0.0023 | 0.9624 ± 0.0005 | 24, 25, 25 | 50, 50, 51 | 12.8 |

## Two-language words on the test split (Stage 2, correct / total, summed over runs)

| Word | Gold | roman | deva_oth | deva_ign |
|---|---|---|---|---|
| *to* | UR | 243/249 | 247/249 | 247/249 |
| *to* | EN | 115/123 | 116/123 | 111/123 |
| *he* | UR | 30/33 | 30/33 | 30/33 |
| *he* | EN | 30/30 | 30/30 | 30/30 |
| *is* | UR | 33/33 | 33/33 | 30/33 |
| *is* | EN | 107/108 | 107/108 | 108/108 |
| *do* | UR | 39/39 | 39/39 | 39/39 |
| *do* | EN | 15/15 | 15/15 | 15/15 |
| *main* | UR | 96/96 | 96/96 | 96/96 |
| *band* | UR | 6/6 | 5/6 | 6/6 |
| *bar* | UR | 18/18 | 18/18 | 18/18 |
| *me* | UR | 285/294 | 287/294 | 288/294 |
| *me* | EN | 19/24 | 18/24 | 19/24 |
| *or* | UR | 25/27 | 23/27 | 24/27 |
| *or* | EN | 3/3 | 2/3 | 3/3 |
| *hi* | UR | 258/258 | 258/258 | 258/258 |
| *hi* | EN | 3/6 | 3/6 | 3/6 |
| *hi* | OTH | 0/3 | 0/3 | 0/3 |
| *use* | UR | 8/27 | 10/27 | 12/27 |
| *use* | EN | 37/39 | 36/39 | 37/39 |
| *bat* | UR | 5/6 | 6/6 | 6/6 |
| *bat* | EN | 0/3 | 0/3 | 0/3 |

## Runs

| Variant | Seed | Best epoch | Valid macro-F1 | Test acc. / macro-F1 (Stage 2) | Stage 3 changes (fixed / broke) | Minutes |
|---|---|---|---|---|---|---|
| roman | 1 | 5.0 | 0.9424 | 0.9616 / 0.9348 | 2 (2 / 0) | 5.0 |
| deva_oth | 1 | 3.0 | 0.9439 | 0.9590 / 0.9307 | 1 (0 / 1) | 12.9 |
| deva_ign | 1 | 5.0 | 0.9445 | 0.9628 / 0.9371 | 3 (1 / 1) | 12.8 |
| roman | 2 | 4.0 | 0.9452 | 0.9634 / 0.9386 | 1 (1 / 0) | 4.0 |
| deva_oth | 2 | 4.0 | 0.9432 | 0.9624 / 0.9369 | 2 (0 / 2) | 12.8 |
| deva_ign | 2 | 5.0 | 0.9448 | 0.9619 / 0.9353 | 0 (0 / 0) | 12.8 |
| roman | 3 | 5.0 | 0.9458 | 0.9628 / 0.9371 | 3 (2 / 1) | 4.0 |
| deva_oth | 3 | 5.0 | 0.9443 | 0.9641 / 0.9398 | 1 (1 / 0) | 12.8 |
| deva_ign | 3 | 4.0 | 0.9449 | 0.9625 / 0.9370 | 1 (0 / 1) | 12.8 |
