# Experiment 03: Stage 1 feeds Stage 2

**Question.** The proposal's Stage 2 reads each word *with* Stage 1's candidate readings (§7.2), and the
evaluation plan measures that as rung 2 of the ablation ladder: plain encoder → + candidate expansion →
+ verification (§8). In iteration 1 the encoder saw the bare words only, so rungs 1 and 2 were the same system.
Shehreyar Rashid's candidate-conditioned XLM-R (`src/classifier/candidate_model.py`, merged in `cd8fd91`) makes
rung 2 trainable. Does it help, and where?

## What the model sees

Stage 1 gives every word one evidence tag (`src/classifier/features.py`) and, for a known spelling variant,
its Urdu canonical reading:

| Word (running example) | Tag | Reading |
|---|---|---|
| *delivery* | EN (only in the English list) | – |
| *hui* | UR (only in the Urdu list) | – |
| *to, band, main, is* | AMB (in both lists) | – |
| *nhi, bht* | VAR (attested variant) | *nahi, bohat* |
| *bar, he* | BOTH (variant that is also English) | *baar, hai* |
| a word in no list | UNK | – |

Each word piece's input vector becomes `word vector + tag vector + gate × mean(reading's piece vectors)`.
The tag vectors and the gate start at zero, so before training the model is exactly the plain one; only what
the encoder sees differs between versions.

## Design

- **Three versions**, `--candidates none | tag | tag+reading`, × seeds 1, 2, 3 (9 runs).
- **The iteration-1 settings** for all of them: all 6,540 Roman-script training sentences, XLM-R base,
  up to 5 epochs (experiment 02 showed that is enough), best epoch by validation macro-F1, lr 3e-5, warm-up
  0.1, weight decay 0.01, batch 16, bf16, max 256 pieces. The new parameters get their own learning rate
  (1e-3, no weight decay), and while training 15% of the words have their tag hidden (shown as UNK), so the
  model can't rely on the lexicon alone.
- **One Stage 1 for every version**: the seed lexicons plus the words induced from the training split's labels
  (`scripts/build_lexicon.py`, defaults: a word seen at least 3 times, at least 90% one language, or at least
  20% each for "both lists"). Validation and test labels are never used. The same lexicon is used to score
  `none`, so the per-category tables and Stage 3 see the same Stage 1 for all three.
- **Scored** with `../score.py` on the Roman-only test split (750 sentences), the 25 running-example words and
  the 51-word demo slice, Stage 2 alone (τ = 0) and with Stage 3 (τ = 0.55). The best model of each version (by
  validation macro-F1) is kept in the git-ignored `checkpoints/experiments/03_stage1_to_stage2/`; the others
  are deleted after scoring.

## How to run

```bash
python -X utf8 scripts/build_lexicon.py --train data/cache/comi-lingua-lid/roman/train.csv \
  --out data/resources_induced --check data/cache/comi-lingua-lid/roman/valid.csv
python -X utf8 experiments/03_stage1_to_stage2/run.py
```

`results/summary.md` and `results/runs.json` are rewritten after every run; per-run reports are in
`results/scores/`. Training logs and per-word predictions stay local (git-ignored).

## Results (2026-11-16, 9 runs, 95 minutes)

Full tables: `results/summary.md`; everything per run (validation history, test reports, the learned tag and
gate weights): `results/runs.json`.

**The lexicon.** Built from the training split only: 1,399 English and 2,102 Roman Urdu words, 170 of them in
both lists. It covers 75.6% of the validation words; the seed lists alone cover 34.1%. Many of the 166 words it
puts in both lists are names (*abhishek, amitabh, delhi*), because COMI-LINGUA labels names inconsistently.

**Overall** (test, Stage 2 alone, mean ± s.d. of 3 seeds):

| Version | Valid macro-F1 | Test accuracy | Test macro-F1 | EN recall | OTH F1 | Running ex. /25 | Demo /51 |
|---|---|---|---|---|---|---|---|
| `none` | 0.9447 ± 0.0012 | 0.9633 ± 0.0004 | 0.9376 ± 0.0008 | 0.9485 | 0.8797 | 25, 25, 25 | 51, 51, 51 |
| `tag` | 0.9455 ± 0.0007 | **0.9643 ± 0.0010** | **0.9394 ± 0.0013** | 0.9487 | 0.8836 | 25, 25, 25 | 51, 50, 50 |
| `tag+reading` | 0.9445 ± 0.0023 | 0.9642 ± 0.0006 | 0.9392 ± 0.0007 | 0.9504 | 0.8834 | 25, 25, 25 | 51, 51, 50 |

**Paired by seed** (test accuracy minus `none` with the same seed): `tag` +0.0014, +0.0002, +0.0014 (mean
+0.0010); `tag+reading` +0.0006, +0.0017, +0.0004 (mean +0.0009). Better in all 3 seeds, but small: about
15 of 15,207 words, and with 3 seeds not statistically clear (paired t ≈ 2.5, p ≈ 0.13).

**Where the gain comes from** (test accuracy by Stage-1 category, Stage 2 alone):

| Version | standard (10,670 words) | ambiguous (687) | variant (152) | both (27) | unknown (3,671) |
|---|---|---|---|---|---|
| `none` | 0.9805 | 0.8894 | 0.9934 | 0.9630 | 0.9258 |
| `tag` | 0.9820 | 0.8860 | 0.9934 | 0.9383 | 0.9266 |
| `tag+reading` | 0.9820 | 0.8884 | 0.9934 | 0.9506 | 0.9256 |

**What it means.**
- **The gain is a lexicon lookup, not ambiguity resolution.** Nearly all of it is on *standard* words, the
  ones the lexicon puts in exactly one language (+0.0015 on 10,670 words). On the *ambiguous* words, the
  words the method is about, the tag doesn't help (0.8894 → 0.8860, within the seed spread), and English
  *he* is read correctly 27 times instead of 29 (of 30, summed over the 3 seeds). This is what README §10.1 predicted for proxy data with
  lexicons this size: a candidate-conditioned model mostly measures lexicon coverage.
- **The reading adds nothing yet.** The learned gate stays at about 0 (−0.008, 0.030, 0.014), and
  `tag+reading` ≈ `tag`. Readings come only from the 9 attested pairs in `data/resources/variant_pairs.tsv`,
  so very few words have one; the channel needs mined variant pairs (Dakshina) before it can be tested.
- **The tags are used**: each tag's learned vector is about 1/6 the length of a word-piece vector (0.6–1.1 vs
  5.9). The UNK vector is the shortest, which fits "no evidence".
- **One new error on real Roman Urdu**: two of the three `tag` models (and one `tag+reading`) read the demo
  slice's *he* (= *hai*, s3) as English, which every `none` model gets right.
- **Caveat**: the induced lexicon comes from the same training labels the model learns from, so on training
  sentences a word's tag is nearly its label; hiding 15% of the tags only partly offsets that. Inducing the
  tags of each training fold from the other folds would give training tags as noisy as the test ones.

## The kept model

`tag`, seed 1: the best version by mean validation macro-F1 and, within it, the best seed (validation macro-F1
0.9463). It is kept locally as `checkpoints/ruen-lid-tag` (git-ignored, not released). Everything below was chosen
on validation; the test split was scored once per setting, for reporting. Files in `results/kept_model/`.

1. **Calibration** (`python -m src.classifier.calibrate`): temperature T = 1.42, fitted on validation. Expected
   calibration error 0.019 → 0.005, validation NLL 0.125 → 0.110. Labels don't change; the confidences that
   Stage 3's gate reads do. (`calibration.json`)
2. **τ** (`scripts/tune_tau.py`, the iteration-1 selection rule): τ = 0.5617, the largest value at which the
   rules change no validation label. No τ beats Stage 2 alone on validation. (`stage3_tau/`, saved as
   `stage3.json` in the checkpoint)
3. **Context penalty** (`scripts/tune_stage3.py`, 7 τ × 6 penalties on validation): the best cell, τ = 0.7 and
   penalty 0.5, re-decodes 32 validation words and nets +7 (validation accuracy 0.9662 → 0.9669). **On test it
   does the opposite**: 44 words changed, 9 fixed, 32 broken (0.9651 → 0.9636). The gain was the best of 42 cells
   on validation and doesn't carry over, so the context decoder stays off (penalty 0, the default).
   (`stage3_grid.json`, `settings_compare.txt`)

| Setting (chosen on validation) | Valid acc. / macro-F1 | Test acc. / macro-F1 | Test Stage-3 changes (fixed / broke) |
|---|---|---|---|
| Stage 2 alone (τ = 0) | 0.9662 / 0.9456 | 0.9651 / 0.9405 | – |
| **τ = 0.5617, context off (kept)** | 0.9661 / 0.9455 | **0.9651 / 0.9405** | 3 (2 / 1) |
| τ = 0.7, context penalty 0.5 | 0.9669 / 0.9461 | 0.9636 / 0.9381 | 52 (13 / 36) |

With the kept setting: running examples 25/25, demo slice 51/51 (`test_report.md`; Stage 2 alone:
`test_report_tau0.md`). The released iteration-1 model scores 0.9618 / 0.9352 on the same test split. Most of
that 0.0033 gap is the luck of this seed: the controlled comparison above (3 seeds each) puts the gain from the
tag at +0.0010.

```bash
python -X utf8 -m src.classifier.calibrate --checkpoint checkpoints/ruen-lid-tag --valid data/cache/comi-lingua-lid/roman/valid.csv
python -X utf8 scripts/tune_tau.py --checkpoint checkpoints/ruen-lid-tag --test data/cache/comi-lingua-lid/roman/test.csv \
  --out-dir experiments/03_stage1_to_stage2/results/kept_model/stage3_tau
python -X utf8 scripts/tune_stage3.py --data data/cache/comi-lingua-lid/roman/valid.csv --checkpoint checkpoints/ruen-lid-tag \
  --taus 0 0.561734676361084 0.6 0.7 0.8 0.9 0.95 --penalties 0 0.25 0.5 1 2 4 --json
python -X utf8 scripts/evaluate.py --data data/cache/comi-lingua-lid/roman/test.csv --checkpoint checkpoints/ruen-lid-tag
```

Each command finds the checkpoint's own lexicon (`stage1_resources/`) and tuned τ (`stage3.json`) by itself.
