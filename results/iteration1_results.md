# Iteration 1 results: Stage 2 trained

Roman Urdu–English token-level LID pipeline · 2026-10-25

The previous iteration-1 results ([`seed_results.md`](seed_results.md), README §9) were measured with a
**fixed-guess fallback** standing in for Stage 2, because no trained model existed. This document reports
the same pipeline with **Stage 2 replaced by a fine-tuned XLM-R model**, and compares the two.

## Summary

| Test set | Before: fallback in place of Stage 2 | After: trained Stage 2 + Stage 3 |
|---|---|---|
| Demo slice, 51 words, gold categories (the previous results' data) | 0.8235 accuracy · 0.7984 macro-F1 · 9 errors | **1.0000 · 1.0000 · 0 errors** |
| The proposal's two running examples, 25 words | 19/25 (0.76) | **25/25** |
| COMI-LINGUA Roman-script test set, 750 sentences, 15,207 words | 0.4191 · 0.4027 | **0.9617 · 0.9352** |

1. **The ambiguous words are now handled.** They were the whole of the previous error (8 of 9 errors on the
   demo slice). The fallback labelled every ambiguous word English; the trained model gets all 8 right on
   the demo slice and 216 of 223 on the COMI test set.
2. **Learning curve:** test accuracy rises from 0.950 with 1,000 training sentences to 0.963 with 5,000,
   then **levels off** (0.963 with all 6,540). On this data, about 5,000 sentences is enough.
3. **Stage 3 currently adds nothing.** The trained model is so confident that the rules almost never apply:
   on the COMI test set they changed 1 word out of 15,207, and that change was wrong.
4. **Caveat:** the training and main test data are romanized *Hindi*–English, used as a stand-in for Roman
   Urdu–English. These numbers show how well Stage 2 learns the task on the stand-in. They are not yet a
   measurement on Roman Urdu.

---

## 1. What changed since the previous results

```
 previous:  sentence ─▶ Stage 1 ─▶ [ fixed guesses: variant/both → UR 0.40, ambiguous → EN 0.30, … ] ─▶ Stage 3 ─▶ labels
 now:       sentence ─▶ Stage 1 ─▶ [ XLM-R base, fine-tuned: a label + confidence per word        ] ─▶ Stage 3 ─▶ labels
```

| Part | Previous results | This iteration |
|---|---|---|
| Stage 1 (candidate expansion) | Seed word lists, 8 variant pairs | Unchanged |
| Stage 2 (classifier) | Heuristic fallback (`pipeline.py`) | `xlm-roberta-base`, fine-tuned (`checkpoints/ruen-lid/`) |
| Stage 3 (rules) | 3 rules, τ = 0.55 | Unchanged. It never fired in the previous results either, so those were the fallback alone |
| Training data | None | 6,540 Roman-script sentences from COMI-LINGUA (Section 2) |
| Test sets | Demo slice | Demo slice + the proposal's running examples + COMI-LINGUA Roman-script test set |

Code added or changed in this iteration:
- New `scripts/prepare_comi_lingua.py` builds the data.
- New `scripts/learning_curve.py` runs and scores the learning curve.
- An `IGN` label for words the annotators couldn't agree on, handled in `train.py`, `metrics.py` and
  `evaluate.py`.
- `train.py`: scoring on validation after every epoch, keeping the best epoch, and new options (seed,
  warm-up, weight decay, bf16, max length).
- `predict.py`: max length 512, and inputs are moved to the model's device (GPU prediction works).
- `fetch_data.py`: the test-file URL is fixed (`LID_test.csv` isn't in Git LFS).
- `evaluate.py`: the `--tau` default comes from `DEFAULT_TAU`.
- `accelerate` added to `requirements.txt`.
- 4 new unit tests (15 in total, all passing).

---

## 2. Setup

### Data

No Roman Urdu corpus has word-level language labels, and human annotation is planned for a later phase. So
Stage 2 is trained on a stand-in: the **Latin-script part of COMI-LINGUA** (Sheth et al., 2025), a
Hindi–English corpus with expert word-level labels.

- **Why it's a fair stand-in:** romanized Hindi is labelled Hindi, not English (a Latin-script *hai* is tagged
  Hindi 3,319 times and English twice). It also shares the everyday words of Roman Urdu (*hai, nahi, ko, ki,
  se, kya, bhi*).
- **Labels:** Hindi → UR, English → EN, other → OTH (names, @mentions, links and similar).

Preparation (`scripts/prepare_comi_lingua.py`):
1. Gold label = the 2-of-3 annotator majority. Where all three disagree, the word gets **`IGN`**: it stays in
   the sentence as context but is never trained on or scored. These are mostly ambiguous words (*to, me, he,
   is*); dropping their sentences would have removed 15% of the training and 23% of the test sentences.
2. Keep Latin-script sentences only.
3. Remove from training every sentence that also appears in the test file: 4,107 of the 4,944 test
   sentences were duplicated in COMI-LINGUA's training file.
4. Split words exactly the way the pipeline does, so training and testing see the same words.
5. Split 90/10 into train and validation (fixed seed). The learning-curve subsets are nested: each smaller
   set is contained in the larger ones.

| File | Sentences | Scored words | UR | EN | OTH | IGN (not scored) |
|---|---|---|---|---|---|---|
| train | 6,540 | 88,075 | 57,363 | 19,527 | 11,185 | 1,289 |
| validation | 727 | 9,848 | 6,419 | 2,161 | 1,268 | 174 |
| test | 750 | 15,207 | 9,921 | 3,541 | 1,745 | 252 |

All 750 test sentences mix both languages, and they are longer than the training sentences (median 20
words, against 12).

### Model and training

| Setting | Value |
|---|---|
| Model | `xlm-roberta-base`, token classification (UR / EN / OTH), first word piece of each word labelled |
| Passes over the data | up to 5; validation scored after each; the best epoch by validation macro-F1 is kept |
| Optimiser | learning rate 3e-5, 10% warm-up then linear decay, weight decay 0.01, batch 16 |
| Other | bf16 mixed precision, max 256 word pieces, seeds 1/2/3 |
| Hardware and time | RTX 4070 Laptop GPU (8 GB); 12 runs in 41 minutes including scoring |

The model kept for the pipeline is the full-data run with the best validation score (seed 2, validation
macro-F1 0.9447).

### Evaluation

Three systems are compared, all scored with the repo's own `src/eval/metrics.py`:
- **Fallback**: the previous setup (no model, fixed guesses), with Stage 3 at τ = 0.55.
- **Stage 2 alone**: the trained model with τ = 0, so no Stage 3 rule can fire.
- **Stage 2 + 3**: the trained model with the default τ = 0.55. This is the full pipeline.

Metrics are word accuracy and macro-F1 (F1 averaged over the labels, so the smaller labels count as much as
UR). They are reported overall and per token category. On the demo slice and the running examples the
categories are **gold** (assigned by hand). COMI-LINGUA has no gold categories, so there they are Stage 1's
**spelling-based** categories, and they include an extra "unknown" for words in neither seed word list.

---

## 3. Results

### 3.1 Demo slice: direct comparison with the previous results

The 12 hand-labelled sentences of `data/demo/seed_labels.csv`. The "before" numbers reproduce
`seed_results.md` exactly.

| | Before (fallback) | After (Stage 2 + 3) |
|---|---|---|
| Accuracy | 0.8235 | **1.0000** |
| Macro-F1 | 0.7984 | **1.0000** |
| Errors | 9 / 51 | **0 / 51** |
| Ambiguous words | 0 / 8 | **8 / 8** |
| "Both" words | 2 / 3 | **3 / 3** |
| Standard words | 34 / 34 | 34 / 34 |
| Variant words | 6 / 6 | 6 / 6 |
| Error share by cause | ambiguity 88.89%, both 11.11% | no errors left to attribute |

The previous error came entirely from the fallback's fixed guesses: "ambiguous → EN" and "both → UR".
The only "both" error was the English *he* in *usne kaha he will come*. The trained model gets that right
too.

### 3.2 The proposal's running examples

| Sentence | Before | After |
|---|---|---|
| Example 1: *Delivery to time pe hui but box band nhi tha, main bht upset hoon, is bar bhi same issue he.* | 16/20. Wrong: *to, band, main, is* (all labelled EN) | **20/20** |
| Example 2: *ajj main mood main houn* | 3/5. Wrong: both *main* (labelled EN) | **5/5** |

The trained model was 98–100% confident on every one of these words, so Stage 3 had nothing to do.
Figure 2's illustrative story, where *nhi* is a low-confidence slip that Stage 3 repairs, doesn't happen
with this model.

### 3.3 COMI-LINGUA Roman-script test set

750 sentences, 15,207 scored words (252 `IGN` words not scored).

| | Fallback | Stage 2 alone | Stage 2 + 3 |
|---|---|---|---|
| Accuracy | 0.4191 | 0.9618 | 0.9617 |
| Macro-F1 | 0.4027 | 0.9352 | 0.9352 |
| Errors | 8,833 | 581 | 582 |

Per label (Stage 2 + 3; the fallback in brackets):

| Label | Precision | Recall | F1 | Gold words |
|---|---|---|---|---|
| UR | 0.979 (0.995) | 0.983 (0.379) | **0.981** (0.549) | 9,921 |
| EN | 0.947 (0.699) | 0.949 (0.251) | **0.948** (0.369) | 3,541 |
| OTH | 0.889 (0.170) | 0.864 (0.989) | **0.876** (0.290) | 1,745 |

Per spelling-based category (correct / total):

| Category | Fallback | Stage 2 + 3 |
|---|---|---|
| Standard (in the seed word lists) | 4,457 / 4,707 (0.947) | 4,665 / 4,707 (0.991) |
| Ambiguous (*to, is, do, main, band*) | 82 / 223 (0.368) | **216 / 223 (0.969)** |
| Both (*he, bar*) | 17 / 27 (0.630) | 26 / 27 (0.963) |
| Variant | 93 / 94 | 93 / 94 |
| Unknown (in neither word list) | 1,726 / 10,156 (0.170) | **9,625 / 10,156 (0.948)** |

- **Why the fallback scores only 0.42 here:** two thirds of the test words aren't in the small seed word
  lists, and the fallback labels unknown words OTH. The trained model doesn't depend on the word lists.
- **Where the remaining 582 errors are:** **73% involve the OTH label** (OTH↔UR 215, OTH↔EN 212; mostly
  names, hashtags and @mentions, which the annotators also labelled inconsistently). Only **27% are UR↔EN
  confusions** (155).

### 3.4 Learning curve: how much data is enough

Test scores of Stage 2 alone, mean ± standard deviation over 3 seeds. Same test set as above; only the
amount of training data changes.

| Training sentences | Validation macro-F1 | Test accuracy | Test macro-F1 | Ambiguous-word accuracy |
|---|---|---|---|---|
| 1,000 | 0.9202 ± 0.0014 | 0.9501 ± 0.0016 | 0.9157 ± 0.0027 | 0.937 ± 0.009 |
| 2,000 | 0.9334 ± 0.0016 | 0.9547 ± 0.0010 | 0.9238 ± 0.0022 | 0.958 ± 0.014 |
| 5,000 | 0.9441 ± 0.0016 | 0.9631 ± 0.0006 | 0.9378 ± 0.0010 | 0.981 ± 0.007 |
| 6,540 (all) | 0.9436 ± 0.0010 | 0.9631 ± 0.0015 | 0.9374 ± 0.0027 | 0.976 ± 0.007 |

```
 test accuracy
 0.965 ┤                              ●───────●   5,000 and 6,540: no further gain
 0.960 ┤
 0.955 ┤          ●
 0.950 ┤  ●
       └──┬───────┬───────────────────┬───────┬──
        1,000   2,000               5,000   6,540   training sentences
```

- Already 1,000 sentences reach 0.950 (the fallback: 0.419).
- Going from 1,000 to 5,000 sentences adds about 1.3 points of accuracy and 2.2 points of macro-F1.
- Beyond 5,000, nothing changes. The differences are smaller than the spread between seeds.
- The spread between seeds is small (≤ 0.002 in accuracy), so the curve is not noise.
- In 10 of the 12 runs the best epoch was the last one (epoch 5). More epochs might add a little, but the
  learning curve already shows the effect of more *data* levelling off.

### 3.5 Stage 3

| Threshold τ | Words re-checked (confidence < τ) | Labels changed | Fixed | Broke | Test accuracy |
|---|---|---|---|---|---|
| 0 (off) | 0 | 0 | 0 | 0 | 0.9618 |
| 0.55 (default) | 73 (0.5%) | 1 | 0 | 1 | 0.9617 |
| 0.75 (the proposal's figures) | 413 (2.7%) | 3 | 1 | 2 | 0.9617 |

Why so little happens:
1. The model is rarely unsure: only 73 of the 15,207 words fall below 0.55.
2. The rules only cover narrow cases, and only one word at a time. One example: "a standard word with only an
   English reading, labelled UR".

The previous results used Stage 3 too, but it changed nothing there either: the fallback's guesses never
triggered a rule. So with neither the fallback nor the trained model has Stage 3 contributed yet.

### 3.6 Ambiguous and "both" words, one by one (COMI test, Stage 2 + 3)

| Word | Correct | Gold UR / EN |
|---|---|---|
| *is* | 47 / 47 | 11 / 36 |
| *main* | 32 / 32 | 32 / 0 |
| *do* | 18 / 18 | 13 / 5 |
| *bar* | 6 / 6 | 6 / 0 |
| *to* | 118 / 124 | 83 / 41 |
| *he* | 20 / 21 | 11 / 10 |
| *me* | 102 / 106 | 98 / 8 |
| *hi* | 87 / 89 | 86 / 2 (+1 OTH) |
| *or* | 8 / 10 | 9 / 1 |
| *bat* | 2 / 3 | 2 / 1 |
| *band* | 1 / 2 | 2 / 0 |
| ***use*** | **14 / 22** | 9 / 13 |

- The model separates the two readings of *to, he, is* and *do* well, and those words occur in both
  languages in the test set.
- *use* is the weak spot: Hindi/Urdu *use* (= *usay*, 'to him/her') vs English "use".
- *me*, *or*, *bat*, *use* and *hi* are "both"-type or ambiguous forms that the proposal's lists don't
  include yet.

---

## 4. What this means for the research questions

| RQ | What iteration 1 can say |
|---|---|
| RQ1: improvement over tools / a plain encoder | So far only against the fallback: 0.42 → 0.96 on the stand-in test set. The off-the-shelf baselines (Microsoft LID, GlotLID/OpenLID, MaskLID, LLMs) haven't been run yet |
| RQ2: share of error by cause | Not measurable yet. The demo slice has no errors left, and on COMI-LINGUA the categories are spelling-based (91% of the remaining errors fall on words "unknown" to the seed lists). Gold categories need the planned annotation |
| RQ3: contribution of each stage | Stage 3: ≈ 0 (one word worse at τ 0.55). Stage 1: can't be measured yet, because the classifier doesn't receive Stage 1's readings, so "plain encoder" and "+ candidate expansion" are currently the same system |
| RQ4: behaviour as mixing increases | Not measured. It needs a per-sentence code-mixing index. The current CMI function returns one corpus-level number, and its formula should be checked against Gambäck & Das |

## 5. Limitations

- **Stand-in data:** romanized Hindi–English, not Roman Urdu–English. Training and test also come from the
  same corpus, so real Roman Urdu text (other sources, other spellings) will probably score lower.
- **Small hand-labelled sets:** the demo slice and running examples have only 76 words, built from the
  proposal's own sentences. 100% there shows the obvious cases work; it isn't a strong result.
- **Spelling-based categories** on COMI-LINGUA, so the per-category numbers there are approximate.
- **The OTH label** (names, mentions, links) isn't part of the proposal's UR/EN task, but most of the
  remaining errors involve it.
- **No baselines yet**, and only one model type (XLM-R base).

## 6. Next steps (iteration 2)

1. Feed Stage 1's readings into the classifier, so that candidate expansion can be measured (RQ3, ablation
   rung 2).
2. Use the re-spelling augmentation and ambiguous-sentence oversampling in training, with realistic
   re-spellings.
3. Re-think Stage 3 for a confident model: either fewer, sharper rules or a role beyond low-confidence words.
   Or report it as not needed.
4. Annotate a Roman Urdu–English test set with gold categories (P1–P2). This is the test that matters for
   the thesis.
5. Run the off-the-shelf baselines.
6. Add *me, or, bat, use, hi* to the ambiguous/variant lists and the annotation guidelines.

## 7. Reproducing these numbers

```bash
python3 data/fetch_data.py --dataset comi-lingua-lid --yes-large
```

```bash
python3 scripts/prepare_comi_lingua.py --subsets 1000,2000,5000
```

```bash
python3 scripts/learning_curve.py
```

```bash
python3 scripts/evaluate.py --checkpoint checkpoints/ruen-lid
```

```bash
python3 scripts/evaluate.py --data data/cache/comi-lingua-lid/roman/test.csv --checkpoint checkpoints/ruen-lid
```

The learning curve writes `results/learning_curve/summary.md` and `runs.json` (all 12 runs), and keeps the best
full-data model in `checkpoints/ruen-lid/` (1.1 GB). `checkpoints/` is git-ignored, so the model is shared
separately.

## Appendix: all 12 runs

| Training sentences | Seed | Best epoch | Validation macro-F1 | Test accuracy / macro-F1 (Stage 2) | (Stage 2 + 3) |
|---|---|---|---|---|---|
| 1,000 | 1 | 4 | 0.9187 | 0.9486 / 0.9131 | 0.9486 / 0.9131 |
| 1,000 | 2 | 5 | 0.9202 | 0.9499 / 0.9154 | 0.9500 / 0.9154 |
| 1,000 | 3 | 5 | 0.9216 | 0.9518 / 0.9185 | 0.9518 / 0.9185 |
| 2,000 | 1 | 5 | 0.9324 | 0.9553 / 0.9252 | 0.9553 / 0.9252 |
| 2,000 | 2 | 5 | 0.9351 | 0.9552 / 0.9249 | 0.9552 / 0.9249 |
| 2,000 | 3 | 5 | 0.9325 | 0.9536 / 0.9213 | 0.9536 / 0.9213 |
| 5,000 | 1 | 5 | 0.9453 | 0.9636 / 0.9383 | 0.9636 / 0.9383 |
| 5,000 | 2 | 5 | 0.9423 | 0.9631 / 0.9384 | 0.9631 / 0.9384 |
| 5,000 | 3 | 5 | 0.9448 | 0.9625 / 0.9366 | 0.9625 / 0.9366 |
| 6,540 | 1 | 4 | 0.9436 | 0.9648 / 0.9405 | 0.9648 / 0.9405 |
| 6,540 | 2 | 5 | 0.9447 | 0.9618 / 0.9352 | 0.9617 / 0.9352 |
| 6,540 | 3 | 5 | 0.9426 | 0.9628 / 0.9366 | 0.9627 / 0.9365 |
