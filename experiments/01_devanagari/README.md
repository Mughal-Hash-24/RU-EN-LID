# Experiment 01: keep the Devanagari sentences

**Question (supervisor's suggestion).** `scripts/prepare_comi_lingua.py` keeps only Roman-script sentences, so
18,538 leak-free COMI-LINGUA training sentences that contain Devanagari are thrown away. What if we keep them
and label the Hindi-script words `OTH`?

## What those sentences contain

Counts after the same cleaning as the main prep script (2-of-3 vote, sentences that also appear in the test
file removed, duplicates removed). From `results/data_stats.json`.

| | Sentences | Devanagari words | Latin words |
|---|---|---|---|
| Kept today (Roman only) | 6,540 train | 0 | 89,364 |
| Dropped today, added here | 18,538 (15,946 also contain Latin words) | 310k: 281k gold UR, 28k gold EN, 1k OTH | 70k: 61k EN, 8k OTH, 1k UR |

- The Devanagari words already have real gold labels: Hindi (our UR stand-in), and English written in Hindi
  script (ऑफ "of", इंडिया "India", स्टेशन "station").
- The Latin words in these sentences are almost all English. *to* is EN 113 times and UR once here, against
  UR 659 / EN 293 in the Roman-only data. The few Latin "UR" words are mostly names and slogans
  (*mandir, ram, yatra, bharat jodo*).

## Design

Three training sets. Validation (727 sentences) and test (750 sentences, 15,207 scored words) are the **same
Roman-only files** for all three, so every variant is scored on exactly the same words.

| Variant | Training data | Devanagari words | Share of trained words UR / EN / OTH |
|---|---|---|---|
| `roman` | today's 6,540 Roman sentences (baseline, re-run here) | – | 65% / 22% / 13% |
| `deva_oth` | + the 18,538 sentences (25,078 in total) | **OTH** (trained on) | 12% / 17% / 70% |
| `deva_ign` | + the same 18,538 sentences | **IGN** (context only) | 37% / 51% / 12% |

`deva_ign` is the control: the same sentences and the same extra English words, but the Devanagari words are
not a training target. If `deva_oth` beats `roman` and `deva_ign` does too, the gain comes from the extra
English, not from the OTH labels.

- Training settings are the learning curve's: XLM-R base, up to 5 epochs, best epoch by validation macro-F1,
  lr 3e-5, warm-up 0.1, weight decay 0.01, batch 16, bf16, max 256 pieces. No sentence is longer than 106 pieces,
  so nothing is cut off. Seeds 1, 2, 3.
- Splitting words: Latin words are split exactly as the pipeline does (`expander.py` `WORD_RE`, checked word by
  word). A Devanagari word is a run of Devanagari letters, signs and digits; the dandas (।, ॥) are punctuation and
  are dropped. A COMI token such as `IPLमें` becomes `IPL` + `में`, each keeping the token's label (then the
  Devanagari part is relabelled).
- Scoring (`../score.py`, shared by all experiments): Stage 2 alone (τ = 0) and with Stage 3 (τ = 0.55), per-label precision/recall/F1, the
  confusion matrix, and the two-language words (*to, he, is, do, main, band, bar, me, or, hi, use, bat*) split by
  gold label. Also the 25 running-example words and the 51-word demo slice.
- Checked: `../score.py` on `checkpoints/ruen-lid` reproduces the iteration-1 numbers (test 0.9618 / 0.9617 with
  Stage 3, 25/25, 51/51).

## How to run

```bash
python -I -X utf8 scripts/prepare_comi_lingua.py --subsets 1000,2000,5000   # if data/cache/comi-lingua-lid/roman/ is missing
python -X utf8 experiments/01_devanagari/build_data.py
python -X utf8 experiments/01_devanagari/run.py
```

`run.py` rewrites `results/summary.md` and `results/runs.json` after every run. Per-run reports are in
`results/scores/`. Training logs (`results/logs/`, `results/progress.log`) and per-word predictions
(`data/cache/experiments/01_devanagari/predictions/`) stay local and are git-ignored.

## Results (2026-11-09, 9 runs, 90 minutes)

Full tables: `results/summary.md`; per run: `results/runs.json` and `results/scores/`.

**Test split** (Roman only, 15,207 words; Stage 2 alone; mean ± s.d. over 3 seeds):

| Variant | Accuracy | Macro-F1 | UR recall | EN recall | EN precision | OTH F1 | Minutes per run |
|---|---|---|---|---|---|---|---|
| `roman` (today) | 0.9626 ± 0.0009 | 0.9368 ± 0.0019 | 0.984 | **0.948** ± 0.005 | 0.951 | 0.879 | 4 |
| `deva_oth` | 0.9618 ± 0.0026 | 0.9358 ± 0.0046 | 0.985 | **0.938** ± 0.010 | 0.958 | 0.878 | 13 |
| `deva_ign` | 0.9624 ± 0.0005 | 0.9365 ± 0.0010 | 0.984 | 0.945 ± 0.004 | 0.953 | 0.879 | 13 |

Errors by type (mean words per run; gold → predicted):

| Variant | UR→EN | UR→OTH | EN→UR | EN→OTH | OTH→UR | OTH→EN | All |
|---|---|---|---|---|---|---|---|
| `roman` | 78 | 77 | 74 | 111 | 134 | 95 | 569 |
| `deva_oth` | 66 | 87 | 88 | **132** | 130 | 79 | 581 |
| `deva_ign` | 73 | 86 | 77 | 116 | 128 | 92 | 572 |

**Real Roman Urdu** (the 25 running-example words + the 51-word demo slice, per seed): `roman` is 76/76 in all three
seeds. `deva_oth` misses one word in two seeds (*time* → UR; *he* = *hai* → EN); `deva_ign` misses *he* → EN in two
seeds (3 words in total).

**What it means.**
- **No gain in accuracy.** All three are within the seed spread (0.9626 / 0.9618 / 0.9624), at three times the
  training time.
- **Labelling the Hindi-script words OTH makes the model hold back on English.** OTH becomes 70% of what it trains
  on, and 28k of those OTH words are English written in Hindi script. On Roman text it then labels fewer words EN:
  EN recall drops by about one point (more English words go to OTH and UR), and the runs vary three times as much.
- **The extra English alone doesn't help either.** The IGN control has the same sentences and 3× the English
  examples, and it scores like the baseline.
- **The worry from the design holds a little on real Roman Urdu.** The added Roman words are 98% English, and both
  Devanagari versions sometimes read *he* (= *hai*) as English, the very ambiguity the proposal is about. The
  Roman-only model never does. The numbers are small (76 words × 3 seeds), so this is a warning sign, not proof.
- **Recommendation:** keep dropping the Devanagari sentences, as `scripts/prepare_comi_lingua.py` does now.

Not checked: how the models label Devanagari words themselves (the test split has none, and Roman Urdu–English text
never contains them). All 9 models were deleted after scoring; the per-word predictions are kept.
