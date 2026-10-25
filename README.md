# Ru–En Token-Level LID — Variation-Aware, Ambiguity-Aware Pipeline

**Iteration 1 — implementation of the proposed methodology.**
This repository turns the proposed pipeline (candidate expansion → context-aware classifier → rule-based
verification) into working code and demonstrates it end-to-end on a small hand-labelled slice. Stage 2 is
trained on proxy data (the Latin-script part of COMI-LINGUA) and runs inside the pipeline; results are in
[`results/iteration1_results.md`](results/iteration1_results.md).

---

## 1. Topic and problem statement

Roman Urdu is Urdu written in the Latin script with no fixed spelling standard. In Roman Urdu–English
code-mixed text, token-level language identification (labelling every word as Urdu or English) is hard for
two distinct reasons that prior work never separates:

- **Cross-lingual lexical ambiguity** — one form, two languages: *main* ("I" / English *main*), *band*
  ("closed" / musical band), *is*, *to*.
- **Orthographic variation** — one word, many spellings: *nahi / nai / nhi*, *hai / he / hy*.

Existing studies show LID degrades on code-mixed input but do not isolate either error source, so errors
cannot be attributed to a cause and every downstream code-mixing analysis rests on unmeasured label error.
The proposed pipeline targets both, and reports error shares **per cause** (ambiguity / spelling variation /
their overlap).

## 2. What this iteration implements

| Stage | Proposal step | Status in this repository |
|---|---|---|
| 1 | Candidate expansion (surface form kept, readings attached, never rewritten) | **Working** — lexicons, attested variant pairs, four-category taxonomy, per-token candidate sets |
| 2 | Context-aware classifier (fine-tuned multilingual encoder) | **Trained (proxy data)** — XLM-R base fine-tuned on 6,540 Latin-script COMI-LINGUA sentences (hi → UR), per-epoch validation, inference wrapper; augmentation (synthetic romanizations, ambiguity oversampling) implemented, not yet used in training |
| 3 | Rule-based verification (priors, confidence gating, consistency) | **Working** — conservative rules with per-token action log |
| 4 | Evaluation (per-category metrics, error attribution, CMI) | **Working** — metrics module + report script + learning-curve runner + MixLID baseline adapter |

Stages 1, 3 and the evaluation harness run on the **Python standard library alone** — no GPU, no
downloaded models. Stage 2 requires `torch`/`transformers`/`accelerate` and a trained checkpoint; see §7.

## 3. Architecture

```mermaid
flowchart LR
    A["Input sentence<br/>(Roman Urdu–English)"] --> B["Stage 1 — Candidate expansion<br/>English lexicon ∩ Roman Urdu lexicon<br/>attested variant pairs<br/>surface form kept"]
    B --> C["Stage 2 — Context-aware classifier<br/>XLM-R token classification<br/>fine-tuned on COMI-LINGUA<br/>Latin-script proxy (hi → UR)"]
    C --> D["Stage 3 — Rule-based verification<br/>lexicon priors · confidence gating ·<br/>consistency checks (conservative)"]
    D --> E["Token labels (UR / EN)<br/>+ category + per-rule attribution"]

    R["Resources<br/>seed lexicons · variant_pairs.tsv<br/>Dakshina (planned)"] --> B
    T["Training data (proxy)<br/>COMI-LINGUA Latin-script splits<br/>scripts/prepare_comi_lingua.py"] -.-> C
```

## 4. Repository layout

```
ruen-lid-pipeline/
├── README.md                  # this file
├── METHODOLOGY_TO_CODE.md     # required step → file/function mapping
├── requirements.txt           # Stage-2 training dependencies
├── data/
│   ├── README.md              # dataset links, licences, roles
│   ├── fetch_data.py          # downloads into data/cache (never committed)
│   ├── resources/             # seed English + Roman Urdu lexicons, variant_pairs.tsv
│   └── demo/seed_labels.csv   # hand-labelled demo slice (12 sentences / 51 tokens)
├── src/
│   ├── candidate_expansion/   # Stage 1
│   ├── classifier/            # Stage 2 (dataset, augment, train, predict)
│   ├── verification/          # Stage 3
│   ├── eval/                  # metrics + baseline adapters
│   └── pipeline.py            # end-to-end CLI
├── scripts/
│   ├── evaluate.py            # token-level evaluation + markdown report
│   ├── prepare_comi_lingua.py # Stage-2 proxy data: COMI-LINGUA Latin-script splits
│   ├── learning_curve.py      # Stage-2 learning curve (subset sizes × seeds) + scoring
│   └── train_poc.py           # Stage-2 entry point (dry-run / smoke / real)
├── tests/test_pipeline.py     # 15 unit tests (stdlib unittest)
├── results/
│   ├── seed_results.md        # demo-slice evaluation in fallback mode (before Stage 2 was trained)
│   ├── iteration1_results.md  # trained Stage 2 vs fallback, learning curve, Stage 3 effect
│   └── learning_curve/        # learning-curve summary.md + runs.json
└── checkpoints/               # trained models (git-ignored)
```

## 5. Setup

**Stages 1/3/CLI/evaluation — no setup required** (Python ≥ 3.10, standard library only).

**Stage 2 training** needs a virtualenv on a Python version with PyTorch wheels (none for 3.14 yet; the
iteration-1 model was trained on Python 3.13 with torch 2.11 + CUDA). `requirements.txt` includes
`accelerate`, which the Hugging Face `Trainer` requires:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 6. Quick start

```bash
# End-to-end pipeline on the running example (heuristic fallback, no checkpoint)
python3 -m src.pipeline --text "Delivery to time pe hui but box band nhi tha, main bht upset hoon, is bar bhi same issue he."

# The same with the trained Stage-2 checkpoint (§7)
python3 -m src.pipeline --checkpoint checkpoints/ruen-lid --text "Delivery to time pe hui but box band nhi tha, main bht upset hoon, is bar bhi same issue he."

# Unit tests (15 tests)
python3 -m unittest discover -s tests

# Evaluation on the hand-labelled demo slice -> results/seed_results.md (fallback)
python3 scripts/evaluate.py --out results/seed_results.md

# The same evaluation with the trained checkpoint
python3 scripts/evaluate.py --checkpoint checkpoints/ruen-lid

# Data pipeline + augmentation statistics (no torch required)
python3 scripts/train_poc.py --dry-run

# Known datasets, licences and download status
python3 data/fetch_data.py --list
```

Example pipeline output (fallback mode, abbreviated):

```
token      category    readings   label  conf  rules
Delivery   standard    EN         EN     0.9
band       ambiguous   EN/band    EN     0.3
nhi        variant     nahi       UR     0.4
bar        both        EN/baar    UR     0.4
he         both        EN/hai     UR     0.4
```

With the trained checkpoint (abbreviated; the ambiguous *band* is now resolved from context):

```
token      category    readings   label  conf   rules
Delivery   standard    EN         EN     0.998
band       ambiguous   EN/band    UR     1.0
nhi        variant     nahi       UR     1.0
bar        both        EN/baar    UR     0.999
he         both        EN/hai     UR     0.984
```

## 7. Stage-2 training (proxy data)

The training pipeline is language-agnostic. No Roman Urdu–English corpus carries word-level LID labels (that
is exactly why the proposal schedules a manual annotation phase), so for this iteration Stage 2 is trained on
**proxy, publicly available token-level LID data**: the **Latin-script part of COMI-LINGUA**, a Hindi–English
corpus with expert word-level labels. Romanized Hindi is labelled Hindi there (not English) and shares the
everyday words of Roman Urdu (*hai, nahi, ko, ki, se*), so `hi` maps to UR. An RU–EN checkpoint follows the
annotation phase.

```bash
# 1. Proxy token-level LID data (Hi-En, COMI-LINGUA; ~146 MB)
python3 data/fetch_data.py --dataset comi-lingua-lid --yes-large

# 2. Latin-script train / valid / test splits, plus nested subsets for the learning curve
python3 scripts/prepare_comi_lingua.py --subsets 1000,2000,5000

# 3a. One training run (XLM-R as proposed; MuRIL is the documented alternative)
python3 -m src.classifier.train \
  --train data/cache/comi-lingua-lid/roman/train.csv \
  --valid data/cache/comi-lingua-lid/roman/valid.csv \
  --model xlm-roberta-base --epochs 5 --batch-size 16 --learning-rate 3e-5 \
  --warmup-ratio 0.1 --weight-decay 0.01 --max-length 256 --bf16 --select-best \
  --out checkpoints/run1

# 3b. Or the learning curve: 1k / 2k / 5k / all training sentences x 3 seeds, each scored on the
#     test split; keeps the best full-data model in checkpoints/ruen-lid
python3 scripts/learning_curve.py

# 4. Inference with a trained checkpoint
python3 -m src.pipeline --text "kal office jana hai" --checkpoint checkpoints/ruen-lid
```

**Proxy splits** (`scripts/prepare_comi_lingua.py`):
- the gold label is the 2-of-3 annotator majority; words without one are labelled `IGN` (kept as context,
  never trained on or scored);
- only Latin-script sentences are kept;
- training sentences that also occur in the test file are removed (4,107 of its 4,944 usable sentences are
  duplicated in COMI-LINGUA's train file);
- words are re-split with the pipeline tokenizer, so training and inference see the same tokens;
- the result is 6,540 train / 727 valid / 750 test sentences.

**Augmentation (proposal §7.2):** spelling variation is synthesized rule-by-rule
(`src/classifier/augment.py:synthetic_variants`), while ambiguity cannot be synthesized and is covered by
oversampling real sentences that contain known ambiguous forms (`oversample_ambiguous`). Both are implemented
and the dry-run prints their statistics, but `train.py` does not use them yet (next iteration).

## 8. Evaluation protocol

Implemented in `src/eval/metrics.py`:

- token accuracy and macro-F1, overall and **per category** (standard / ambiguous / variant / both);
- **error share by cause** — what fraction of errors sit on ambiguous tokens, on variant tokens, and on
  their overlap;
- **CMI** (Gambäck & Das, 2014) with the length normalization `100·(1−Σpₗ²)/(1−1/n)`;
- words labelled `IGN` (no annotator majority) are left out of every score and reported as `ignored`;
- **stage-wise comparison**: `scripts/evaluate.py --checkpoint … --tau 0` scores Stage 2 alone (no rule can
  fire); the default τ scores Stage 2 + Stage 3;
- categories: the demo slice carries gold categories; COMI-LINGUA has none, so its per-category report uses
  Stage 1's form-level categories, including `unknown` for words outside the seed lexicons;
- baseline adapters (`src/eval/baselines/`): the model-free fallback, plus an off-the-shelf **MixLID**
  adapter (2026 token-level LID model; its label inventory has `urd_Arab` but **no `urd_Latn`**, so it
  cannot label Latin-script Urdu — the adapter reports this failure mode).

## 9. Results — iteration 1

Full write-up: [`results/iteration1_results.md`](results/iteration1_results.md). These are **proxy-data
results, not yet a Roman Urdu result** (see §10).

**Trained Stage 2 vs the fallback** (accuracy · macro F1):

| Test set | Fallback (no Stage-2 model) | Trained Stage 2 + Stage 3 |
|---|---|---|
| Demo slice (12 sentences / 51 tokens, gold categories) | 0.8235 · 0.7984 (9 errors) | 1.0000 · 1.0000 (0 errors) |
| Proposal's running examples (25 tokens) | 19 / 25 | 25 / 25 |
| COMI-LINGUA Latin-script test (750 sentences / 15,207 tokens) | 0.4191 · 0.4027 | 0.9617 · 0.9352 |

**Learning curve** (COMI-LINGUA test, Stage 2 alone, mean of 3 seeds; accuracy standard deviation ≤ 0.0016):

| Training sentences | 1,000 | 2,000 | 5,000 | 6,540 |
|---|---|---|---|---|
| Token accuracy | 0.9501 | 0.9547 | 0.9631 | 0.9631 |
| Macro F1 | 0.9157 | 0.9238 | 0.9378 | 0.9374 |

The curve levels off between 5,000 and 6,540 training sentences. Stage 3 changes almost nothing with the
trained model: on the COMI-LINGUA test split it changed 1 of 15,207 tokens at τ = 0.55, and that change was
wrong, because only 0.5% of tokens fall below τ.

### Fallback mode (before Stage 2 was trained)

From `results/seed_results.md` (12 sentences, 51 tokens; **demo, not a thesis result**):

| Metric | Value |
|---|---|
| Token accuracy | 0.8235 |
| Macro F1 | 0.7984 |
| CMI (gold labels, corpus-level) | 38.75 |
| Errors | 9 |

| Category | Tokens | Accuracy | Macro F1 |
|---|---|---|---|
| standard | 34 | 1.0000 | 1.0000 |
| variant | 6 | 1.0000 | 1.0000 |
| both | 3 | 0.6667 | 0.4000 |
| ambiguous | 8 | 0.0000 | 0.0000 |

| Cause | Share of all errors |
|---|---|
| ambiguity | 88.89% |
| both (ambiguity × variation overlap) | 11.11% |

**Interpretation.** In fallback mode the entire residual error sits on ambiguous tokens and their overlap,
because the fallback labels every ambiguous form EN and every "both" form UR (Stage 3 changed no label). The
trained context classifier resolves all of these on the demo slice (8/8 ambiguous, 3/3 both), and 216 of 223
ambiguous-category tokens on the COMI-LINGUA test split.

## 10. Proof-of-concept scope, deviations and limits

Documented deviations, each deliberate and justified:

1. **Stage 2 is trained on proxy data**: the Latin-script part of COMI-LINGUA (Hindi–English, `hi` → UR),
   after removing test sentences that are duplicated in its train file. Romanized Hindi shares Roman Urdu's
   everyday words, but the §9 numbers are proxy results; RU–EN training and evaluation require the seed
   annotation from the proposal's P1–P2 phase.
2. **Heuristic fallback** is used by the CLI/evaluation only when no `--checkpoint` is given: variants → UR,
   ambiguous → EN (low confidence), then Stage-3 rules run. The fallback is clearly labelled in output
   and is not presented as the methodology's classifier.
3. **Variant pairs are attested (seeded)** rather than mined at scale; `cluster_variants` (UrduPhone-style
   phonetic clustering) is included as the discovery tool for expanding `variant_pairs.tsv` with Dakshina
   attestations.
4. **Seed lexicons are compact** (hundreds of words). Production runs point the loader at full English and
   Roman Urdu lexicons (`--resources` / custom files).
5. **Stage-2 training was executed** on Python 3.13 / torch 2.11 with an RTX 4070 Laptop GPU (12
   learning-curve runs in 41 minutes). The smoke mode's tiny test model
   (`hf-internal-testing/tiny-random-XLMRobertaForTokenClassification`) is no longer on Hugging Face, so
   `--smoke` currently fails; real runs are unaffected.
6. **MixLID baseline adapter is included but not executed** here (requires `transformers`); it is the
   planned modern baseline, and its missing `urd_Latn` label is itself evidence for the research gap.
7. **Unresolved annotations are labelled `IGN`.** COMI-LINGUA words on which all three annotators disagree
   (mostly ambiguous forms such as *to, me, he, is*) are kept as context but excluded from training and
   scoring; dropping their sentences instead would remove 15% of the training and 23% of the test sentences.
8. **A third label, `OTH`** (names, mentions, URLs), is kept from the proxy data; the proposal's task is UR/EN.
9. **Stage 2 does not yet receive Stage 1's candidate readings** (the proposal's input is surface form +
   readings + context), so the ablation rungs "plain encoder" and "+ candidate expansion" currently coincide.
10. **Augmentation is implemented but not yet used in training** (§7).
11. **Stage 3 has almost no effect with the trained model** (§9): its rules only re-check tokens below τ, and
    the classifier is rarely that unsure.

## 11. Datasets

See [`data/README.md`](data/README.md) for the full table (roles, licences, links). No corpus is
redistributed; `fetch_data.py` downloads into the git-ignored `data/cache/`.

## 12. Methodology → code mapping

The required step-by-step mapping table lives in [`METHODOLOGY_TO_CODE.md`](METHODOLOGY_TO_CODE.md).

## 13. Roadmap (next iterations)

1. Annotation guidelines and pilot annotation with per-category κ (proposal P1).
2. Full annotation of the RU–EN evaluation sample (P2).
3. Replace seed lexicons with full lexicons + Dakshina attestation mining; add the ambiguous/"both" forms
   found in COMI-LINGUA (*me, or, bat, use, hi*).
4. Train the XLM-R classifier on the annotated RU–EN data (P3); proxy-data training is done (§7, §9).
5. Feed Stage 1's candidate readings into the classifier and use the augmentation in training, so the
   ablation ladder can separate the stages; revisit Stage 3 for a confident classifier.
6. Run baselines (off-the-shelf tools, MixLID, LLMs) and the ablation ladder (P5–P6).
7. Error-share-by-cause and CMI-band analysis (per-sentence CMI); report per-category results (P6).

## 14. References

- Rafae et al., *A clustering framework for lexical normalization of Roman Urdu*, Natural Language Engineering, 2021.
- Benton et al., *Improving informally romanized language identification*, EMNLP, 2025.
- Sheth et al., *COMI-LINGUA: Expert annotated large-scale dataset for multitask NLP in Hindi-English code-mixing*, Findings of EMNLP, 2025.
- Gambäck & Das, *On measuring the complexity of code-mixing*, ICON, 2014.
- Hong et al., *MixLID: Token-level language identification in mixed-language text*, 2026 ([model](https://huggingface.co/nlpai-lab/MixLID)).
- Burchell et al., *Code-switched language identification is harder than you think*, EACL, 2024.
- Kargaran et al., *MaskLID: Code-switching language identification through iterative masking*, ACL, 2024.
- Kitamura et al., *Doppelganger-JC: Benchmarking LLMs' understanding of cross-lingual homographs*, IJCNLP-AACL, 2025.
- Jamil & Rafi, *Code-switching as a safety failure mode in large language models*, AbjadNLP, 2026.
