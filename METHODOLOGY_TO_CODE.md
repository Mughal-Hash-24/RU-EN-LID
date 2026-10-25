# Methodology → Code Mapping

Every step of the proposed methodology maps to a specific file or function.

## Stage 1 — Candidate expansion (no destructive rewriting)

| Methodology step | File | Function / artefact |
|---|---|---|
| Tokenization (whitespace/punctuation aware; punctuation unlabelled) | `src/candidate_expansion/expander.py` | `tokenize()` |
| Load English and Roman Urdu lexicons | `src/candidate_expansion/lexicons.py` | `read_wordlist()`, `load_lexicons()` |
| Validate English lexicon against false friends (*hai, ko, ki, se, …*) | `src/candidate_expansion/lexicons.py` | `EXCLUSIVELY_ROMAN_URDU` guard + subtraction in `load_lexicons()` |
| Attested variant pairs (canonical ↔ variant) | `data/resources/variant_pairs.tsv` | resource file; loaded by `variants.py:load_variant_pairs()` |
| Phonetic encoding and variant discovery (UrduPhone-style) | `src/candidate_expansion/variants.py` | `urduphone()`, `cluster_variants()`, `canonical_of()` |
| Dakshina attestation loading (planned expansion of variant pairs) | `src/candidate_expansion/variants.py` | `load_dakshina_lexicon()` |
| Four-category taxonomy (standard / ambiguous / variant / both) | `src/candidate_expansion/ambiguous.py` | `build_taxonomy()` |
| Per-token candidate reading set (surface kept, EN reading, Urdu canonical(s), category) | `src/candidate_expansion/expander.py` | `CandidateExpander.expand_token()`, `CandidateExpander.expand()` |

## Stage 2 — Context-aware classifier

| Methodology step | File | Function / artefact |
|---|---|---|
| Token-level dataset loading (seed format + proxy CSVs with schema detection) | `src/classifier/dataset.py` | `read_any()`, `read_seed_csv()`, `read_proxy_lid_csv()` |
| Proxy training data: COMI-LINGUA Latin-script subset (annotator majority, leakage removal, pipeline tokenization, train/valid/test + nested subsets) | `scripts/prepare_comi_lingua.py` | `majority_record()`, `retokenize()`, `main()` (`--subsets`) |
| Unresolved annotations kept as context but excluded from training (no annotator majority → `IGN`) | `src/classifier/dataset.py`, `src/classifier/train.py` | `IGNORE_LABEL`, `align_labels()` |
| Synthetic romanization augmentation (spelling variation is synthesizable) | `src/classifier/augment.py` | `synthetic_variants()`, `augment_sentence()` (implemented; not yet called by `train.py`) |
| Ambiguity oversampling (ambiguity is lexical; real contexts are duplicated) | `src/classifier/augment.py` | `oversample_ambiguous()` (implemented; not yet called by `train.py`) |
| Fine-tuned multilingual encoder training (XLM-R primary; MuRIL alternative) | `src/classifier/train.py` | `main()` (token classification, subword→word label alignment), `python3 -m src.classifier.train` |
| Per-epoch validation; best epoch selected by validation macro-F1 | `src/classifier/train.py` | `--select-best`, `compute_metrics()`, `training_summary.json` |
| Learning curve (how much training data is sufficient; several seeds per size) | `scripts/learning_curve.py` | `main()` → `results/learning_curve/` |
| Smoke mode for a fast end-to-end training check | `src/classifier/train.py` | `--smoke` (tiny random checkpoint, 10 steps; its test model is no longer on Hugging Face, to be replaced) |
| Word-level inference from a checkpoint | `src/classifier/predict.py` | `TokenLIDPredictor.predict_words()` |
| Proof-of-concept training entry point | `scripts/train_poc.py` | `main()`, `--dry-run` for augmentation statistics |

## Stage 3 — Rule-based verification

| Methodology step | File | Function / artefact |
|---|---|---|
| Conservative post-checks: variant consistency, lexicon priors, confidence gating | `src/verification/rules.py` | `apply_rules()` (`variant-consistency`, `english-lexicon-prior`, `urdu-lexicon-prior`) |
| Per-token decision record with applied-rule attribution | `src/verification/rules.py` | `Decision` dataclass |
| Threshold (τ) control | `src/verification/rules.py`, `src/pipeline.py` | `DEFAULT_TAU`, `--tau` |

## Pipeline orchestration

| Methodology step | File | Function / artefact |
|---|---|---|
| End-to-end Stage 1 → 2 → 3 execution | `src/pipeline.py` | `LIDPipeline.run()`, `LIDPipeline.run_records()` |
| Model-free fallback for demos/tests (clearly labelled) | `src/pipeline.py` | `heuristic_predictions()` |
| Command-line interface | `src/pipeline.py` | `main()` (`python3 -m src.pipeline --text …`) |

## Evaluation (proposal §8)

| Methodology step | File | Function / artefact |
|---|---|---|
| Token accuracy and macro-F1 (overall and per category) | `src/eval/metrics.py` | `accuracy()`, `macro_f1()`, `summarize()` (`IGN` words excluded, counted as `ignored`) |
| Stage-wise comparison: Stage 2 alone vs with Stage 3 | `scripts/evaluate.py` | `--checkpoint …` with `--tau 0` (no rule can fire) vs the default τ |
| Error share by cause (ambiguity / spelling / overlap) | `src/eval/metrics.py` | `summarize()` → `error_share_by_cause` |
| Code-Mixing Index (Gambäck & Das normalization) | `src/eval/metrics.py` | `cmi()` |
| Off-the-shelf baseline adapter (MixLID, 111 labels, no `urd_Latn`) | `src/eval/baselines/mixlid.py` | `MixLIDBaseline.predict_words()` |
| Evaluation report (markdown) | `scripts/evaluate.py` | `main()`, `markdown_report()` |

## Data and artefacts

| Methodology step | File | Function / artefact |
|---|---|---|
| Dataset registry, links, licences, downloads (no redistribution) | `data/README.md`, `data/fetch_data.py` | `DATASETS`, `download()`, `--list` |
| Hand-labelled seed slice (four-category gold labels) | `data/demo/seed_labels.csv` | 12 sentences / 51 tokens |
| Seed resource lists | `data/resources/english_seed.txt`, `data/resources/roman_urdu_seed.txt` | compact PoC lexicons |
| Unit tests (taxonomy, variants, rules, metrics, `IGN` handling, label alignment, proxy-data preparation, seed round-trip) | `tests/test_pipeline.py` | 15 `unittest` cases |
| Iteration-1 evaluation output, fallback (before Stage 2 was trained) | `results/seed_results.md` | generated by `scripts/evaluate.py` |
| Iteration-1 results with the trained Stage 2 (vs fallback, learning curve, Stage 3 effect) | `results/iteration1_results.md`, `results/learning_curve/` | from `scripts/evaluate.py` and `scripts/learning_curve.py` |
