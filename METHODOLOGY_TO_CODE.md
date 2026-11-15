# Methodology → Code Mapping

Every step of the proposed methodology maps to a specific file or function.

## Iteration-1 scope

Iteration 1 implements every methodological stage and the end-to-end path, and reports the **plain-encoder**
rung of the proposal's ablation ladder (§8). The proposal leaves the encoding of candidate readings open
(§7.2); the **+ candidate-expansion** rung that conditions the encoder on those readings is iteration 2.
Stage 3 is implemented and ablated, but kept deliberately conservative (§7.3), so its near-zero effect on
the confident proxy model is reported rather than hidden. README §10.1–§10.3 gives the full justification.

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
| Stage 1 → Stage 2 handoff (iteration 2): each word's evidence tag (EN/UR/AMB/VAR/BOTH/UNK) and Urdu canonical reading | `src/classifier/features.py` | `stage1_tag()`, `reading_of()`, `piece_features()` |
| Candidate-conditioned XLM-R: tag and reading added to the input embedding; zero-initialised so step 0 equals the plain model | `src/classifier/candidate_model.py` | `CandidateXLMRForTokenClassification`, `reset_candidate_parameters()`; `train.py --candidates none\|tag\|tag+reading`, `--cand-dropout` |
| Temperature scaling of Stage-2 confidences (fit on validation) | `src/classifier/calibrate.py` | `fit_temperature()`; stored as `temperature` in the checkpoint config |
| Corpus-induced lexicons: EN / UR / AMB from training-split label counts (seeds never edited) | `src/candidate_expansion/induce.py`, `scripts/build_lexicon.py` | `induce()`; `--min-count`, `--purity`, `--ambiguous-share`; `induction_report.json` |
| A candidate checkpoint keeps the Stage-1 lexicons it was trained with, and every tool reuses them | `src/classifier/train.py`, `src/pipeline.py` | `<checkpoint>/stage1_resources/`; `resolve_resources()`, `hub_resources()` (`--resources` > saved with the checkpoint, a local folder or its Hugging Face repository > `data/resources`) |
| Ablation rung 1 vs rung 2: plain encoder vs + Stage-1 tag vs + tag and reading, same splits and seeds | `experiments/03_stage1_to_stage2/` | `run.py` (3 versions × 3 seeds), `results/summary.md` (paired by seed, by Stage-1 category), `results/kept_model/` (calibration, τ, context penalty, test report of the kept model) |
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
| Threshold (τ) control | `src/verification/rules.py`, `src/verification/tau.py`, `src/pipeline.py` | `DEFAULT_TAU`, `--tau`; `resolve_tau()` (explicit `--tau` > tuned `stage3.json` of the checkpoint, a local folder or its Hugging Face repository > `DEFAULT_TAU`) |
| Threshold (τ) selected on validation data: exhaustive sweep over every τ where a decision can change, best validation macro-F1 (then accuracy), ties to the largest τ before the score first drops, checked on a 0.0001 grid; test reported only | `scripts/tune_tau.py`, `src/verification/tau.py` | `candidate_taus()`, `sweep()`, `select_tau()`, `save_tuned_tau()` → `results/stage3_tau/` |
| Confidence calibration diagnostics (ECE, temperature scaling) and per-rule τ assessment | `src/eval/calibration.py`, `scripts/tune_tau.py` | `expected_calibration_error()`, `fit_temperature()`; `--calibration`; `per_rule()` |
| Deferred words: Stage 2 below τ hands the word to Stage 3 | `src/verification/rules.py` | `Decision.deferred`, `Decision.stage2_label` |
| Context decoder (iteration 2): deferred ambiguous/unknown words re-decoded with a UR↔EN switch penalty; accepted and rule-fixed words pinned | `src/verification/context.py`, `src/verification/rules.py` | `decode()` (two-state Viterbi), `apply_context()`, `CONTEXT_CATEGORIES`; `--context-penalty` |
| Tuning τ and the switch penalty on validation | `scripts/tune_stage3.py` | grid over τ × penalty; Stage 2 runs once per sentence |
| Selective evaluation: coverage, accuracy of accepted vs deferred words, Stage-3 fixes and breaks per rule, per Stage-1 category | `src/eval/selective.py`, `scripts/evaluate.py` | `collect_rows()`, `selective_report()` |
| Stage-3 ablation (τ = 0 vs default vs 0.75) and the near-zero effect of conservative rules | `scripts/evaluate.py`, `results/iteration1_results.md` §3.5, `results/stage3_tau/summary.md` | `--tau`; per-token `Decision.rules`; README §10.2 |

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
| Stage-wise comparison: Stage 2 alone vs with Stage 3 | `scripts/evaluate.py` | `--checkpoint …` with `--tau 0` (no rule can fire) vs the tuned or default τ (no `--tau`) |
| Error share by cause (ambiguity / spelling / overlap) | `src/eval/metrics.py` | `summarize()` → `error_share_by_cause` |
| Code-Mixing Index, Gambäck & Das (2014), per sentence and averaged | `src/eval/metrics.py` | `cmi()`, `mean_sentence_cmi()` |
| Off-the-shelf baseline adapter (MixLID, 111 labels, no `urd_Latn`) | `src/eval/baselines/mixlid.py` | `MixLIDBaseline.predict_words()` |
| Evaluation report (markdown) | `scripts/evaluate.py` | `main()`, `markdown_report()` |

## Data and artefacts

| Methodology step | File | Function / artefact |
|---|---|---|
| Dataset registry, links, licences, downloads (no redistribution) | `data/README.md`, `data/fetch_data.py` | `DATASETS`, `download()`, `--list` |
| Hand-labelled seed slice (four-category gold labels) | `data/demo/seed_labels.csv` | 12 sentences / 51 tokens |
| Proposal's two running examples (four-category gold labels) | `data/demo/running_examples.csv` | 2 sentences / 25 tokens; evaluated with `scripts/evaluate.py` |
| Seed resource lists | `data/resources/english_seed.txt`, `data/resources/roman_urdu_seed.txt` | compact PoC lexicons |
| Unit tests (taxonomy, variants, rules, τ and lexicon precedence, metrics, `IGN` handling, label alignment, proxy-data preparation, seed round-trip) | `tests/test_pipeline.py` | 28 `unittest` cases (65 across `tests/`, with `test_candidate_model.py`, `test_stage3_context.py`, `test_robustness.py`, `test_diagrams.py`) |
| Iteration-1 evaluation output, fallback (before Stage 2 was trained) | `results/seed_results.md` | generated by `scripts/evaluate.py` |
| Iteration-1 results with the trained Stage 2 (vs fallback, learning curve, Stage 3 effect) | `results/iteration1_results.md`, `results/learning_curve/` | from `scripts/evaluate.py` and `scripts/learning_curve.py` |
| Stage-3 threshold selection for the iteration-1 model (sweep, per-rule effect, calibration) | `results/stage3_tau/` | `summary.md`, `sweep.json`, `sweep.csv`, `zoom.csv` from `scripts/tune_tau.py`; the selected τ in `checkpoints/ruen-lid/stage3.json` (git-ignored) and in the Hugging Face model repository |
| No shared sentences between the proxy splits (same text, same word sequence, near copies by word-set Jaccard) | `scripts/check_split_overlap.py` → `results/split_overlap.json` | counts only, no corpus text; `--predictions` adds test accuracy with and without near copies |
| Side experiments: Devanagari sentences as extra training data; up to 15 epochs; Stage 1 → Stage 2 (rung 2) | `experiments/01_devanagari/`, `experiments/02_longer_training/`, `experiments/03_stage1_to_stage2/` | each folder: `README.md` (question, design, results), `run.py`, `results/summary.md` + `runs.json`; shared scorer `experiments/score.py` |
