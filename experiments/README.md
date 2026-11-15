# Experiments

Side experiments on top of the iteration-1 pipeline, mostly from supervisor suggestions. Each one lives in its
own numbered folder with its code, a `README.md` (question, design, how to run, results) and a `results/`
folder. Nothing here changes `src/` or `scripts/`; the experiments import them.

Derived data and per-word predictions contain corpus text, so they go to the git-ignored `data/cache/experiments/`.
Models go to the git-ignored `checkpoints/experiments/` and are deleted after scoring.

| # | Question | Status |
|---|---|---|
| [01](01_devanagari/) | Keep COMI-LINGUA's Devanagari sentences and label the Hindi-script words OTH, instead of dropping them? | Done: no gain; OTH labels lower EN recall by 1 point. Keep dropping them |
| [02](02_longer_training/) | Train for up to 15 epochs instead of 5: where do the curves level off or drop? | Done: flat from epoch 3; 5 epochs were enough |

`score.py` (shared) scores one checkpoint on the Roman-only test split, the running examples and the demo slice,
with Stage 2 alone and with Stage 3.
