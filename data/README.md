# Datasets

This project does **not** redistribute corpora. Files are downloaded into `data/cache/`
(ignored by git) by `fetch_data.py`, or obtained from the links below.

| Dataset | Role | Labels | Licence | Link |
|---|---|---|---|---|
| Roman Urdu Data Set (UCI) | Main RU-EN pool | Sentiment (3) - **no LID labels** | CC BY 4.0 | https://archive.ics.uci.edu/dataset/458/roman+urdu+data+set |
| RUHSOLD | Main RU-EN pool (tweets) | Hate / offensive - **no LID labels** | MIT | https://doi.org/10.18653/v1/2020.emnlp-main.197 |
| RUECD (RUSAS release) | Genre breadth | None | **Not stated** - do not redistribute | RUSAS/RUECD release referenced in the proposal |
| Propaganda corpus | Dense-mixing stress set | 20 techniques | MIT | Referenced in the proposal; verify URL before annotation |
| RUSAD | Candidate pool | Sentiment (2) - **no LID labels** | CC BY 4.0 | https://doi.org/10.24432/C59046 |
| Dakshina (Urdu) | Variant / romanization reference | Native-script pairs | CC BY-SA 4.0 | https://github.com/google-research-datasets/dakshina |
| COMI-LINGUA (LID subset) | **Proxy** Stage-2 training and evaluation data (Latin-script sentences only; hi → UR) | hi / en / ot | CC BY-SA 4.0 (repository `LICENSE.txt`; its README says CC BY 4.0) | https://github.com/lingo-iitgn/COMI-LINGUA |
| LinCE | Proxy token-level LID (Hi-En, Es-En, Ne-En) | Word-level LID | Research use (see site) | https://ritual.uh.edu/lince/ |
| MixLID (model) | Off-the-shelf token-level LID **baseline** | 111 language labels | MIT | https://huggingface.co/nlpai-lab/MixLID |

**Important:** none of the Roman Urdu corpora carry word-level language labels, which is why the
proposal includes a manual annotation phase (P1-P2). The seed file in `data/demo/seed_labels.csv`
is a small hand-labelled demonstration slice created for this repository. `data/demo/running_examples.csv`
adds the proposal's two running examples (25 tokens) with gold labels and categories; it is the reproducible
form of `results/iteration1_results.md` §3.2.

The iteration-1 proxy checkpoint (`OmarNiazi/ruen-lid-xlmr-base-iter1`) is distributed under CC BY-SA 4.0,
following the COMI-LINGUA training data.

COMI-LINGUA stores `LID_train.csv` with Git LFS (~123 MB) but `LID_test.csv` as a plain git file (~23 MB);
`fetch_data.py` uses the matching URL for each and asks for confirmation before large downloads.

**Stage-2 proxy splits.** `scripts/prepare_comi_lingua.py` turns the two files into
`data/cache/comi-lingua-lid/roman/{train,valid,test}.csv` (seed CSV format):
- 2-of-3 annotator majority per word; words with no majority are labelled `IGN` (kept as context, never
  trained on or scored);
- Latin-script sentences only;
- training sentences that also occur in `LID_test.csv` are removed (4,107 of its 4,944 usable sentences are
  duplicated in `LID_train.csv`);
- words are re-split with the pipeline tokenizer;
- 90/10 train/validation split.

Result: 6,540 train / 727 validation / 750 test sentences. `--subsets 1000,2000,5000` adds nested random
training subsets for the learning curve.
