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
| COMI-LINGUA (LID subset) | **Proxy** token-level LID for the Stage-2 proof of concept | hi / en / ot | CC BY 4.0 | https://github.com/lingo-iitgn/COMI-LINGUA |
| LinCE | Proxy token-level LID (Hi-En, Es-En, Ne-En) | Word-level LID | Research use (see site) | https://ritual.uh.edu/lince/ |
| MixLID (model) | Off-the-shelf token-level LID **baseline** | 111 language labels | MIT | https://huggingface.co/nlpai-lab/MixLID |

**Important:** none of the Roman Urdu corpora carry word-level language labels, which is why the
proposal includes a manual annotation phase (P1-P2). The seed file in `data/demo/seed_labels.csv`
is a small hand-labelled demonstration slice created for this repository.

COMI-LINGUA stores its CSVs with Git LFS (the `LID_train.csv` object is ~123 MB).
`fetch_data.py` uses the LFS media URL and asks for confirmation before large downloads.
