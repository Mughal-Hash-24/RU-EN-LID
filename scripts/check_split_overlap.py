"""Check that the prepared train / validation / test splits share no sentences.

Three checks for every pair of splits, from strict to loose:
  same text    the same COMI-LINGUA sentence (whitespace-normalized text)
  same words   the same word sequence after the pipeline tokenizer, ignoring case and punctuation
  near copy    word-set overlap (Jaccard) of at least --near (default 0.9): e.g. a copy with one word changed
For comparison it runs the same checks before cleaning: the Roman-script training sentences as COMI-LINGUA
ships them, against the test split. With --predictions (a test.csv written by experiments/score.py) it also
reports the test accuracy with and without the test sentences that have a near copy in train or validation.
Writes counts only (no corpus text) to --out.

    python -I -X utf8 scripts/check_split_overlap.py [--predictions <score.py predictions>/test.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_comi_lingua as prep  # noqa: E402


def raw_texts(path: Path) -> dict[int, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {index: prep.normalize_text(row[prep.TEXT_COLUMN] or "")
                for index, row in enumerate(csv.DictReader(handle))}


def read_prepared(path: Path, texts: dict[int, str]) -> list[dict]:
    sentences: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            entry = sentences.setdefault(row["sentence_id"], {"row": int(row["source_row"]), "words": []})
            entry["words"].append(row["token"])
    return [{"id": sentence_id, "text": texts[s["row"]], "key": tuple(w.lower() for w in s["words"]),
             "set": frozenset(w.lower() for w in s["words"])} for sentence_id, s in sentences.items()]


def from_records(records: list[prep.Record]) -> list[dict]:
    return [{"text": r.text, "key": tuple(w.lower() for w in r.words), "set": frozenset(w.lower() for w in r.words)}
            for r in records]


def best_jaccard(query: frozenset, index: dict[str, list[int]], sets: list[frozenset]) -> float:
    """Highest word-set Jaccard between query and any indexed set (candidates share at least one word)."""
    shared: dict[int, int] = defaultdict(int)
    for word in query:
        for i in index.get(word, ()):
            shared[i] += 1
    best = 0.0
    for i, inter in shared.items():
        best = max(best, inter / (len(query) + len(sets[i]) - inter))
    return best


def compare(a: list[dict], b: list[dict], near: float, near_ids: set | None = None) -> dict:
    """How many sentences of b have a match in a, per check. Adds the ids of b's near copies to near_ids."""
    texts = {s["text"] for s in a}
    keys = {s["key"] for s in a}
    sets = [s["set"] for s in a]
    index: dict[str, list[int]] = defaultdict(list)
    for i, words in enumerate(sets):
        for word in words:
            index[word].append(i)
    scores = [best_jaccard(s["set"], index, sets) for s in b]
    if near_ids is not None:
        near_ids.update(s["id"] for s, score in zip(b, scores) if score >= near)
    return {
        "sentences": len(b),
        "same_text": sum(1 for s in b if s["text"] in texts),
        "same_words": sum(1 for s in b if s["key"] in keys),
        "near_copy": sum(1 for score in scores if score >= near),
        "max_jaccard": round(max(scores, default=0.0), 4),
        "jaccard_at_least": {f"{t:.1f}": sum(1 for score in scores if score >= t) for t in (0.5, 0.6, 0.7, 0.8, 0.9)},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", default="data/cache/comi-lingua-lid/LID_train.csv")
    parser.add_argument("--test", default="data/cache/comi-lingua-lid/LID_test.csv")
    parser.add_argument("--splits", default="data/cache/comi-lingua-lid/roman")
    parser.add_argument("--near", type=float, default=0.9)
    parser.add_argument("--out", default="results/split_overlap.json")
    parser.add_argument("--predictions", help="per-word test predictions (experiments/score.py --predictions-dir)")
    args = parser.parse_args(argv)
    csv.field_size_limit(10**9)

    train_raw, test_raw = ROOT / args.train, ROOT / args.test
    train_texts, test_texts = raw_texts(train_raw), raw_texts(test_raw)
    splits_dir = ROOT / args.splits
    splits = {
        "train": read_prepared(splits_dir / "train.csv", train_texts),
        "valid": read_prepared(splits_dir / "valid.csv", train_texts),
        "test": read_prepared(splits_dir / "test.csv", test_texts),
    }
    near_test: set[str] = set()
    after = {
        "train_vs_test": compare(splits["train"], splits["test"], args.near, near_test),
        "valid_vs_test": compare(splits["valid"], splits["test"], args.near, near_test),
        "train_vs_valid": compare(splits["train"], splits["valid"], args.near),
    }
    # Before cleaning: every Roman-script training sentence of the original file, against the same test split.
    roman_train, _, _ = prep.load(train_raw)
    roman_train = prep.retokenize(roman_train, {"steps": [], "tokens": {}})
    before = {"train_file_vs_test": compare(from_records(roman_train), splits["test"], args.near)}

    report = {"near_copy_threshold": args.near, "after_cleaning": after, "before_cleaning": before,
              "roman_train_sentences_before_cleaning": len(roman_train),
              "test_sentences_with_near_copy": len(near_test)}
    if args.predictions:
        with Path(args.predictions).open(newline="", encoding="utf-8") as handle:
            words = [row for row in csv.DictReader(handle) if row["gold"] != "IGN"]

        def accuracy(rows: list[dict]) -> dict:
            return {"words": len(rows), "accuracy": round(sum(r["stage2"] == r["gold"] for r in rows) / len(rows), 4)}

        report["test_accuracy_stage2"] = {
            "predictions": (Path(os.path.relpath(args.predictions, ROOT)).as_posix()
                            if Path(args.predictions).resolve().is_relative_to(ROOT) else Path(args.predictions).name),
            "all": accuracy(words),
            "without_near_copies": accuracy([r for r in words if r["sentence_id"] not in near_test]),
            "near_copies_only": accuracy([r for r in words if r["sentence_id"] in near_test]),
        }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    failed = [pair for pair, r in after.items() if r["same_text"] or r["same_words"]]
    print("OK: no shared sentences between splits" if not failed else f"OVERLAP in {failed}")
    print(f"[written] {out.relative_to(ROOT) if out.resolve().is_relative_to(ROOT) else out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
