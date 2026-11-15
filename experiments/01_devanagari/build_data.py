"""Experiment 01: training files that keep COMI-LINGUA's Devanagari sentences.

scripts/prepare_comi_lingua.py keeps only Latin-script sentences. This script adds back every leak-free
COMI-LINGUA train sentence that contains Devanagari, in two variants that differ only in the label of the
Devanagari words (the sentences are identical):

    deva_oth   Devanagari words -> OTH   (trained on, as "other")
    deva_ign   Devanagari words -> IGN   (context only: never trained on, never scored)

Latin words in those sentences keep their 2-of-3 annotator label, exactly as in the main prep script.
Each variant's train.csv = the Roman-only train.csv, copied unchanged, followed by the added sentences.
Validation and test stay the Roman-only files, so all variants are scored on the same words.

Run after scripts/prepare_comi_lingua.py:
    python -X utf8 experiments/01_devanagari/build_data.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_comi_lingua as prep  # noqa: E402
from src.candidate_expansion.expander import WORD_RE, tokenize  # noqa: E402
from src.classifier.dataset import IGNORE_LABEL, read_seed_csv  # noqa: E402

# A Devanagari word: starts with a Devanagari letter, sign or digit; may contain zero-width (non-)joiners.
# The dandas (U+0964, U+0965) are punctuation and are dropped, like Latin punctuation.
DEVA_CHARS = "ऀ-ॣ०-ॿ"
DEVA_WORD = f"[{DEVA_CHARS}][{DEVA_CHARS}‌‍]*"
PIECE_RE = re.compile(f"{WORD_RE.pattern}|{DEVA_WORD}")
DEVA_PIECE_RE = re.compile(f"^{DEVA_WORD}$")
VARIANTS = {"deva_oth": "OTH", "deva_ign": IGNORE_LABEL}
LABELS = ("UR", "EN", "OTH", IGNORE_LABEL)


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def split_token(token: str) -> list[str]:
    """Pipeline-tokenizer pieces plus Devanagari words, in text order."""
    pieces = PIECE_RE.findall(token)
    latin = [piece for piece in pieces if not DEVA_PIECE_RE.match(piece)]
    if latin != tokenize(token):
        raise RuntimeError(f"Latin pieces of {token!r} differ from the pipeline tokenizer: {latin}")
    return pieces


def load_records(path: Path) -> tuple[list[prep.Record], Counter]:
    notes, unknown, failures = Counter(), Counter(), Counter()
    records = []
    with path.open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            result = prep.majority_record(index, row, notes, unknown)
            if isinstance(result, str):
                failures[result] += 1
            else:
                records.append(result)
    return records, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", default="data/cache/comi-lingua-lid/LID_train.csv")
    parser.add_argument("--test", default="data/cache/comi-lingua-lid/LID_test.csv")
    parser.add_argument("--roman-dir", default="data/cache/comi-lingua-lid/roman")
    parser.add_argument("--out-dir", default="data/cache/experiments/01_devanagari")
    parser.add_argument("--stats", default="experiments/01_devanagari/results/data_stats.json")
    parser.add_argument("--tokenizer", default="xlm-roberta-base", help="count subword pieces ('' to skip)")
    args = parser.parse_args(argv)
    csv.field_size_limit(10**9)

    roman_dir, out_dir = resolve(args.roman_dir), resolve(args.out_dir)
    roman_train_path = roman_dir / "train.csv"
    roman_train = read_seed_csv(str(roman_train_path))
    roman_held_out = read_seed_csv(str(roman_dir / "valid.csv")) + read_seed_csv(str(roman_dir / "test.csv"))
    held_out_keys = {tuple(word.lower() for word in sentence.words) for sentence in roman_held_out}

    _, test_texts, _ = prep.load(resolve(args.test))
    records, failures = load_records(resolve(args.train))
    steps = [{"step": "parsed_rows", "remaining": len(records), "parse_failures": dict(failures)}]
    records = prep.keep_where(steps, "has_devanagari", records, lambda r: bool(prep.DEVANAGARI_RE.search(r.text)))
    records = prep.keep_where(steps, "text_in_test_file", records, lambda r: r.text not in test_texts)
    records = prep.keep_where(steps, "duplicate_text", records, prep.first_seen(lambda r: r.text))

    added: list[tuple[prep.Record, list[bool]]] = []  # record with per-word "is Devanagari" flags
    for record in records:
        words, labels, is_deva = [], [], []
        for token, label in zip(record.words, record.labels):
            for piece in split_token(token):
                words.append(piece)
                labels.append(label)
                is_deva.append(bool(DEVA_PIECE_RE.match(piece)))
        added.append((prep.Record(record.source_row, record.text, words, labels), is_deva))
    count = len(added)
    added = [(r, d) for r, d in added if r.words]
    steps.append({"step": "no_words_left", "dropped": count - len(added), "remaining": len(added)})
    count = len(added)
    added = [(r, d) for r, d in added if "UR" in r.labels or "EN" in r.labels]
    steps.append({"step": "no_ur_or_en_word (gold labels)", "dropped": count - len(added), "remaining": len(added)})
    count = len(added)
    added = [(r, d) for r, d in added if prep.token_key(r) not in held_out_keys]
    steps.append({"step": "token_sequence_in_valid_or_test", "dropped": count - len(added), "remaining": len(added)})
    keep = prep.first_seen(prep.token_key)
    count = len(added)
    added = [(r, d) for r, d in added if keep(r)]
    steps.append({"step": "duplicate_token_sequence", "dropped": count - len(added), "remaining": len(added)})

    for record, _ in added:
        if PIECE_RE.findall(" ".join(record.words)) != record.words:
            raise RuntimeError(f"row {record.source_row}: words do not survive re-splitting")

    # Gold labels of the added words, before relabelling, by script.
    gold = {"devanagari": Counter(), "latin_or_digit": Counter()}
    for record, is_deva in added:
        for label, deva in zip(record.labels, is_deva):
            gold["devanagari" if deva else "latin_or_digit"][label] += 1
    def kind(record: prep.Record, is_deva: list[bool]) -> str:
        if any(not deva and prep.ASCII_LETTER_RE.search(word) for word, deva in zip(record.words, is_deva)):
            return "with_latin_words"
        return "devanagari_and_digits_only"

    kinds = Counter(kind(record, is_deva) for record, is_deva in added)

    first_number = len(roman_train) + 1
    variant_stats = {}
    for variant, deva_label in VARIANTS.items():
        variant_dir = out_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        path = variant_dir / "train.csv"
        labels_total = Counter(label for sentence in roman_train for label in sentence.labels)
        with path.open("w", newline="", encoding="utf-8") as handle:
            handle.write(roman_train_path.read_text(encoding="utf-8"))
            writer = csv.writer(handle, lineterminator="\n")
            for number, (record, is_deva) in enumerate(added, start=first_number):
                for index, (word, label, deva) in enumerate(zip(record.words, record.labels, is_deva)):
                    final = deva_label if deva else label
                    labels_total[final] += 1
                    writer.writerow([f"s{number}", word, index, final, record.source_row])
        check = read_seed_csv(str(path))
        if len(check) != len(roman_train) + len(added):
            raise RuntimeError(f"{path}: expected {len(roman_train) + len(added)} sentences, read {len(check)}")
        trained = sum(labels_total[label] for label in ("UR", "EN", "OTH"))
        variant_stats[variant] = {
            "path": os.path.relpath(path, ROOT),
            "sentences": len(check),
            "words": sum(labels_total.values()),
            "labels": {label: labels_total[label] for label in LABELS},
            "trained_label_share": {label: round(labels_total[label] / trained, 4) for label in ("UR", "EN", "OTH")},
        }
    roman_labels = Counter(label for sentence in roman_train for label in sentence.labels)
    roman_trained = sum(roman_labels[label] for label in ("UR", "EN", "OTH"))
    variant_stats["roman"] = {
        "path": os.path.relpath(roman_train_path, ROOT),
        "sentences": len(roman_train),
        "words": sum(roman_labels.values()),
        "labels": {label: roman_labels[label] for label in LABELS},
        "trained_label_share": {label: round(roman_labels[label] / roman_trained, 4) for label in ("UR", "EN", "OTH")},
    }

    pieces = {}
    if args.tokenizer:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        for name, sentences in (("roman", [s.words for s in roman_train]), ("added", [r.words for r, _ in added])):
            lengths = [len(ids) for ids in tokenizer(sentences, is_split_into_words=True)["input_ids"]]
            pieces[name] = {
                "sentences": len(lengths),
                "total_pieces": sum(lengths),
                "median": statistics.median(lengths),
                "max": max(lengths),
                "over_256": sum(1 for length in lengths if length > 256),
            }

    stats = {
        "filtering": steps,
        "added_sentences": {"total": len(added), **kinds},
        "added_words_gold_labels": {script: {l: c[l] for l in LABELS} for script, c in gold.items()},
        "variants": variant_stats,
        "xlmr_pieces_per_sentence": pieces,
    }
    stats_path = resolve(args.stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"[written] {', '.join(v['path'] for v in variant_stats.values() if v['path'] != variant_stats['roman']['path'])}"
          f"; stats -> {os.path.relpath(stats_path, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
