"""Turn COMI-LINGUA's LID files into seed-format train / valid / test CSVs (Latin script only).

COMI-LINGUA is Hindi-English; its romanized sentences are the proxy for Roman Urdu-English
(hi -> UR). Gold labels are the 2-of-3 annotator majority ("Predicted tags" is ignored); words
without a majority are labelled IGN, so they stay as context but are never trained on or scored.
Train sentences that also occur in the test file are removed, and tokens are re-split with
the pipeline tokenizer so that scripts/evaluate.py sees exactly the stored words.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import re
import statistics
import sys
from collections import Counter
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.candidate_expansion.expander import tokenize  # noqa: E402
from src.classifier.dataset import IGNORE_LABEL  # noqa: E402

TEXT_COLUMN = "Sentences"
ANNOTATOR_COLUMNS = ("Annotated by: Annotator 1", "Annotated by: Annotator 2", "Annotated by: Annotator 3")
TAG_MAP = {"en": "EN", "e": "EN", "hi": "UR", "h": "UR", "ot": "OTH", "u": "OTH"}
LABELS = ("UR", "EN", "OTH", IGNORE_LABEL)
AMBIGUOUS_FORMS = ("main", "band", "is", "to", "do", "he", "bar")
OUTPUT_FIELDS = ["sentence_id", "token", "token_index", "label", "source_row"]
PARSE_FAILURES = ("unparseable_tags", "token_list_mismatch", "empty_annotation", "unknown_tag")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
ASCII_LETTER_RE = re.compile(r"[A-Za-z]")


@dataclass
class Record:
    source_row: int
    text: str
    words: list[str]
    labels: list[str]


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def is_latin(record: Record) -> bool:
    return not DEVANAGARI_RE.search(record.text) and bool(ASCII_LETTER_RE.search(record.text))


def token_key(record: Record) -> tuple[str, ...]:
    return tuple(word.lower() for word in record.words)


def parse_tags(raw: str) -> list[tuple[str, str]]:
    """One annotator column: a list of {'key', 'value'} dicts or (token, tag) pairs."""
    pairs = []
    for item in ast.literal_eval(raw):
        if isinstance(item, dict):
            pairs.append((str(item["key"]), str(item["value"])))
        else:
            token, tag = item
            pairs.append((str(token), str(tag)))
    return pairs


def majority_record(index: int, row: dict, notes: Counter, unknown_tags: Counter) -> Record | str:
    """Majority-labelled record for one CSV row, or the name of the check it fails."""
    try:
        annotations = [parse_tags(row[column] or "") for column in ANNOTATOR_COLUMNS]
    except (ValueError, SyntaxError, TypeError, KeyError):
        return "unparseable_tags"
    tokens = [token for token, _ in annotations[0]]
    if any([token for token, _ in other] != tokens for other in annotations[1:]):
        return "token_list_mismatch"
    if not tokens:
        return "empty_annotation"
    unknown = [tag for annotation in annotations for _, tag in annotation if tag not in TAG_MAP]
    if unknown:
        unknown_tags.update(unknown)
        return "unknown_tag"
    labels: list[str] = []
    for votes in zip(*([TAG_MAP[tag] for _, tag in annotation] for annotation in annotations)):
        label, count = Counter(votes).most_common(1)[0]
        if count < 2:
            label = IGNORE_LABEL
            notes["tokens_without_majority"] += 1
        labels.append(label)
    return Record(index, normalize_text(row[TEXT_COLUMN] or ""), tokens, labels)


def keep_where(steps: list[dict], step: str, records: list[Record], keep: Callable[[Record], bool]) -> list[Record]:
    kept = [record for record in records if keep(record)]
    steps.append({"step": step, "dropped": len(records) - len(kept), "remaining": len(kept)})
    return kept


def first_seen(key_of: Callable[[Record], Hashable]) -> Callable[[Record], bool]:
    seen: set = set()

    def keep(record: Record) -> bool:
        key = key_of(record)
        if key in seen:
            return False
        seen.add(key)
        return True

    return keep


def load(path: Path) -> tuple[list[Record], set[str], dict]:
    """Majority-labelled Latin-only records, every normalized sentence text, and the filter log."""
    failures: Counter = Counter()
    notes: Counter = Counter()
    unknown_tags: Counter = Counter()
    records: list[Record] = []
    texts: set[str] = set()
    rows = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {TEXT_COLUMN, *ANNOTATOR_COLUMNS} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for index, row in enumerate(reader):
            rows += 1
            texts.add(normalize_text(row[TEXT_COLUMN] or ""))
            result = majority_record(index, row, notes, unknown_tags)
            if isinstance(result, str):
                failures[result] += 1
            else:
                records.append(result)
    steps: list[dict] = [{"step": "rows", "remaining": rows}]
    remaining = rows
    for reason in PARSE_FAILURES:
        remaining -= failures[reason]
        steps.append({"step": reason, "dropped": failures[reason], "remaining": remaining})
    records = keep_where(steps, "not_latin_script", records, is_latin)
    log = {
        "path": str(path),
        "steps": steps,
        "tokens": {"without_majority": notes["tokens_without_majority"]},
        "unknown_tag_values": dict(unknown_tags),
    }
    return records, texts, log


def retokenize(records: list[Record], log: dict) -> list[Record]:
    """Re-split COMI tokens with the pipeline tokenizer; each piece inherits its token's label."""
    tokens = log["tokens"]
    tokens.update(comi_tokens_in=0, without_word_dropped=0, split_into_pieces=0, with_non_ascii=0, words_out=0)
    dropped_by_label: Counter = Counter()
    result = []
    for record in records:
        words: list[str] = []
        labels: list[str] = []
        for token, label in zip(record.words, record.labels):
            pieces = tokenize(token)
            tokens["comi_tokens_in"] += 1
            tokens["with_non_ascii"] += any(ord(char) > 127 for char in token)
            if not pieces:
                tokens["without_word_dropped"] += 1
                dropped_by_label[label] += 1
            tokens["split_into_pieces"] += len(pieces) > 1
            words.extend(pieces)
            labels.extend([label] * len(pieces))
        tokens["words_out"] += len(words)
        result.append(Record(record.source_row, record.text, words, labels))
    tokens["dropped_by_label"] = {label: dropped_by_label[label] for label in LABELS}
    steps = log["steps"]
    result = keep_where(steps, "no_tokens_left", result, lambda record: bool(record.words))
    return keep_where(steps, "no_ur_or_en_token", result, lambda record: "UR" in record.labels or "EN" in record.labels)


def split_stats(records: list[Record]) -> dict:
    lengths = [len(record.words) for record in records]
    labels = Counter(label for record in records for label in record.labels)
    return {
        "sentences": len(records),
        "tokens": sum(lengths),
        "labels": {label: labels[label] for label in LABELS},
        "code_mixed_sentences": sum(1 for record in records if "UR" in record.labels and "EN" in record.labels),
        "sentences_with_ignored": sum(1 for record in records if IGNORE_LABEL in record.labels),
        "median_length": float(statistics.median(lengths)) if lengths else 0.0,
        "max_length": max(lengths, default=0),
    }


def ambiguous_counts(records: list[Record]) -> dict:
    counts: dict[str, Counter] = {form: Counter() for form in AMBIGUOUS_FORMS}
    for record in records:
        for word, label in zip(record.words, record.labels):
            if word.lower() in counts:
                counts[word.lower()][label] += 1
    return {form: {label: counts[form][label] for label in LABELS} for form in AMBIGUOUS_FORMS}


def write_split(path: Path, records: list[Record]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(OUTPUT_FIELDS)
        for number, record in enumerate(records, start=1):
            for index, (word, label) in enumerate(zip(record.words, record.labels)):
                writer.writerow([f"s{number}", word, index, label, record.source_row])


def print_summary(stats: dict) -> None:
    config = stats["config"]
    print(f"COMI-LINGUA -> seed CSVs (seed {config['seed']}, valid fraction {config['valid_fraction']})")
    for role, log in stats["filtering"].items():
        print(f"\n{role}: {log['path']}")
        for step in log["steps"]:
            if "dropped" in step:
                print(f"  - {step['step']:<26} {step['dropped']:>6}  -> {step['remaining']:>6}")
            else:
                print(f"  {step['step']:<28} {step['remaining']:>6}")
        tokens = log["tokens"]
        print(
            f"  tokens: {tokens['without_majority']} without majority (labelled {IGNORE_LABEL}); "
            f"{tokens['comi_tokens_in']} COMI tokens -> {tokens['words_out']} words, "
            f"{tokens['without_word_dropped']} punctuation-only dropped {tokens['dropped_by_label']}, "
            f"{tokens['split_into_pieces']} split into pieces, {tokens['with_non_ascii']} with non-ASCII chars"
        )
        for key, value in log.get("notes", {}).items():
            print(f"  {key}: {value}")
    print(
        f"\n{'split':<6} {'sents':>6} {'tokens':>7} {'UR':>6} {'EN':>6} {'OTH':>6} {'IGN':>6} {'w/IGN':>6} "
        f"{'mixed':>6} {'median':>6} {'max':>4}"
    )
    for name, split in stats["splits"].items():
        labels = split["labels"]
        print(
            f"{name:<6} {split['sentences']:>6} {split['tokens']:>7} {labels['UR']:>6} {labels['EN']:>6} "
            f"{labels['OTH']:>6} {labels[IGNORE_LABEL]:>6} {split['sentences_with_ignored']:>6} "
            f"{split['code_mixed_sentences']:>6} {split['median_length']:>6} {split['max_length']:>4}"
        )
    names = list(stats["ambiguous_forms"])
    print("\nambiguous forms, UR/EN/OTH/IGN: " + "  ".join(f"{name:>16}" for name in names))
    for form in AMBIGUOUS_FORMS:
        cells = []
        for name in names:
            counts = stats["ambiguous_forms"][name][form]
            cells.append("/".join(str(counts[label]) for label in LABELS))
        print(f"  {form:<29} " + "  ".join(f"{cell:>16}" for cell in cells))


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/cache/comi-lingua-lid/LID_train.csv")
    parser.add_argument("--test", default="data/cache/comi-lingua-lid/LID_test.csv")
    parser.add_argument("--out-dir", default="data/cache/comi-lingua-lid/roman")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--subsets", default="", help="sizes of nested random train subsets, e.g. 1000,2000,5000")
    args = parser.parse_args(argv)
    if not 0.0 <= args.valid_fraction < 1.0:
        parser.error("--valid-fraction must be in [0, 1)")
    csv.field_size_limit(10**9)

    test_records, test_texts, test_log = load(resolve(args.test))
    test_records = retokenize(test_records, test_log)
    test_keys = Counter(token_key(record) for record in test_records)
    test_log["notes"] = {"duplicate_token_sequences_kept": sum(count - 1 for count in test_keys.values())}

    train_records, _, train_log = load(resolve(args.train))
    steps = train_log["steps"]
    train_records = keep_where(steps, "text_in_test_file", train_records, lambda record: record.text not in test_texts)
    train_records = keep_where(steps, "duplicate_text", train_records, first_seen(lambda record: record.text))
    train_records = retokenize(train_records, train_log)
    train_records = keep_where(
        steps, "token_sequence_in_test", train_records, lambda record: token_key(record) not in test_keys
    )
    train_records = keep_where(steps, "duplicate_token_sequence", train_records, first_seen(token_key))

    shuffled = list(train_records)
    random.Random(args.seed).shuffle(shuffled)
    valid_size = round(len(shuffled) * args.valid_fraction)
    splits = {
        "train": sorted(shuffled[valid_size:], key=lambda record: record.source_row),
        "valid": sorted(shuffled[:valid_size], key=lambda record: record.source_row),
        "test": test_records,
    }
    for name, records in splits.items():
        for record in records:
            if tokenize(" ".join(record.words)) != record.words:
                raise RuntimeError(f"{name} row {record.source_row}: words do not survive re-tokenization")

    # Nested subsets for learning curves: every smaller subset is a prefix of the same shuffled train split.
    sizes = sorted(int(size) for size in args.subsets.split(",") if size.strip())
    if sizes and sizes[-1] > len(splits["train"]):
        parser.error(f"--subsets: {sizes[-1]} is larger than the train split ({len(splits['train'])})")
    order = list(splits["train"])
    random.Random(args.seed).shuffle(order)
    subsets = {f"train_{size}": sorted(order[:size], key=lambda record: record.source_row) for size in sizes}

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, records in {**splits, **subsets}.items():
        write_split(out_dir / f"{name}.csv", records)
    stats = {
        "config": {
            "train": args.train,
            "test": args.test,
            "out_dir": args.out_dir,
            "seed": args.seed,
            "valid_fraction": args.valid_fraction,
        },
        "filtering": {"train_file": train_log, "test_file": test_log},
        "splits": {name: split_stats(records) for name, records in splits.items()},
        "ambiguous_forms": {name: ambiguous_counts(records) for name, records in splits.items()},
        "subsets": {name: split_stats(records) for name, records in subsets.items()},
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print_summary(stats)
    print(f"\n[written] {', '.join(f'{name}.csv' for name in {**splits, **subsets})}, stats.json -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
