"""Token-level LID dataset readers.

Supports the repository's seed format (sentence_id, token, token_index, label,
category) and proxy CSVs such as COMI-LINGUA's LID files, whose schema is
detected from the header.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass

PROXY_LABEL_MAP = {"hi": "UR", "hin": "UR", "ur": "UR", "en": "EN", "eng": "EN", "ot": "OTH", "other": "OTH"}
# Gold label for words the annotators could not agree on: kept as context, never trained on or scored.
IGNORE_LABEL = "IGN"
GOLD_LABELS = frozenset({"UR", "EN", "OTH", IGNORE_LABEL})


def _sentence_order(sentence_id: str) -> tuple[int, int, str]:
    """Numeric ids (s2 before s10) first, then any other id alphabetically; never mixes int and str."""
    digits = sentence_id.lstrip("s")
    return (0, int(digits), "") if digits.isdigit() else (1, 0, sentence_id)


@dataclass
class Sentence:
    sentence_id: str
    words: list[str]
    labels: list[str]
    categories: list[str] | None = None


def read_seed_csv(path: str) -> list[Sentence]:
    grouped: dict[str, list[tuple[int, str, str, str | None]]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = {name.strip().lower() for name in (reader.fieldnames or [])}
        missing = {"sentence_id", "token", "token_index", "label"} - fields
        if missing:
            raise ValueError(f"seed CSV missing columns: {sorted(missing)}")
        for line, row in enumerate(reader, start=2):  # line 1 is the header
            row = {key.strip().lower(): (value or "").strip() for key, value in row.items() if key}
            label = row["label"].upper()
            if label not in GOLD_LABELS:
                raise ValueError(f"{path}:{line}: unknown label {row['label']!r}; expected {sorted(GOLD_LABELS)}")
            try:
                index = int(row["token_index"])
            except ValueError:
                raise ValueError(f"{path}:{line}: token_index {row['token_index']!r} is not an integer") from None
            grouped.setdefault(row["sentence_id"], []).append((index, row["token"], label, row.get("category") or None))
    sentences: list[Sentence] = []
    for sentence_id in sorted(grouped, key=_sentence_order):
        rows = sorted(grouped[sentence_id], key=lambda item: item[0])
        indices, words, labels, categories = (list(column) for column in zip(*rows, strict=True))
        if len(set(indices)) != len(indices):
            raise ValueError(f"{path}: sentence {sentence_id!r} repeats a token_index")
        sentences.append(
            Sentence(
                sentence_id=sentence_id,
                words=words,
                labels=labels,
                categories=categories if all(categories) else None,
            )
        )
    return sentences


def read_proxy_lid_csv(path: str, label_map: dict[str, str] | None = None) -> list[Sentence]:
    mapping = dict(PROXY_LABEL_MAP)
    if label_map:
        mapping.update({key.lower(): value for key, value in label_map.items()})
    sentences: list[Sentence] = []
    skipped = 0
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = {name.strip().lower(): name for name in (reader.fieldnames or [])}
        text_column = next((columns[key] for key in ("sentence", "text", "words", "tokens") if key in columns), None)
        tag_column = next(
            (columns[key] for key in ("lid", "labels", "tags", "lid tags", "language") if key in columns), None
        )
        if text_column is None or tag_column is None:
            raise ValueError(f"could not detect token/tag columns in {columns.keys()}")
        for index, row in enumerate(reader):
            words = (row.get(text_column) or "").split()
            raw_tags = (row.get(tag_column) or "").split()
            if not words or len(words) != len(raw_tags):
                skipped += 1
                continue
            labels = [mapping.get(tag.lower(), "OTH") for tag in raw_tags]
            sentences.append(Sentence(sentence_id=f"row{index}", words=words, labels=labels))
    if skipped:
        print(f"[dataset] skipped {skipped} malformed rows in {path}", file=sys.stderr)
    return sentences


def read_any(path: str, label_map: dict[str, str] | None = None) -> list[Sentence]:
    with open(path, newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle), [])
    columns = {name.strip().lower() for name in header}
    if {"sentence_id", "token", "token_index"} <= columns:
        return read_seed_csv(path)
    return read_proxy_lid_csv(path, label_map=label_map)
