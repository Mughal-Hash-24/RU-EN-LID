"""Stage-2 proof-of-concept entry point.

--dry-run builds the data pipeline and augmentation statistics without torch.
A real (or smoke) training run forwards to src.classifier.train.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.candidate_expansion.expander import CandidateExpander  # noqa: E402
from src.classifier.augment import augment_sentence, oversample_ambiguous, synthetic_variants  # noqa: E402
from src.classifier.dataset import read_any  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="build/augment data and print statistics (no torch)")
    parser.add_argument("--train", default="data/demo/seed_labels.csv")
    parser.add_argument("--valid")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model", default="xlm-roberta-base")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--out", default="checkpoints/ruen-lid")
    return parser


def _resolve(path: str) -> str:
    candidate = Path(path)
    return str(candidate if candidate.is_absolute() else ROOT / candidate)


def dry_run(args: argparse.Namespace) -> int:
    sentences = read_any(_resolve(args.train))
    expander = CandidateExpander.from_resources(str(ROOT / "data" / "resources"))
    augmented = [augment_sentence(sentence.words, sentence.labels, probability=0.5) for sentence in sentences]
    oversampled = oversample_ambiguous(sentences, set(expander.taxonomy.ambiguous))
    total_tokens = sum(len(sentence.words) for sentence in sentences)
    print(f"loaded {len(sentences)} sentences / {total_tokens} tokens from {args.train}")
    print(f"augmented corpus: {len(augmented)} sentences")
    print(f"after ambiguity oversampling: {len(oversampled)} sentences")
    print(f"ambiguous forms: {', '.join(sorted(expander.taxonomy.ambiguous))}")
    label_counts: dict[str, int] = {}
    for sentence in sentences:
        for label in sentence.labels:
            label_counts[label] = label_counts.get(label, 0) + 1
    print(f"label distribution: {label_counts}")
    example_index = next((i for i, (words, _) in enumerate(augmented) if words != sentences[i].words), 0)
    original = sentences[example_index]
    words, labels = augmented[example_index]
    changed = sum(1 for before, after in zip(original.words, words) if before != after)
    print(f"\nexample augmentation ({original.sentence_id}, {changed} spelling change(s)):")
    for word, new_word, label in zip(original.words, words, labels):
        marker = " (variant)" if word != new_word else ""
        print(f"  {word:<10} -> {new_word:<10} [{label}]{marker}")
    print("\nexample synthetic variants for 'bohat':", synthetic_variants("bohat"))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.dry_run:
        return dry_run(args)
    from src.classifier.train import main as train_main

    forwarded = ["--train", _resolve(args.train), "--out", _resolve(args.out), "--epochs", str(args.epochs), "--model", args.model]
    if args.valid:
        forwarded += ["--valid", _resolve(args.valid)]
    if args.smoke:
        forwarded += ["--smoke"]
    if args.max_steps >= 0:
        forwarded += ["--max-steps", str(args.max_steps)]
    return train_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
