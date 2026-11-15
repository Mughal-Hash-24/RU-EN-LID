"""Build an expanded Stage-1 resource folder: hand-built seeds + words induced from training data.

    python scripts/build_lexicon.py --train <train.csv> --out data/resources_induced \
        [--check <test.csv>]

The output folder has the same three files as data/resources, so every tool takes it
through --resources (train.py, evaluate.py, tune_stage3.py, pipeline). The seeds are
never edited. Induced ambiguous words go into both lists, which is how Stage 1 defines
"ambiguous". A word the seeds put in one language but the data puts in the other
also ends up in both lists; these conflicts are counted in induction_report.json.
Use the TRAINING split only (see src/candidate_expansion/induce.py).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.candidate_expansion.induce import induce  # noqa: E402
from src.candidate_expansion.lexicons import EXCLUSIVELY_ROMAN_URDU, read_wordlist  # noqa: E402
from src.classifier.dataset import read_any  # noqa: E402
from src.pipeline import DEFAULT_RESOURCES  # noqa: E402


def write_list(path: Path, words: set[str], note: str) -> None:
    path.write_text(f"# {note}\n" + "\n".join(sorted(words)) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True, help="training CSV (never validation or test)")
    parser.add_argument("--out", required=True, help="new resource folder to write")
    parser.add_argument("--seeds", default=DEFAULT_RESOURCES, help="seed resource folder (default: data/resources)")
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--purity", type=float, default=0.9)
    parser.add_argument("--ambiguous-share", type=float, default=0.2)
    parser.add_argument("--check", help="optional held-out CSV: report how much of it the lexicon covers")
    args = parser.parse_args(argv)
    if any(part in Path(args.train).name.lower() for part in ("test", "valid", "dev")):
        print(f"[warning] {args.train} does not look like a training split", file=sys.stderr)

    seeds = Path(args.seeds)
    out = Path(args.out)
    if out.resolve() == seeds.resolve():
        raise SystemExit("--out must differ from the seed folder; the seeds are never overwritten")
    seed_en = read_wordlist(str(seeds / "english_seed.txt"))
    seed_ur = read_wordlist(str(seeds / "roman_urdu_seed.txt"))

    lexicon = induce(read_any(args.train), args.min_count, args.purity, args.ambiguous_share)
    english = seed_en | lexicon.english | lexicon.ambiguous
    urdu = seed_ur | lexicon.roman_urdu | lexicon.ambiguous
    # Same guard Stage 1 applies on load; reported so the numbers below match what Stage 1 sees.
    guarded = {word for word in english if word in EXCLUSIVELY_ROMAN_URDU}
    # Seeds put the word in one language only, the data in the other (both-list seeds are not conflicts).
    conflicts = sorted(((seed_en - seed_ur) & lexicon.roman_urdu) | ((seed_ur - seed_en) & lexicon.english))

    out.mkdir(parents=True, exist_ok=True)
    note = f"seeds + induced from {Path(args.train).name}"
    write_list(out / "english_seed.txt", english, note)
    write_list(out / "roman_urdu_seed.txt", urdu, note)
    if (seeds / "variant_pairs.tsv").exists():
        shutil.copy(seeds / "variant_pairs.tsv", out / "variant_pairs.tsv")

    report = {
        "train": str(args.train),
        "thresholds": {"min_count": args.min_count, "purity": args.purity, "ambiguous_share": args.ambiguous_share},
        "word_types_seen": len(lexicon.counts),
        "induced": {
            "english": len(lexicon.english),
            "roman_urdu": len(lexicon.roman_urdu),
            "ambiguous": len(lexicon.ambiguous),
        },
        "skipped": {"rare": lexicon.skipped_rare, "mixed": lexicon.skipped_mixed},
        "final_sizes": {
            "english": len(english - guarded),
            "roman_urdu": len(urdu),
            "ambiguous": len((english - guarded) & urdu),
        },
        "seed_vs_data_conflicts": conflicts,
        "ambiguous_words": sorted(lexicon.ambiguous),
    }
    if args.check:
        tokens = [word.lower() for sentence in read_any(args.check) for word in sentence.words]
        known = english | urdu
        report["check"] = {
            "file": str(args.check),
            "tokens": len(tokens),
            "covered_by_seeds": round(sum(t in seed_en | seed_ur for t in tokens) / max(len(tokens), 1), 4),
            "covered_after_induction": round(sum(t in known for t in tokens) / max(len(tokens), 1), 4),
        }
    (out / "induction_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "ambiguous_words"}, indent=2))
    print(f"[written] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
