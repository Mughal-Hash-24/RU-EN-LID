"""Fetch public datasets into data/cache/ (never committed to git).

Usage:
    python data/fetch_data.py --list
    python data/fetch_data.py --dataset uci-roman-urdu
    python data/fetch_data.py --dataset comi-lingua-lid --yes-large
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"

DATASETS = {
    "uci-roman-urdu": {
        "url": "https://archive.ics.uci.edu/static/public/458/roman+urdu+data+set.zip",
        "filename": "roman-urdu-data-set.zip",
        "licence": "CC BY 4.0",
        "note": "Main RU-EN pool (sentiment labels only; no word-level LID labels).",
        "large": False,
    },
    "comi-lingua-lid": {
        "urls": [
            "https://media.githubusercontent.com/media/lingo-iitgn/COMI-LINGUA/main/LID_train.csv",
            # LID_test.csv is a plain git file, not Git LFS, so the LFS media URL returns 404.
            "https://raw.githubusercontent.com/lingo-iitgn/COMI-LINGUA/main/LID_test.csv",
        ],
        "licence": "CC BY-SA 4.0",
        "note": "Proxy token-level LID (Hindi-English; Latin-script subset trains and evaluates Stage 2).",
        "large": True,
    },
    "mixlid": {
        "url": "https://huggingface.co/nlpai-lab/MixLID",
        "licence": "MIT",
        "note": "Off-the-shelf token-level LID baseline; load through transformers instead of downloading here.",
        "large": False,
        "skip_download": True,
    },
}


def describe() -> None:
    print(f"{'name':<18} {'large':<6} licence")
    for name, spec in DATASETS.items():
        print(f"{name:<18} {str(spec['large']):<6} {spec['licence']} - {spec['note']}")


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}\n        -> {target}")
    urllib.request.urlretrieve(url, target)
    print(f"done ({target.stat().st_size / 1e6:.1f} MB)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list known datasets")
    parser.add_argument("--dataset", help="dataset key to download")
    parser.add_argument("--yes-large", action="store_true", help="confirm large downloads (>100 MB)")
    args = parser.parse_args(argv)

    if args.list or not args.dataset:
        describe()
        return 0

    spec = DATASETS.get(args.dataset)
    if spec is None:
        print(f"unknown dataset: {args.dataset}", file=sys.stderr)
        return 2
    if spec.get("skip_download"):
        print(f"{args.dataset}: {spec['note']}")
        return 0
    if spec["large"] and not args.yes_large:
        print(f"{args.dataset} is a large download; re-run with --yes-large", file=sys.stderr)
        return 2

    urls = spec.get("urls") or [spec["url"]]
    for url in urls:
        download(url, CACHE / args.dataset / url.rsplit("/", 1)[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
