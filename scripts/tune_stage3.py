"""Tune Stage 3 on VALIDATION data: the confidence gate tau and the context penalty.

Grid-searches tau x penalty, prints every cell, and reports the best by final
accuracy (ties -> fewer Stage-3 breaks, then the smaller penalty). Report the
chosen pair, then evaluate it once on test with scripts/evaluate.py. Never
tune on test.

    python scripts/tune_stage3.py --data <valid.csv> --checkpoint <ckpt>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classifier.dataset import read_any  # noqa: E402
from src.eval.selective import collect_rows, selective_report  # noqa: E402
from src.pipeline import DEFAULT_RESOURCES, LIDPipeline, non_negative, probability  # noqa: E402

TAUS = [0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95]
PENALTIES = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="validation CSV (never the test split)")
    parser.add_argument("--checkpoint", help="Stage-2 checkpoint (omit for the heuristic fallback)")
    parser.add_argument("--resources", default=DEFAULT_RESOURCES)
    parser.add_argument("--taus", type=probability, nargs="+", default=TAUS)
    parser.add_argument("--penalties", type=non_negative, nargs="+", default=PENALTIES)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if "test" in Path(args.data).name.lower():
        print(f"[warning] {args.data} looks like a test split; tune on validation only", file=sys.stderr)

    sentences = read_any(args.data)
    pipeline = LIDPipeline.from_resources(args.resources, checkpoint=args.checkpoint)
    # Stage 2 runs once per sentence; only the gate and Stage 3 change across the grid.
    if pipeline.predictor is not None:
        cache: dict[tuple[str, ...], list] = {}
        predict = pipeline.predictor.predict_words

        def cached(words, candidates=None):
            key = tuple(words)
            if key not in cache:
                cache[key] = predict(words, candidates=candidates)
            return cache[key]

        pipeline.predictor.predict_words = cached

    grid = []
    for tau in args.taus:
        for penalty in args.penalties:
            pipeline.tau, pipeline.context_penalty = tau, penalty
            report = selective_report(collect_rows(pipeline, sentences))
            grid.append(
                {
                    "tau": tau,
                    "penalty": penalty,
                    "final_accuracy": report["final_accuracy"],
                    "coverage": report["coverage"],
                    "fixes": report["stage3_fixes"],
                    "breaks": report["stage3_breaks"],
                }
            )
    best = max(grid, key=lambda cell: (cell["final_accuracy"] or 0.0, -cell["breaks"], -cell["penalty"]))
    if args.json:
        print(json.dumps({"best": best, "grid": grid}, indent=2))
        return 0
    print(f"{'tau':>6}{'penalty':>9}{'final acc':>11}{'coverage':>10}{'fixes':>7}{'breaks':>8}")
    for cell in grid:
        marker = "  <- best" if cell is best else ""
        print(
            f"{cell['tau']:>6.2f}{cell['penalty']:>9.2f}{cell['final_accuracy']:>11.4f}"
            f"{cell['coverage']:>10.4f}{cell['fixes']:>7}{cell['breaks']:>8}{marker}"
        )
    print(f"\nbest: tau={best['tau']} context_penalty={best['penalty']} (validation only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
