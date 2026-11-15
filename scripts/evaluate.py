"""Evaluate the pipeline on a token-labelled CSV (seed slice by default)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classifier.dataset import read_any  # noqa: E402
from src.eval.metrics import summarize  # noqa: E402
from src.eval.selective import collect_rows, selective_markdown, selective_report  # noqa: E402
from src.pipeline import LIDPipeline, non_negative, probability  # noqa: E402
from src.verification.rules import DEFAULT_TAU  # noqa: E402


def markdown_report(report: dict, data_path: str, selective: dict | None = None) -> str:
    lines = [
        "# Evaluation report",
        "",
        f"Data: `{data_path}` ({report['tokens']} scored tokens)",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Token accuracy | {report['accuracy']:.4f} |",
        f"| Macro F1 | {report['macro_f1']:.4f} |",
        f"| CMI (gold labels, mean per sentence) | {report['cmi']:.2f} |",
        f"| Errors | {report['errors']} |",
    ]
    if report["ignored"]:
        lines.append(f"| Ignored (no gold label) | {report['ignored']} |")
    lines += [
        "",
        "## Per category",
        "",
        "| Category | Tokens | Accuracy | Macro F1 |",
        "|---|---|---|---|",
    ]
    for category, values in report["per_category"].items():
        lines.append(f"| {category} | {values['tokens']} | {values['accuracy']:.4f} | {values['macro_f1']:.4f} |")
    lines += ["", "## Error share by cause", "", "| Cause | Share of all errors |", "|---|---|"]
    for category, share in report["error_share_by_cause"].items():
        lines.append(f"| {category} | {share:.2%} |")
    lines.append("")
    if selective is not None:
        lines += selective_markdown(selective)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/demo/seed_labels.csv")
    parser.add_argument("--checkpoint", help="trained token-classification checkpoint (optional)")
    parser.add_argument(
        "--tau",
        type=probability,
        help=f"Stage 3 threshold (default: the value tuned for --checkpoint by scripts/tune_tau.py, "
        f"else {DEFAULT_TAU}; --tau 0 = Stage 2 alone)",
    )
    parser.add_argument(
        "--resources", help="Stage-1 lexicon folder (default: the one saved with --checkpoint, else data/resources)"
    )
    parser.add_argument(
        "--context-penalty", type=non_negative, default=0.0, help="Stage-3 switch penalty for deferred words (0 = off)"
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", help="write the markdown report to this path")
    args = parser.parse_args(argv)

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    display_path = Path(os.path.relpath(data_path, ROOT)).as_posix() if data_path.is_relative_to(ROOT) else str(data_path)
    sentences = read_any(str(data_path))
    pipeline = LIDPipeline.from_resources(
        args.resources, checkpoint=args.checkpoint, tau=args.tau, context_penalty=args.context_penalty
    )
    print(f"[stage1] resources = {pipeline.resources_dir} ({pipeline.resources_source})", file=sys.stderr)
    print(f"[stage3] tau = {pipeline.tau} ({pipeline.tau_source})", file=sys.stderr)
    rows = collect_rows(pipeline, sentences)

    report = summarize(rows)
    selective = selective_report(rows)
    if args.json:
        print(json.dumps({**report, "selective": selective}, indent=2))
    else:
        text = markdown_report(report, display_path, selective)
        print(text)
        if args.out:
            out_path = Path(args.out)
            if not out_path.is_absolute():
                out_path = ROOT / out_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text, encoding="utf-8")
            print(f"[written] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
