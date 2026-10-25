"""Evaluate the pipeline on a token-labelled CSV (seed slice by default)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classifier.dataset import read_any  # noqa: E402
from src.eval.metrics import summarize  # noqa: E402
from src.pipeline import LIDPipeline  # noqa: E402
from src.verification.rules import DEFAULT_TAU  # noqa: E402


def markdown_report(report: dict, data_path: str) -> str:
    lines = [
        f"# Evaluation report",
        "",
        f"Data: `{data_path}` ({report['tokens']} tokens, demo slice)",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Token accuracy | {report['accuracy']:.4f} |",
        f"| Macro F1 | {report['macro_f1']:.4f} |",
        f"| CMI (gold labels) | {report['cmi']:.2f} |",
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
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/demo/seed_labels.csv")
    parser.add_argument("--checkpoint", help="trained token-classification checkpoint (optional)")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", help="write the markdown report to this path")
    args = parser.parse_args(argv)

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    sentences = read_any(str(data_path))
    pipeline = LIDPipeline.from_resources(checkpoint=args.checkpoint, tau=args.tau)

    rows = []
    for sentence in sentences:
        pairs = pipeline.run(" ".join(sentence.words))
        if len(pairs) != len(sentence.words):
            print(f"[warn] tokenization mismatch for {sentence.sentence_id}; skipped", file=sys.stderr)
            continue
        for index, (info, decision) in enumerate(pairs):
            rows.append(
                {
                    "token": decision.token,
                    "gold": sentence.labels[index],
                    "pred": decision.label,
                    "category": (sentence.categories[index] if sentence.categories else info.category),
                }
            )

    report = summarize(rows)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        text = markdown_report(report, str(data_path))
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
