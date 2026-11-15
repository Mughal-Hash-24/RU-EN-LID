"""Score one checkpoint on the Roman-only test split, the running examples and the demo slice.

Stage 2 runs once per sentence; Stage 3 is then applied at tau 0 (no rule can fire) and at the default tau,
exactly as in the pipeline. Writes a JSON report. Per-word predictions contain corpus text, so they go to a
git-ignored folder (--predictions-dir), never to the results folder.

    python -X utf8 experiments/score.py --checkpoint <dir> --out <report.json> [--predictions-dir <dir>]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.candidate_expansion.expander import CandidateExpander  # noqa: E402
from src.classifier.dataset import IGNORE_LABEL, read_any  # noqa: E402
from src.classifier.predict import TokenLIDPredictor  # noqa: E402
from src.eval.metrics import summarize  # noqa: E402
from src.pipeline import DEFAULT_RESOURCES  # noqa: E402
from src.verification.rules import DEFAULT_TAU, apply_rules  # noqa: E402

EVAL_SETS = {
    "test": "data/cache/comi-lingua-lid/roman/test.csv",
    "running_examples": "data/demo/running_examples.csv",
    "demo": "data/demo/seed_labels.csv",
}
AMBIGUOUS_FORMS = ("to", "he", "is", "do", "main", "band", "bar", "me", "or", "hi", "use", "bat")
LABELS = ("UR", "EN", "OTH")


def per_label(rows: list[dict]) -> dict:
    report = {}
    for label in LABELS:
        tp = sum(1 for r in rows if r["gold"] == label and r["pred"] == label)
        fp = sum(1 for r in rows if r["gold"] != label and r["pred"] == label)
        fn = sum(1 for r in rows if r["gold"] == label and r["pred"] != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    return report


def report_for(rows: list[dict], key: str, baseline_key: str | None) -> dict:
    scored = [{**row, "pred": row[key]} for row in rows if row["gold"] != IGNORE_LABEL]
    report = summarize([{**row, "pred": row[key]} for row in rows])
    report["per_label"] = per_label(scored)
    report["confusion"] = {f"{g}->{p}": n for (g, p), n in sorted(Counter((r["gold"], r["pred"]) for r in scored).items())}
    forms: dict[str, dict[str, list[int]]] = {}
    for row in scored:
        form = row["token"].lower()
        if form in AMBIGUOUS_FORMS:
            cell = forms.setdefault(form, {}).setdefault(row["gold"], [0, 0])
            cell[0] += row["pred"] == row["gold"]
            cell[1] += 1
    report["ambiguous_forms"] = {form: forms[form] for form in AMBIGUOUS_FORMS if form in forms}
    if baseline_key:
        changed = [r for r in scored if r[key] != r[baseline_key]]
        report["stage3_changes"] = {
            "changed": len(changed),
            "fixed": sum(1 for r in changed if r[key] == r["gold"]),
            "broke": sum(1 for r in changed if r[baseline_key] == r["gold"]),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True, help="JSON report path")
    parser.add_argument("--predictions-dir", help="write per-word predictions here (git-ignored location)")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--device", default="auto", help="cuda, cpu or auto")
    args = parser.parse_args(argv)

    import torch

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    expander = CandidateExpander.from_resources(DEFAULT_RESOURCES)
    predictor = TokenLIDPredictor(args.checkpoint, device=device)

    checkpoint = Path(args.checkpoint).resolve()
    local = checkpoint.exists() and checkpoint.is_relative_to(ROOT)  # a hub id is kept as given
    shown = checkpoint.relative_to(ROOT).as_posix() if local else str(args.checkpoint)
    result = {"checkpoint": shown, "tau": args.tau, "device": device, "sets": {}}
    for name, relative in EVAL_SETS.items():
        sentences = read_any(str(ROOT / relative))
        rows = []
        for sentence in sentences:
            tokens = expander.expand(" ".join(sentence.words))
            if len(tokens) != len(sentence.words):
                raise ValueError(f"{name} {sentence.sentence_id}: {len(sentence.words)} words, {len(tokens)} tokens")
            outputs = predictor.predict_words([t.surface for t in tokens], candidates=tokens)
            predictions = [(p.label, p.confidence) for p in outputs]
            stage2 = apply_rules(tokens, predictions, 0.0)
            stage3 = apply_rules(tokens, predictions, args.tau)
            for index, (info, plain, verified) in enumerate(zip(tokens, stage2, stage3)):
                rows.append({
                    "sentence_id": sentence.sentence_id,
                    "index": index,
                    "token": info.surface,
                    "gold": sentence.labels[index],
                    "category": sentence.categories[index] if sentence.categories else info.category,
                    "stage2": plain.label,
                    "confidence": round(plain.confidence, 4),
                    "stage3": verified.label,
                    "rules": "+".join(verified.rules),
                })
        result["sets"][name] = {
            "file": relative,
            "sentences": len(sentences),
            "stage2": report_for(rows, "stage2", None),
            "stage3": report_for(rows, "stage3", "stage2"),
        }
        if args.predictions_dir:
            out_dir = Path(args.predictions_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            with (out_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
        s2 = result["sets"][name]["stage2"]
        print(f"{name}: Stage 2 accuracy {s2['accuracy']:.4f}, macro-F1 {s2['macro_f1']:.4f} "
              f"({s2['tokens']} scored, {s2['ignored']} ignored)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[written] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
