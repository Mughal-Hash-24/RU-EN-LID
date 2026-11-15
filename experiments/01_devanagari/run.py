"""Experiment 01 runner: train every variant with several seeds, score each run, then delete its model.

Variants (training files from build_data.py; validation and test are the Roman-only files for all three):
    roman      today's training data (6,540 Roman-script sentences), the baseline, re-run in this harness
    deva_oth   + 18,538 sentences with Devanagari; Devanagari words labelled OTH
    deva_ign   + the same sentences; Devanagari words labelled IGN (context only)

Training settings are the learning curve's (scripts/learning_curve.py). Each run's report goes to
<results>/scores/<run>.json; per-word predictions go to the git-ignored --predictions folder.
<results>/runs.json and summary.md are rewritten after every run, so partial results are readable.

    python -X utf8 experiments/01_devanagari/run.py                     # all 9 runs
    python -X utf8 experiments/01_devanagari/run.py --variants deva_oth --seeds 1 --max-steps 40 --tag rehearsal
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import learning_curve as lc  # noqa: E402  (run(), best_epoch(), mean_sd(), CHILD_ENV)

HERE = Path(__file__).resolve().parent
TRAIN_FILES = {
    "roman": "data/cache/comi-lingua-lid/roman/train.csv",
    "deva_oth": "data/cache/experiments/01_devanagari/deva_oth/train.csv",
    "deva_ign": "data/cache/experiments/01_devanagari/deva_ign/train.csv",
}
VALID_FILE = "data/cache/comi-lingua-lid/roman/valid.csv"


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def count_sentences(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return len({row["sentence_id"] for row in csv.DictReader(handle)})


def pooled_forms(rows: list[dict], set_name: str = "test") -> dict:
    """Ambiguous-form counts summed over runs: {form: {gold: [correct, total]}}."""
    pooled: dict[str, dict[str, list[int]]] = {}
    for row in rows:
        for form, by_gold in row["scores"]["sets"][set_name]["stage2"]["ambiguous_forms"].items():
            for gold, (correct, total) in by_gold.items():
                cell = pooled.setdefault(form, {}).setdefault(gold, [0, 0])
                cell[0] += correct
                cell[1] += total
    return pooled


def markdown(rows: list[dict], config: dict, variants: list[str]) -> str:
    lines = [
        "# Experiment 01: keep the Devanagari sentences",
        "",
        "Test = the Roman-only COMI-LINGUA test split (750 sentences, 15,207 scored words) for every variant.",
        "Mean ± standard deviation over seeds. Stage 2 alone = tau 0; + Stage 3 = default tau.",
        "",
        f"Settings: `{json.dumps(config)}`",
        "",
        "## Overall",
        "",
        "| Variant | Runs | Train sentences | Valid macro-F1 | Test acc. | Test macro-F1 | UR recall | EN recall "
        "| OTH F1 | Test acc. + Stage 3 | Running ex. /25 | Demo /51 | Minutes |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for variant in variants:
        group = [row for row in rows if row["variant"] == variant]
        if not group:
            continue
        test = [row["scores"]["sets"]["test"] for row in group]
        correct = lambda name: [  # noqa: E731
            row["scores"]["sets"][name]["stage2"]["tokens"] - row["scores"]["sets"][name]["stage2"]["errors"]
            for row in group
        ]
        lines.append(
            f"| {variant} | {len(group)} | {group[0]['train_sentences']} "
            f"| {lc.mean_sd([r['valid_macro_f1'] for r in group])} "
            f"| {lc.mean_sd([t['stage2']['accuracy'] for t in test])} "
            f"| {lc.mean_sd([t['stage2']['macro_f1'] for t in test])} "
            f"| {lc.mean_sd([t['stage2']['per_label']['UR']['recall'] for t in test])} "
            f"| {lc.mean_sd([t['stage2']['per_label']['EN']['recall'] for t in test])} "
            f"| {lc.mean_sd([t['stage2']['per_label']['OTH']['f1'] for t in test])} "
            f"| {lc.mean_sd([t['stage3']['accuracy'] for t in test])} "
            f"| {', '.join(str(c) for c in correct('running_examples'))} "
            f"| {', '.join(str(c) for c in correct('demo'))} "
            f"| {sum(r['minutes'] for r in group) / len(group):.1f} |"
        )
    lines += [
        "",
        "## Two-language words on the test split (Stage 2, correct / total, summed over runs)",
        "",
        "| Word | Gold | " + " | ".join(variants) + " |",
        "|---|---|" + "---|" * len(variants),
    ]
    pooled = {variant: pooled_forms([r for r in rows if r["variant"] == variant]) for variant in variants}
    forms = list(dict.fromkeys(form for p in pooled.values() for form in p))
    for form in forms:
        golds = sorted({gold for p in pooled.values() for gold in p.get(form, {})}, key=["UR", "EN", "OTH"].index)
        for gold in golds:
            cells = []
            for variant in variants:
                correct_total = pooled[variant].get(form, {}).get(gold)
                cells.append(f"{correct_total[0]}/{correct_total[1]}" if correct_total else "–")
            lines.append(f"| *{form}* | {gold} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Runs",
        "",
        "| Variant | Seed | Best epoch | Valid macro-F1 | Test acc. / macro-F1 (Stage 2) | Stage 3 changes "
        "(fixed / broke) | Minutes |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        test = row["scores"]["sets"]["test"]
        changes = test["stage3"]["stage3_changes"]
        lines.append(
            f"| {row['variant']} | {row['seed']} | {row['best_epoch']} | {row['valid_macro_f1']:.4f} "
            f"| {test['stage2']['accuracy']:.4f} / {test['stage2']['macro_f1']:.4f} "
            f"| {changes['changed']} ({changes['fixed']} / {changes['broke']}) | {row['minutes']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", default="roman,deva_oth,deva_ign")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--model", default="xlm-roberta-base")
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="rehearsal only: stop after this many steps")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tag", default="", help="suffix for the results folder, e.g. 'rehearsal'")
    parser.add_argument("--work-dir", default="checkpoints/experiments/01_devanagari")
    parser.add_argument("--predictions", default="data/cache/experiments/01_devanagari/predictions")
    parser.add_argument("--keep-models", action="store_true", help="keep each run's model instead of deleting it")
    args = parser.parse_args(argv)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in TRAIN_FILES]
    if unknown:
        parser.error(f"unknown variants {unknown}; choose from {list(TRAIN_FILES)}")
    missing = [TRAIN_FILES[v] for v in variants if not resolve(TRAIN_FILES[v]).exists()]
    if missing:
        parser.error(f"missing training files (run build_data.py first): {missing}")
    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]
    results = HERE / ("results" if not args.tag else f"results_{args.tag}")
    work_dir, predictions = resolve(args.work_dir), resolve(args.predictions)
    for folder in (results / "logs", results / "scores", work_dir):
        folder.mkdir(parents=True, exist_ok=True)
    config = {key: getattr(args, key) for key in ("model", "epochs", "max_steps", "batch_size", "learning_rate",
                                                  "warmup_ratio", "weight_decay", "max_length", "bf16")}
    sentence_counts = {v: count_sentences(resolve(TRAIN_FILES[v])) for v in variants}

    rows: list[dict] = []
    for seed in seeds:
        for variant in variants:
            name = f"{variant}_seed{seed}"
            out = work_dir / (name + (f"_{args.tag}" if args.tag else ""))
            log = results / "logs" / f"{name}.log"
            log.write_text("", encoding="utf-8")
            started = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] training {name} ({sentence_counts[variant]} sentences)", flush=True)
            command = [
                sys.executable, "-m", "src.classifier.train", "--train", str(resolve(TRAIN_FILES[variant])),
                "--valid", str(resolve(VALID_FILE)), "--model", args.model, "--out", str(out),
                "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                "--learning-rate", str(args.learning_rate), "--warmup-ratio", str(args.warmup_ratio),
                "--weight-decay", str(args.weight_decay), "--max-length", str(args.max_length),
                "--seed", str(seed), "--logging-steps", "50", "--select-best",
            ]
            if args.max_steps > 0:
                command += ["--max-steps", str(args.max_steps)]
            if args.bf16:
                command.append("--bf16")
            lc.run(command, log)
            for leftover in out.glob("checkpoint-*"):
                shutil.rmtree(leftover)
            summary = json.loads((out / "training_summary.json").read_text(encoding="utf-8"))
            trained_minutes = (time.time() - started) / 60
            print(f"[{time.strftime('%H:%M:%S')}] scoring {name}", flush=True)
            score_path = results / "scores" / f"{name}.json"
            lc.run(
                [sys.executable, str(HERE.parent / "score.py"), "--checkpoint", str(out), "--out", str(score_path),
                 "--predictions-dir", str(predictions / (name + (f"_{args.tag}" if args.tag else "")))],
                log,
            )
            scores = json.loads(score_path.read_text(encoding="utf-8"))
            row = {
                "variant": variant,
                "seed": seed,
                "train_sentences": sentence_counts[variant],
                "best_epoch": lc.best_epoch(summary),
                "valid_macro_f1": summary["final_eval"]["eval_macro_f1"],
                "valid_accuracy": summary["final_eval"]["eval_token_accuracy"],
                "eval_history": [
                    {k: entry[k] for k in ("epoch", "eval_loss", "eval_macro_f1", "eval_token_accuracy") if k in entry}
                    for entry in summary["eval_history"]
                ],
                "train_minutes": trained_minutes,
                "minutes": (time.time() - started) / 60,
                "scores": scores,
            }
            rows.append(row)
            if not args.keep_models:
                shutil.rmtree(out)
            (results / "runs.json").write_text(json.dumps({"config": config, "runs": rows}, indent=2) + "\n",
                                               encoding="utf-8")
            (results / "summary.md").write_text(markdown(rows, config, variants), encoding="utf-8")
            test = scores["sets"]["test"]["stage2"]
            print(f"  valid macro-F1 {row['valid_macro_f1']:.4f}; test acc {test['accuracy']:.4f}, "
                  f"macro-F1 {test['macro_f1']:.4f}; {row['minutes']:.1f} min", flush=True)
    print(f"[written] {results / 'summary.md'}, {results / 'runs.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
