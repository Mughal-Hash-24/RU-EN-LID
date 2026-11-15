"""Experiment 02: train for longer, so the curves show where they level off or drop.

Iteration 1 stopped every run at 5 epochs, and the best validation epoch was the last one in 10 of 12 runs: the
curves were cut off at the limit we chose. This re-runs the same 12 learning-curve runs (sizes 1k / 2k / 5k / all,
seeds 1-3) with the same settings, except up to 15 epochs. The learning rate still decays linearly to 0 over all
steps, so a 15-epoch run is not a 5-epoch run continued: its learning rate falls more slowly.

Per run it records the validation history (loss, macro-F1, accuracy after every epoch), the training loss every
50 steps, and the test scores of the best-validation epoch (experiments/score.py). Models are deleted after
scoring, except the best full-data one (--keep).

    python -X utf8 experiments/02_longer_training/run.py
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import learning_curve as lc  # noqa: E402  (run(), mean_sd())

HERE = Path(__file__).resolve().parent
DATA = ROOT / "data/cache/comi-lingua-lid/roman"
ITERATION1 = ROOT / "results/learning_curve/runs.json"


def train_file(size: str) -> Path:
    return DATA / ("train.csv" if size == "all" else f"train_{size}.csv")


def count_sentences(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        next(handle)
        return len({line.split(",", 1)[0] for line in handle})


def epoch_history(summary: dict) -> list[dict]:
    """One entry per epoch (the final evaluate() repeats the best epoch's scores at the last step: dropped)."""
    seen: dict[float, dict] = {}
    for entry in summary["eval_history"]:
        seen.setdefault(entry["epoch"], entry)
    return [
        {"epoch": round(e["epoch"]), "step": e["step"], "loss": e["eval_loss"], "macro_f1": e["eval_macro_f1"],
         "accuracy": e["eval_token_accuracy"]}
        for e in sorted(seen.values(), key=lambda e: e["epoch"])
    ]


def training_loss(log_text: str, steps_per_epoch: float) -> list[dict]:
    entries = [ast.literal_eval(m) for m in re.findall(r"\{'loss'[^}]*\}", log_text)]
    return [{"step": round(e["epoch"] * steps_per_epoch), "loss": e["loss"], "learning_rate": e["learning_rate"]}
            for e in entries]


def markdown(rows: list[dict], config: dict) -> str:
    old = {}
    if ITERATION1.exists():
        for r in json.loads(ITERATION1.read_text(encoding="utf-8"))["runs"]:
            old.setdefault(r["train_sentences"], []).append(r)
    lines = [
        "# Experiment 02: train for longer (up to 15 epochs)",
        "",
        "Test = the Roman-only test split, Stage 2 alone (tau 0), at the best-validation epoch.",
        "Mean ± standard deviation over seeds. \"Up to 5\" = iteration 1 (`results/learning_curve/`).",
        "",
        f"Settings: `{json.dumps(config)}`",
        "",
        "## Learning curve",
        "",
        "| Train sentences | Runs | Best epochs | Valid macro-F1 | Test acc. (up to 15) | Test acc. (up to 5) "
        "| Test macro-F1 (up to 15) | Test macro-F1 (up to 5) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for size in sorted({r["train_sentences"] for r in rows}):
        group = [r for r in rows if r["train_sentences"] == size]
        before = old.get(size, [])
        test = [r["scores"]["sets"]["test"]["stage2"] for r in group]
        lines.append(
            f"| {size} | {len(group)} | {', '.join(str(r['best_epoch']) for r in group)} "
            f"| {lc.mean_sd([r['valid_macro_f1'] for r in group])} "
            f"| {lc.mean_sd([t['accuracy'] for t in test])} | {lc.mean_sd([r['stage2']['accuracy'] for r in before])} "
            f"| {lc.mean_sd([t['macro_f1'] for t in test])} | {lc.mean_sd([r['stage2']['macro_f1'] for r in before])} |"
        )
    lines += ["", "## Validation after every epoch (mean over seeds)", ""]
    sizes = sorted({r["train_sentences"] for r in rows}, reverse=True)
    epochs = sorted({e["epoch"] for r in rows for e in r["history"]})
    lines.append("| Epoch | " + " | ".join(f"{s}: macro-F1 | {s}: loss" for s in sizes) + " |")
    lines.append("|---|" + "---|---|" * len(sizes))
    for epoch in epochs:
        cells = []
        for size in sizes:
            values = [e for r in rows if r["train_sentences"] == size for e in r["history"] if e["epoch"] == epoch]
            cells.append(f"{statistics.mean(v['macro_f1'] for v in values):.4f}" if values else "–")
            cells.append(f"{statistics.mean(v['loss'] for v in values):.4f}" if values else "–")
        lines.append(f"| {epoch} | " + " | ".join(cells) + " |")
    lines += ["", "## Runs", "", "| Train sentences | Seed | Best epoch | Valid macro-F1 | Test acc. / macro-F1 "
              "| Running ex. /25 | Demo /51 | Minutes |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sets = r["scores"]["sets"]
        correct = {k: sets[k]["stage2"]["tokens"] - sets[k]["stage2"]["errors"] for k in ("running_examples", "demo")}
        t = sets["test"]["stage2"]
        lines.append(f"| {r['train_sentences']} | {r['seed']} | {r['best_epoch']} | {r['valid_macro_f1']:.4f} "
                     f"| {t['accuracy']:.4f} / {t['macro_f1']:.4f} | {correct['running_examples']} "
                     f"| {correct['demo']} | {r['minutes']:.1f} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", default="all,5000,2000,1000", help="run order; 'all' = train.csv")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--model", default="xlm-roberta-base")
    parser.add_argument("--epochs", type=float, default=15.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--work-dir", default="checkpoints/experiments/02_longer_training")
    parser.add_argument("--keep", default="checkpoints/experiments/02_longer_training/best_full_data")
    parser.add_argument("--predictions", default="data/cache/experiments/02_longer_training/predictions")
    args = parser.parse_args(argv)

    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    work_dir, keep, predictions = ROOT / args.work_dir, ROOT / args.keep, ROOT / args.predictions
    results = HERE / "results"
    for folder in (results / "logs", results / "scores", work_dir):
        folder.mkdir(parents=True, exist_ok=True)
    counts = {size: count_sentences(train_file(size)) for size in sizes}
    config = {key: getattr(args, key) for key in ("model", "epochs", "batch_size", "learning_rate", "warmup_ratio",
                                                  "weight_decay", "max_length")}
    config["bf16"] = True

    rows: list[dict] = []
    best_full: tuple[float, Path] | None = None
    for size in sizes:
        for seed in seeds:
            name = f"n{counts[size]}_seed{seed}"
            out, log = work_dir / name, results / "logs" / f"{name}.log"
            log.write_text("", encoding="utf-8")
            started = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] training {name}", flush=True)
            lc.run(
                [sys.executable, "-m", "src.classifier.train", "--train", str(train_file(size)),
                 "--valid", str(DATA / "valid.csv"), "--model", args.model, "--out", str(out),
                 "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                 "--learning-rate", str(args.learning_rate), "--warmup-ratio", str(args.warmup_ratio),
                 "--weight-decay", str(args.weight_decay), "--max-length", str(args.max_length),
                 "--seed", str(seed), "--logging-steps", "50", "--select-best", "--bf16"],
                log,
            )
            for leftover in out.glob("checkpoint-*"):
                shutil.rmtree(leftover)
            summary = json.loads((out / "training_summary.json").read_text(encoding="utf-8"))
            history = epoch_history(summary)
            best = max(history, key=lambda e: e["macro_f1"])
            if abs(best["macro_f1"] - summary["best_metric"]) > 1e-9:
                raise RuntimeError(f"{name}: best epoch in the history does not match the Trainer's best metric")
            steps_per_epoch = history[-1]["step"] / history[-1]["epoch"]
            log_text = log.read_text(encoding="utf-8")
            runtime = ast.literal_eval(re.search(r"\{'train_runtime'[^}]*\}", log_text).group(0))
            print(f"[{time.strftime('%H:%M:%S')}] scoring {name} (best epoch {best['epoch']})", flush=True)
            score_path = results / "scores" / f"{name}.json"
            lc.run([sys.executable, str(HERE.parent / "score.py"), "--checkpoint", str(out), "--out", str(score_path),
                    "--predictions-dir", str(predictions / name)], log)
            row = {
                "size": size,
                "train_sentences": counts[size],
                "seed": seed,
                "best_epoch": best["epoch"],
                "valid_macro_f1": summary["best_metric"],
                "history": history,
                "train_loss": training_loss(log_text, steps_per_epoch),
                "train_runtime_s": runtime["train_runtime"],
                "minutes": (time.time() - started) / 60,
                "scores": json.loads(score_path.read_text(encoding="utf-8")),
            }
            rows.append(row)
            if size == "all" and (best_full is None or row["valid_macro_f1"] > best_full[0]):
                if best_full is not None:
                    shutil.rmtree(best_full[1])
                best_full = (row["valid_macro_f1"], out)
            else:
                shutil.rmtree(out)
            (results / "runs.json").write_text(json.dumps({"config": config, "runs": rows}, indent=2) + "\n",
                                               encoding="utf-8")
            (results / "summary.md").write_text(markdown(rows, config), encoding="utf-8")
            test = row["scores"]["sets"]["test"]["stage2"]
            print(f"  best epoch {best['epoch']}, valid macro-F1 {row['valid_macro_f1']:.4f}; test acc "
                  f"{test['accuracy']:.4f}, macro-F1 {test['macro_f1']:.4f}; {row['minutes']:.1f} min", flush=True)

    if best_full is not None:
        if keep.exists():
            print(f"[kept] {keep} already exists; best full-data model left at {best_full[1]}")
        else:
            shutil.move(str(best_full[1]), str(keep))
            print(f"[kept] best full-data model (valid macro-F1 {best_full[0]:.4f}) -> {keep}")
    print(f"[written] {results / 'summary.md'}, {results / 'runs.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
