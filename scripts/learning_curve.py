"""Stage-2 learning curve: fine-tune on nested train subsets with several seeds and score every run.

Each run trains with per-epoch validation (best epoch by validation macro-F1), then scores the test file
through the pipeline twice: Stage 2 alone (tau 0, so no rule can fire) and with Stage 3 (default tau).
Models are deleted after scoring, except the best full-data run, which is moved to --keep.
Subset files come from scripts/prepare_comi_lingua.py --subsets.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classifier.dataset import read_any  # noqa: E402
from src.verification.rules import DEFAULT_TAU  # noqa: E402

CHILD_ENV = {**os.environ, "USE_TF": "0", "TF_CPP_MIN_LOG_LEVEL": "3", "PYTHONIOENCODING": "utf-8"}


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def run(command: list[str], log: Path) -> str:
    """Run a command from the repo root, append its output to log, return stdout."""
    result = subprocess.run(
        command, cwd=ROOT, env=CHILD_ENV, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(command)}\n{result.stdout}\n{result.stderr}\n")
    if result.returncode != 0:
        raise RuntimeError(f"command failed (exit {result.returncode}), see {log}:\n{result.stderr[-2000:]}")
    return result.stdout


def score(checkpoint: Path, test: Path, tau: float, log: Path) -> dict:
    stdout = run(
        [sys.executable, "scripts/evaluate.py", "--data", str(test), "--checkpoint", str(checkpoint),
         "--tau", str(tau), "--json"],
        log,
    )
    report = json.loads(stdout[stdout.index("{"):])
    categories = report["per_category"]
    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro_f1"],
        "ambiguous_accuracy": categories.get("ambiguous", {}).get("accuracy"),
        "both_accuracy": categories.get("both", {}).get("accuracy"),
        "tokens": report["tokens"],
        "ignored": report["ignored"],
    }


def best_epoch(summary: dict) -> float | None:
    for entry in summary["eval_history"]:
        if entry.get("eval_macro_f1") == summary["best_metric"]:
            return entry["epoch"]
    return None


def mean_sd(values: list[float]) -> str:
    values = [value for value in values if value is not None]
    if not values:
        return "–"
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.mean(values):.4f} ± {sd:.4f}"


def markdown(rows: list[dict], config: dict) -> str:
    lines = [
        "# Stage 2 learning curve",
        "",
        "Proxy data: Latin-script COMI-LINGUA (Hindi–English, hi → UR). Mean ± standard deviation over seeds.",
        "Stage 2 alone = pipeline with tau 0 (no Stage 3 rule can fire); with Stage 3 = default tau.",
        "",
        f"Settings: `{json.dumps(config)}`",
        "",
        "| Train sentences | Runs | Valid macro-F1 | Test acc. (Stage 2) | Test macro-F1 (Stage 2) "
        "| Test acc. (+ Stage 3) | Test macro-F1 (+ Stage 3) | Ambiguous acc. (+ Stage 3) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for size in dict.fromkeys(row["size"] for row in rows):
        group = [row for row in rows if row["size"] == size]
        lines.append(
            f"| {group[0]['train_sentences']} | {len(group)} | {mean_sd([r['valid_macro_f1'] for r in group])} "
            f"| {mean_sd([r['stage2']['accuracy'] for r in group])} | {mean_sd([r['stage2']['macro_f1'] for r in group])} "
            f"| {mean_sd([r['stage3']['accuracy'] for r in group])} | {mean_sd([r['stage3']['macro_f1'] for r in group])} "
            f"| {mean_sd([r['stage3']['ambiguous_accuracy'] for r in group])} |"
        )
    lines += [
        "",
        "## Runs",
        "",
        "| Train sentences | Seed | Best epoch | Valid macro-F1 | Test acc. / macro-F1 (Stage 2) "
        "| Test acc. / macro-F1 (+ Stage 3) | Minutes |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['train_sentences']} | {row['seed']} | {row['best_epoch']} | {row['valid_macro_f1']:.4f} "
            f"| {row['stage2']['accuracy']:.4f} / {row['stage2']['macro_f1']:.4f} "
            f"| {row['stage3']['accuracy']:.4f} / {row['stage3']['macro_f1']:.4f} | {row['minutes']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/cache/comi-lingua-lid/roman")
    parser.add_argument("--sizes", default="1000,2000,5000,all", help="subset sizes; 'all' = train.csv")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--model", default="xlm-roberta-base")
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--work-dir", default="checkpoints/learning_curve")
    parser.add_argument("--results", default="results/learning_curve")
    parser.add_argument("--keep", default="checkpoints/ruen-lid", help="where the best full-data model is moved")
    args = parser.parse_args(argv)

    data_dir, work_dir, results = resolve(args.data_dir), resolve(args.work_dir), resolve(args.results)
    keep = resolve(args.keep)
    (results / "logs").mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    sizes = [size.strip() for size in args.sizes.split(",") if size.strip()]
    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]
    train_files = {size: data_dir / ("train.csv" if size == "all" else f"train_{size}.csv") for size in sizes}
    missing = [str(path) for path in train_files.values() if not path.exists()]
    if missing:
        parser.error(f"missing train files (run prepare_comi_lingua.py --subsets ...): {missing}")
    counts = {size: len(read_any(str(path))) for size, path in train_files.items()}
    config = {key: getattr(args, key) for key in ("model", "epochs", "batch_size", "learning_rate", "warmup_ratio",
                                                  "weight_decay", "max_length", "bf16")}

    rows: list[dict] = []
    best_full: tuple[float, Path] | None = None
    for seed in seeds:
        for size in sizes:
            name = f"n{counts[size]}_seed{seed}"
            out = work_dir / name
            log = results / "logs" / f"{name}.log"
            log.write_text("", encoding="utf-8")
            started = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] training {name}", flush=True)
            run(
                [sys.executable, "-m", "src.classifier.train", "--train", str(train_files[size]),
                 "--valid", str(data_dir / "valid.csv"), "--model", args.model, "--out", str(out),
                 "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                 "--learning-rate", str(args.learning_rate), "--warmup-ratio", str(args.warmup_ratio),
                 "--weight-decay", str(args.weight_decay), "--max-length", str(args.max_length),
                 "--seed", str(seed), "--logging-steps", "50", "--select-best"]
                + (["--bf16"] if args.bf16 else []),
                log,
            )
            for leftover in out.glob("checkpoint-*"):
                shutil.rmtree(leftover)
            summary = json.loads((out / "training_summary.json").read_text(encoding="utf-8"))
            print(f"[{time.strftime('%H:%M:%S')}] scoring {name}", flush=True)
            row = {
                "size": size,
                "train_sentences": counts[size],
                "seed": seed,
                "best_epoch": best_epoch(summary),
                "valid_macro_f1": summary["final_eval"]["eval_macro_f1"],
                "valid_accuracy": summary["final_eval"]["eval_token_accuracy"],
                "stage2": score(out, data_dir / "test.csv", 0.0, log),
                "stage3": score(out, data_dir / "test.csv", DEFAULT_TAU, log),
                "minutes": (time.time() - started) / 60,
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
            print(f"  valid macro-F1 {row['valid_macro_f1']:.4f}; test acc {row['stage2']['accuracy']:.4f} "
                  f"(Stage 2) / {row['stage3']['accuracy']:.4f} (+ Stage 3); {row['minutes']:.1f} min", flush=True)

    if best_full is not None:
        if keep.exists():
            print(f"[kept] {keep} already exists; best full-data model left at {best_full[1]}")
        else:
            keep.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(best_full[1]), str(keep))
            print(f"[kept] best full-data model (valid macro-F1 {best_full[0]:.4f}) -> {keep}")
    print(f"[written] {results / 'summary.md'}, {results / 'runs.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
