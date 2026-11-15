"""Experiment 03: does feeding Stage 1 into Stage 2 help? (ablation ladder, rung 1 vs rung 2)

Three versions of Stage 2, trained with the iteration-1 settings on the same Roman-only COMI-LINGUA splits and
seeds, so the only difference is what the encoder sees for each word:

  none         the surface word only (iteration 1; rung 1 of the ladder)
  tag          + Stage 1's evidence tag (EN, UR, AMB, VAR, BOTH, UNK) added to the input embedding
  tag+reading  + the tag and the Urdu canonical reading (bht -> bohat), behind a learned gate

Stage 1 uses the lexicon induced from the training split (scripts/build_lexicon.py -> data/resources_induced)
for every version, also for scoring "none", so the per-category tables and Stage 3 see the same Stage 1.
Scores: experiments/score.py on the test split, the running examples and the demo slice, Stage 2 alone (tau 0)
and with Stage 3 (tau 0.55). The best model of each version (validation macro-F1) is kept in --work-dir; the
others are deleted after scoring.

    python -X utf8 scripts/build_lexicon.py --train data/cache/comi-lingua-lid/roman/train.csv \
        --out data/resources_induced --check data/cache/comi-lingua-lid/roman/valid.csv
    python -X utf8 experiments/03_stage1_to_stage2/run.py
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
ARMS = ("none", "tag", "tag+reading")
CATEGORIES = ("standard", "ambiguous", "variant", "both", "unknown")


def arm_name(arm: str) -> str:
    return arm.replace("+", "_")


def epoch_history(summary: dict) -> list[dict]:
    """One entry per epoch (the final evaluate() repeats the best epoch's scores at the last step: dropped)."""
    seen: dict[float, dict] = {}
    for entry in summary["eval_history"]:
        seen.setdefault(entry["epoch"], entry)
    return [
        {"epoch": round(e["epoch"]), "loss": e["eval_loss"], "macro_f1": e["eval_macro_f1"],
         "accuracy": e["eval_token_accuracy"]}
        for e in sorted(seen.values(), key=lambda e: e["epoch"])
    ]


def candidate_weights(checkpoint: Path) -> dict | None:
    """How much the trained model uses Stage 1: the reading gate and the size of each tag's vector."""
    from safetensors import safe_open

    from src.classifier.features import CAND_TAGS

    with safe_open(str(checkpoint / "model.safetensors"), framework="pt") as weights:
        names = set(weights.keys())
        if "cand_embeddings.weight" not in names:
            return None
        tags = weights.get_tensor("cand_embeddings.weight").float()
        words = weights.get_tensor("roberta.embeddings.word_embeddings.weight").float()
        gate = float(weights.get_tensor("reading_gate").float()[0]) if "reading_gate" in names else None
    return {
        "reading_gate": gate,
        "tag_vector_norm": {tag: round(float(tags[index].norm()), 4) for index, tag in enumerate(CAND_TAGS)},
        "mean_word_vector_norm": round(float(words.norm(dim=1).mean()), 4),
    }


def stage2(row: dict, split: str = "test") -> dict:
    return row["scores"]["sets"][split]["stage2"]


def correct(row: dict, split: str) -> int:
    scores = stage2(row, split)
    return scores["tokens"] - scores["errors"]


def recall(scores: dict, label: str) -> float | None:
    return scores["per_label"].get(label, {}).get("recall")


def markdown(rows: list[dict], config: dict, lexicon: dict) -> str:
    arms = [arm for arm in ARMS if any(r["arm"] == arm for r in rows)]
    lines = [
        "# Experiment 03: Stage 1 feeds Stage 2",
        "",
        "Test = the Roman-only COMI-LINGUA test split (750 sentences, 15,207 scored words), the same for every "
        "version. Stage 2 alone (tau 0) unless marked. Mean ± standard deviation over seeds.",
        "",
        f"Settings: `{json.dumps(config)}`",
        "",
        f"Stage 1 lexicon (all versions): `data/resources_induced`, {lexicon['final_sizes']['english']} English / "
        f"{lexicon['final_sizes']['roman_urdu']} Roman Urdu words, {lexicon['final_sizes']['ambiguous']} in both; "
        f"it covers {lexicon['check']['covered_after_induction']:.1%} of validation words "
        f"(seeds alone: {lexicon['check']['covered_by_seeds']:.1%}).",
        "",
        "## Overall",
        "",
        "| Version | Runs | Valid macro-F1 | Test acc. | Test macro-F1 | UR recall | EN recall | OTH F1 "
        "| Test acc. + Stage 3 | Running ex. /25 | Demo /51 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in arms:
        group = [r for r in rows if r["arm"] == arm]
        tests = [stage2(r) for r in group]
        lines.append(
            f"| {arm} | {len(group)} | {lc.mean_sd([r['valid_macro_f1'] for r in group])} "
            f"| {lc.mean_sd([t['accuracy'] for t in tests])} | {lc.mean_sd([t['macro_f1'] for t in tests])} "
            f"| {lc.mean_sd([recall(t, 'UR') for t in tests])} | {lc.mean_sd([recall(t, 'EN') for t in tests])} "
            f"| {lc.mean_sd([t['per_label'].get('OTH', {}).get('f1') for t in tests])} "
            f"| {lc.mean_sd([r['scores']['sets']['test']['stage3']['accuracy'] for r in group])} "
            f"| {', '.join(str(correct(r, 'running_examples')) for r in group)} "
            f"| {', '.join(str(correct(r, 'demo')) for r in group)} |"
        )
    lines += ["", "## Paired by seed: version minus `none` (test accuracy, Stage 2 alone)", "",
              "| Version | Seed 1 | Seed 2 | Seed 3 | Mean ± s.d. | Seeds better |", "|---|---|---|---|---|---|"]
    base = {r["seed"]: stage2(r)["accuracy"] for r in rows if r["arm"] == "none"}
    for arm in arms[1:]:
        diffs = {r["seed"]: stage2(r)["accuracy"] - base[r["seed"]] for r in rows if r["arm"] == arm and r["seed"] in base}
        if not diffs:
            continue
        cells = " | ".join(f"{diffs[s]:+.4f}" if s in diffs else "–" for s in (1, 2, 3))
        lines.append(f"| {arm} | {cells} | {lc.mean_sd(list(diffs.values()))} "
                     f"| {sum(d > 0 for d in diffs.values())} of {len(diffs)} |")
    lines += ["", "## By Stage-1 category (test accuracy, Stage 2 alone; categories from the induced lexicon)", "",
              "| Version | " + " | ".join(f"{c} ({stage2(rows[0])['per_category'].get(c, {}).get('tokens', 0)})"
                                         for c in CATEGORIES) + " |",
              "|---|" + "---|" * len(CATEGORIES)]
    for arm in arms:
        group = [r for r in rows if r["arm"] == arm]
        cells = [lc.mean_sd([stage2(r)["per_category"].get(c, {}).get("accuracy") for r in group]) for c in CATEGORIES]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    forms = sorted({(f, g) for r in rows for f, by_gold in stage2(r)["ambiguous_forms"].items() for g in by_gold})
    if forms:
        lines += ["", "## Two-language words on the test split (correct / total, summed over runs)", "",
                  "| Word | Gold | " + " | ".join(arms) + " |", "|---|---|" + "---|" * len(arms)]
        for form, gold in forms:
            cells = []
            for arm in arms:
                hits = total = 0
                for r in rows:
                    if r["arm"] == arm:
                        counts = stage2(r)["ambiguous_forms"].get(form, {}).get(gold)
                        if counts:  # [correct, total]
                            hits, total = hits + counts[0], total + counts[1]
                cells.append(f"{hits}/{total}" if total else "–")
            lines.append(f"| *{form}* | {gold} | " + " | ".join(cells) + " |")
    lines += ["", "## Runs", "", "| Version | Seed | Best epoch | Valid macro-F1 | Test acc. / macro-F1 | Reading gate "
              "| Stage 3 changes (fixed / broke) | Minutes |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        t = stage2(r)
        change = r["scores"]["sets"]["test"]["stage3"].get("stage3_changes", {})
        gate = (r.get("candidate_weights") or {}).get("reading_gate")
        lines.append(
            f"| {r['arm']} | {r['seed']} | {r['best_epoch']} | {r['valid_macro_f1']:.4f} "
            f"| {t['accuracy']:.4f} / {t['macro_f1']:.4f} | {'–' if gate is None else f'{gate:.3f}'} "
            f"| {change.get('changed', 0)} ({change.get('fixed', 0)} / {change.get('broke', 0)}) | {r['minutes']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--resources", default="data/resources_induced", help="Stage-1 lexicon for every version")
    parser.add_argument("--model", default="xlm-roberta-base")
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--cand-dropout", type=float, default=0.15)
    parser.add_argument("--cand-lr", type=float, default=1e-3)
    parser.add_argument("--work-dir", default="checkpoints/experiments/03_stage1_to_stage2")
    parser.add_argument("--predictions", default="data/cache/experiments/03_stage1_to_stage2/predictions")
    args = parser.parse_args(argv)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    resources = ROOT / args.resources
    report = resources / "induction_report.json"
    if not report.is_file():
        raise SystemExit(f"{report} is missing: build the lexicon first (see this script's docstring)")
    lexicon = json.loads(report.read_text(encoding="utf-8"))
    work_dir, predictions = ROOT / args.work_dir, ROOT / args.predictions
    results = HERE / "results"
    for folder in (results / "logs", results / "scores", work_dir):
        folder.mkdir(parents=True, exist_ok=True)
    config = {key: getattr(args, key) for key in ("model", "epochs", "batch_size", "learning_rate", "warmup_ratio",
                                                  "weight_decay", "max_length", "cand_dropout", "cand_lr")}
    config["bf16"] = True
    config["resources"] = args.resources

    rows: list[dict] = []
    best: dict[str, tuple[float, Path]] = {}
    for seed in seeds:  # seed-major, so a partial run already holds comparable versions
        for arm in arms:
            name = f"{arm_name(arm)}_seed{seed}"
            out, log = work_dir / name, results / "logs" / f"{name}.log"
            log.write_text("", encoding="utf-8")
            started = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] training {name}", flush=True)
            lc.run(
                [sys.executable, "-m", "src.classifier.train", "--train", str(DATA / "train.csv"),
                 "--valid", str(DATA / "valid.csv"), "--model", args.model, "--out", str(out),
                 "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                 "--learning-rate", str(args.learning_rate), "--warmup-ratio", str(args.warmup_ratio),
                 "--weight-decay", str(args.weight_decay), "--max-length", str(args.max_length),
                 "--seed", str(seed), "--logging-steps", "50", "--select-best", "--bf16",
                 "--candidates", arm, "--resources", str(resources), "--cand-dropout", str(args.cand_dropout),
                 "--cand-lr", str(args.cand_lr)],
                log,
            )
            for leftover in out.glob("checkpoint-*"):
                shutil.rmtree(leftover)
            summary = json.loads((out / "training_summary.json").read_text(encoding="utf-8"))
            history = epoch_history(summary)
            best_epoch = max(history, key=lambda e: e["macro_f1"])
            if abs(best_epoch["macro_f1"] - summary["best_metric"]) > 1e-9:
                raise RuntimeError(f"{name}: best epoch in the history does not match the Trainer's best metric")
            runtime = ast.literal_eval(re.search(r"\{'train_runtime'[^}]*\}", log.read_text(encoding="utf-8")).group(0))
            print(f"[{time.strftime('%H:%M:%S')}] scoring {name} (best epoch {best_epoch['epoch']})", flush=True)
            score_path = results / "scores" / f"{name}.json"
            lc.run([sys.executable, str(HERE.parent / "score.py"), "--checkpoint", str(out), "--out", str(score_path),
                    "--resources", str(resources), "--predictions-dir", str(predictions / name)], log)
            row = {
                "arm": arm,
                "seed": seed,
                "best_epoch": best_epoch["epoch"],
                "valid_macro_f1": summary["best_metric"],
                "history": history,
                "candidate_weights": candidate_weights(out),
                "train_runtime_s": runtime["train_runtime"],
                "minutes": (time.time() - started) / 60,
                "scores": json.loads(score_path.read_text(encoding="utf-8")),
            }
            rows.append(row)
            if arm not in best or row["valid_macro_f1"] > best[arm][0]:
                if arm in best:
                    shutil.rmtree(best[arm][1])
                best[arm] = (row["valid_macro_f1"], out)
            else:
                shutil.rmtree(out)
            (results / "runs.json").write_text(
                json.dumps({"config": config, "lexicon": {k: lexicon[k] for k in ("final_sizes", "check")},
                            "kept": {a: Path(p).relative_to(ROOT).as_posix() for a, (_, p) in best.items()},
                            "runs": rows}, indent=2) + "\n",
                encoding="utf-8",
            )
            (results / "summary.md").write_text(markdown(rows, config, lexicon), encoding="utf-8")
            test = stage2(row)
            print(f"  best epoch {best_epoch['epoch']}, valid macro-F1 {row['valid_macro_f1']:.4f}; test acc "
                  f"{test['accuracy']:.4f}, macro-F1 {test['macro_f1']:.4f}; {row['minutes']:.1f} min", flush=True)

    for arm, (score, path) in best.items():
        print(f"[kept] {arm}: {path} (valid macro-F1 {score:.4f})")
    print(f"[written] {results / 'summary.md'}, {results / 'runs.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
