"""Select the Stage 3 threshold tau on validation data and save it with the checkpoint.

Stage 2 runs once per sentence. Stage 3 is then re-applied at every candidate tau: 0, the confidence of every
validation word some rule would re-label (the decisions only change there; each one is tried at the confidence
itself and just above it) and a 0.01 grid. The selected tau keeps the best validation macro-F1 (the metric that
selects the training epoch), then accuracy, and is the largest tau before that score first drops: Stage 3
re-checks every word it can without costing anything on validation. A fine grid around the selected tau
(--zoom-step, default 0.0001) is scored too and written to zoom.csv, as a check and for plots. The test split
and the hand-labelled demo sets are scored afterwards, for reporting only: they never affect the choice.

Writes <checkpoint>/stage3.json for a local checkpoint folder (the pipeline and scripts/evaluate.py use it when
no --tau is given) and the sweep to --out-dir: summary.md, sweep.json, sweep.csv (every candidate) and zoom.csv; counts
and seed-lexicon word forms only, no sentences. With --calibration it also fits temperature scaling on validation
and reports the calibration error before and after, and the sweep on the rescaled confidences.

    python -X utf8 scripts/tune_tau.py --checkpoint checkpoints/ruen-lid \\
        --valid data/cache/comi-lingua-lid/roman/valid.csv --test data/cache/comi-lingua-lid/roman/test.csv \\
        --calibration
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.candidate_expansion.expander import CandidateExpander  # noqa: E402
from src.classifier.dataset import IGNORE_LABEL, read_any  # noqa: E402
from src.eval.calibration import expected_calibration_error, fit_temperature, reliability, softmax  # noqa: E402
from src.pipeline import DEFAULT_RESOURCES  # noqa: E402
from src.verification.rules import DEFAULT_TAU, apply_rules  # noqa: E402
from src.verification.tau import (  # noqa: E402
    ScoredSentence,
    SweepPoint,
    candidate_taus,
    change_points,
    config_path,
    evaluate_tau,
    save_tuned_tau,
    select_tau,
    sweep,
)

RULES = ("variant-consistency", "english-lexicon-prior", "urdu-lexicon-prior")
REPORT_SETS = ("data/demo/seed_labels.csv", "data/demo/running_examples.csv")
BANDS = (0.0, 0.5, 0.55, 0.75, 0.9, 0.99, 1.0)  # reliability bands; the last one is closed
MAX_LISTED_CHANGE_POINTS = 40  # summary.md lists every change point up to this many, else the grid only
SELECTION = (
    "validation macro-F1, then accuracy; among the tau values that tie on them, the largest one before the score "
    "first drops (the rules re-check every word they can at no cost on validation)"
)


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def relative(path: Path) -> str:
    return os.path.relpath(path, ROOT).replace(os.sep, "/") if path.is_relative_to(ROOT) else str(path)


def word_logits(predictor, words: list[str]) -> list[list[float] | None]:
    """First-piece logits of every word, computed exactly as TokenLIDPredictor.predict_words does."""
    encoding = predictor.tokenizer(
        words, is_split_into_words=True, truncation=True, max_length=512, return_tensors="pt"
    )
    inputs = {key: value.to(predictor.device) for key, value in encoding.items()}
    with predictor.torch.no_grad():
        logits = predictor.model(**inputs).logits[0].to("cpu")
    first: dict[int, list[float]] = {}
    for position, word_id in enumerate(encoding.word_ids(0)):
        if word_id is not None and word_id not in first:
            first[word_id] = logits[position].tolist()
    return [first.get(index) for index in range(len(words))]


def run_stage2(predictor, expander: CandidateExpander, path: Path, with_logits: bool):
    """Stages 1 and 2 once per sentence, as in LIDPipeline.run(); optionally the logits for calibration."""
    scored: list[ScoredSentence] = []
    logits: list[list[list[float] | None]] = []
    for sentence in read_any(str(path)):
        tokens = expander.expand(" ".join(sentence.words))
        if len(tokens) != len(sentence.words):
            raise ValueError(f"{path.name} {sentence.sentence_id}: {len(sentence.words)} words, {len(tokens)} tokens")
        words = [token.surface for token in tokens]
        predictions = tuple((p.label, p.confidence) for p in predictor.predict_words(words))
        scored.append(ScoredSentence(tuple(tokens), predictions, tuple(sentence.labels)))
        logits.append(word_logits(predictor, words) if with_logits else [])
    return scored, logits


def scored_words(sentences: list[ScoredSentence]) -> tuple[list[float], list[bool]]:
    confidences: list[float] = []
    correct: list[bool] = []
    for sentence in sentences:
        for (label, confidence), gold in zip(sentence.predictions, sentence.gold):
            if gold != IGNORE_LABEL:
                confidences.append(confidence)
                correct.append(label == gold)
    return confidences, correct


def calibration_report(sentences: list[ScoredSentence]) -> dict:
    confidences, correct = scored_words(sentences)
    return {
        "words": len(confidences),
        "mean_confidence": sum(confidences) / len(confidences),
        "accuracy": sum(correct) / len(correct),
        "ece": expected_calibration_error(confidences, correct),
        "bands": reliability(confidences, correct, BANDS),
    }


def check_logits(sentences: list[ScoredSentence], logits: list, id2label: dict) -> None:
    """The logits must reproduce predict_words: same label, same confidence."""
    for sentence, rows in zip(sentences, logits):
        for (label, confidence), row in zip(sentence.predictions, rows):
            if row is None:
                continue
            probabilities = softmax(row)
            top = max(range(len(row)), key=row.__getitem__)
            if id2label[top] != label or abs(probabilities[top] - confidence) > 1e-4:
                raise RuntimeError(f"logits disagree with predict_words: {label} {confidence} vs {probabilities}")


def rescaled(sentences: list[ScoredSentence], logits: list, temperature: float) -> list[ScoredSentence]:
    """The same sentences with each confidence temperature-scaled (labels cannot change)."""
    result = []
    for sentence, rows in zip(sentences, logits):
        predictions = tuple(
            (label, confidence if row is None else max(softmax(row, temperature)))
            for (label, confidence), row in zip(sentence.predictions, rows)
        )
        result.append(ScoredSentence(sentence.tokens, predictions, sentence.gold))
    return result


def temperature_rows(sentences: list[ScoredSentence], logits: list, label2id: dict) -> tuple[list, list]:
    rows, gold = [], []
    for sentence, sentence_rows in zip(sentences, logits):
        for label, row in zip(sentence.gold, sentence_rows):
            if row is not None and label in label2id:
                rows.append(row)
                gold.append(label2id[label])
    return rows, gold


def sign_test(fixed: int, broke: int) -> float:
    """Two-sided exact sign test of fixed vs broke words (H0: a rule change is a coin flip)."""
    n = fixed + broke
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(fixed, broke) + 1)) / 2**n
    return min(1.0, 2 * tail)


def firing_forms(sentences: list[ScoredSentence], tau: float) -> list[dict]:
    """Per rule and word form, what the rule does at tau. Rules only fire on seed-lexicon / variant-pair forms."""
    table: dict[tuple[str, str], dict] = {}
    for sentence in sentences:
        decisions = apply_rules(list(sentence.tokens), list(sentence.predictions), tau)
        for token, (stage2, confidence), decision, gold in zip(
            sentence.tokens, sentence.predictions, decisions, sentence.gold
        ):
            if not decision.rules or gold == IGNORE_LABEL:
                continue
            rule = "+".join(decision.rules)
            entry = table.setdefault((rule, token.normalized), {
                "rule": rule,
                "form": token.normalized,
                "category": token.category,
                "change": f"{stage2}->{decision.label}",
                "fired": 0,
                "fixed": 0,
                "broke": 0,
                "gold": {},
                "confidences": [],
            })
            entry["fired"] += 1
            entry["fixed"] += decision.label == gold and stage2 != gold
            entry["broke"] += stage2 == gold and decision.label != gold
            entry["gold"][gold] = entry["gold"].get(gold, 0) + 1
            entry["confidences"].append(round(confidence, 4))
    for entry in table.values():
        entry["confidences"].sort()
    return sorted(table.values(), key=lambda entry: (entry["rule"], -entry["fired"], entry["form"]))


def per_rule(valid: list[ScoredSentence], test: list[ScoredSentence] | None) -> list[dict]:
    """Each rule on its own, with its own tau selected on validation (an assessment, not used by the pipeline)."""
    rows = []
    test_alone = evaluate_tau(test, 0.0) if test is not None else None
    for rule in RULES:
        points = sweep(valid, candidate_taus(valid, grid=None, rules={rule}), rules={rule})
        best = select_tau(points)
        row = {
            "rule": rule,
            "valid_at_tau_1": evaluate_tau(valid, 1.0, rules={rule}).as_dict(),
            "valid_best": best.as_dict(),
            "valid_best_gain_macro_f1": best.macro_f1 - points[0].macro_f1,
            "valid_best_sign_test_p": sign_test(best.fixed, best.broke),
        }
        if test is not None:
            at_best = evaluate_tau(test, best.tau, rules={rule})
            row["test_at_tau_1"] = evaluate_tau(test, 1.0, rules={rule}).as_dict()
            row["test_at_valid_best"] = at_best.as_dict()
            row["test_at_valid_best_gain_macro_f1"] = at_best.macro_f1 - test_alone.macro_f1
        rows.append(row)
    return rows


def calibration(valid, valid_logits, test, test_logits, id2label: dict) -> dict:
    check_logits(valid, valid_logits, id2label)
    label2id = {label: index for index, label in id2label.items()}
    temperature = fit_temperature(*temperature_rows(valid, valid_logits, label2id))
    valid_scaled = rescaled(valid, valid_logits, temperature)
    points = sweep(valid_scaled, candidate_taus(valid_scaled))
    best = select_tau(points)
    result = {
        "temperature": temperature,
        "valid_scaled": calibration_report(valid_scaled),
        "valid_scaled_below": {str(tau): evaluate_tau(valid_scaled, tau).below_tau for tau in (DEFAULT_TAU, 0.75)},
        "valid_scaled_selected": best.as_dict(),
        "valid_scaled_stage2_alone": points[0].as_dict(),
    }
    if test is not None:
        check_logits(test, test_logits, id2label)
        test_scaled = rescaled(test, test_logits, temperature)
        result["test_scaled"] = calibration_report(test_scaled)
        result["test_scaled_at_selected"] = evaluate_tau(test_scaled, best.tau).as_dict()
    return result


def fmt_tau(tau: float, edges: frozenset[float] = frozenset()) -> str:
    """A tau just above a change point c prints as '>c', so the printed number is never mistaken for c."""
    if tau == round(tau, 4):
        return f"{tau:g}"
    below = math.nextafter(tau, -math.inf)
    return f">{below:.6f}" if below in edges and tau not in edges else f"{tau:.6f}"


def zoom_taus(chosen: float, step: float, width: float) -> list[float]:
    """A fine grid of the given step over [chosen - width, chosen + width] within [0, 1], plus chosen and the
    smallest tau above it."""
    count = round(width / step)
    taus = {round(chosen + offset * step, 10) for offset in range(-count, count + 1)}
    taus |= {chosen, math.nextafter(chosen, math.inf)}
    return sorted(tau for tau in taus if 0.0 <= tau <= 1.0)


def zoom_check(points: list[SweepPoint], chosen: SweepPoint) -> dict:
    """On the fine grid: every tau up to the chosen one keeps its score, and where the score first drops."""
    target = (chosen.macro_f1, chosen.accuracy)
    below = [point for point in points if point.tau <= chosen.tau]
    above = [point for point in points if point.tau > chosen.tau and (point.macro_f1, point.accuracy) != target]
    return {
        "taus": len(points),
        "from": points[0].tau,
        "to": points[-1].tau,
        "same_score_up_to_selected": all((point.macro_f1, point.accuracy) == target for point in below),
        "first_change_above": above[0].as_dict() if above else None,
    }


def shown_taus(change: list[float], chosen: float) -> set[float]:
    """The sweep rows that go into summary.md and sweep.json (sweep.csv has every candidate)."""
    shown = {0.0, DEFAULT_TAU, 1.0, chosen, math.nextafter(chosen, math.inf)}
    shown |= {round(step * 0.05, 10) for step in range(1, 21)}
    if len(change) <= MAX_LISTED_CHANGE_POINTS:
        shown |= set(change)
    return shown


def write_csv(path: Path, sets: dict[str, list[SweepPoint]]) -> None:
    columns = ["set", "tau", "macro_f1", "accuracy", "below_tau", "fired", "fixed", "broke"]
    columns += [f"{rule}_{count}" for rule in RULES for count in ("fired", "fixed", "broke")]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for name, points in sets.items():
            for point in points:
                by_rule = [count for rule in RULES for count in point.by_rule.get(rule, (0, 0, 0))]
                writer.writerow([
                    name, repr(point.tau), f"{point.macro_f1:.6f}", f"{point.accuracy:.6f}", point.below_tau,
                    point.fired, point.fixed, point.broke, *by_rule,
                ])


def markdown(result: dict) -> str:
    chosen = result["selected"]
    edges = frozenset(result["candidates"]["change_point_values"])
    edges |= frozenset(result.get("test", {}).get("change_point_values", []))
    lines = [
        "# Stage 3 threshold τ: selected on validation",
        "",
        f"Generated by `scripts/tune_tau.py` on {result['date']} · checkpoint `{result['checkpoint']}` · "
        f"device {result['device']}",
        "",
        f"- Validation: `{result['valid']['file']}` ({result['valid']['sentences']} sentences, "
        f"{chosen['tokens']} scored words). Selection: {result['selection']}.",
        f"- Candidates: {result['candidates']['count']} (τ = 0, {result['candidates']['change_points']} change "
        f"points, each at its confidence and just above it, a {result['candidates']['grid']} grid).",
    ]
    if result.get("test"):
        lines.append(f"- Test: `{result['test']['file']}`, **reporting only** (never used for the choice).")
    above = result["valid"]["just_above_selected"]
    verdict = (
        f"{chosen['fixed']} fixed and {chosen['broke']} broken words on validation, {chosen['below_tau']} words "
        f"re-checked (confidence < τ). Just above it, at {fmt_tau(above['tau'], edges)}: {above['fixed']} fixed, "
        f"{above['broke']} broke, macro-F1 {above['macro_f1']:.4f}, accuracy {above['accuracy']:.4f}"
    )
    lines += [
        "",
        f"**Selected τ = {fmt_tau(chosen['tau'], edges)}**: {verdict}.",
        "",
        "## Sweep",
        "",
        f"A selection of rows; `sweep.csv` has all {result['candidates']['count']} candidates. Re-checked = words "
        "with confidence < τ; fired = words a rule re-labelled (fixed: wrong → right; broke: right → wrong).",
        "",
    ]
    lines.append(
        "| τ | Valid macro-F1 | Valid accuracy | Re-checked (< τ) | Fired | Fixed | Broke |"
        + (" Test macro-F1 | Test accuracy | Test fired | Test fixed | Test broke |" if result.get("test") else "")
    )
    lines.append("|---" * (12 if result.get("test") else 7) + "|")
    test_points = {point["tau"]: point for point in result.get("test", {}).get("sweep", [])}
    for point in result["valid"]["sweep"]:
        mark =" ◀ selected" if point["tau"] == chosen["tau"] else ""
        row = (
            f"| {fmt_tau(point['tau'], edges)}{mark} | {point['macro_f1']:.4f} | {point['accuracy']:.4f} | "
            f"{point['below_tau']} | {point['fired']} | {point['fixed']} | {point['broke']} |"
        )
        if test_points:
            test = test_points[point["tau"]]
            row += (
                f" {test['macro_f1']:.4f} | {test['accuracy']:.4f} | {test['fired']} | {test['fixed']} |"
                f" {test['broke']} |"
            )
        lines.append(row)
    if result.get("test"):
        hindsight = result["test"]["hindsight_best"]
        lines += [
            "",
            f"Hindsight only: the τ that would be best on test is {fmt_tau(hindsight['tau'], edges)} "
            f"(test macro-F1 {hindsight['macro_f1']:.4f}, {hindsight['fixed']} fixed / {hindsight['broke']} broke).",
        ]

    zoom = result["zoom"]
    lines += [
        "",
        "## Zoom: a fine grid around the selected τ",
        "",
        f"{zoom['valid']['taus']} values of τ from {zoom['valid']['from']:.4f} to {zoom['valid']['to']:.4f} in steps "
        f"of {zoom['step']:g} (`zoom.csv`). Validation keeps the selected score at every one of them up to the "
        f"selected τ: **{'yes' if zoom['valid']['same_score_up_to_selected'] else 'NO'}**.",
    ]
    for name in ("valid", "test"):
        if name in zoom and zoom[name]["first_change_above"]:
            first = zoom[name]["first_change_above"]
            lines.append(
                f"- {name}: the first grid τ above the selected one that changes the score is "
                f"{fmt_tau(first['tau'], edges)} (macro-F1 {first['macro_f1']:.4f}, accuracy {first['accuracy']:.4f}, "
                f"{first['fixed']} fixed / {first['broke']} broke)."
            )
    lines += [
        "",
        "## Each rule on its own (assessment)",
        "",
        "τ = 1 shows every word the rule could re-label. Own best τ: the rule alone, selected on validation the "
        "same way (not used by the pipeline). Fired / fixed / broke counts.",
        "",
    ]
    lines.append(
        "| Rule | Valid at τ = 1 | Own best τ (valid) | Valid gain (macro-F1) | Sign-test p |"
        + (" Test at τ = 1 | Test at own best τ | Test gain |" if result.get("test") else "")
    )
    lines.append("|---" * (8 if result.get("test") else 5) + "|")
    for row in result["per_rule"]:
        everything, best = row["valid_at_tau_1"], row["valid_best"]
        line = (
            f"| {row['rule']} | {everything['fired']} / {everything['fixed']} / {everything['broke']} | "
            f"{fmt_tau(best['tau'], edges)} | {row['valid_best_gain_macro_f1']:+.4f} | {row['valid_best_sign_test_p']:.2f} |"
        )
        if "test_at_valid_best" in row:
            upper, test = row["test_at_tau_1"], row["test_at_valid_best"]
            gain = row["test_at_valid_best_gain_macro_f1"]
            line += (
                f" {upper['fired']} / {upper['fixed']} / {upper['broke']} |"
                f" {test['fired']} / {test['fixed']} / {test['broke']} | {gain:+.4f} |"
            )
        lines.append(line)

    lines += ["", "## Word forms the rules would re-label at τ = 1", ""]
    lines.append("| Set | Rule | Form | Change | Fired | Fixed | Broke | Gold | Confidences |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name in ("valid", "test"):
        for entry in result.get(name, {}).get("forms_at_tau_1", []):
            gold = ", ".join(f"{label} {count}" for label, count in sorted(entry["gold"].items()))
            confidences = ", ".join(f"{value:.2f}" for value in entry["confidences"][:8])
            confidences += " …" if len(entry["confidences"]) > 8 else ""
            lines.append(
                f"| {name} | {entry['rule']} | *{entry['form']}* | {entry['change']} | {entry['fired']} | "
                f"{entry['fixed']} | {entry['broke']} | {gold} | {confidences} |"
            )

    lines += ["", "## Hand-labelled sets (reporting only)", ""]
    lines.append("| Set | Words | τ = 0 accuracy · macro-F1 | Selected τ | τ = 0.55 (default) |")
    lines.append("|---|---|---|---|---|")
    for name, report in result["reports"].items():
        cells = [
            f"{report[key]['accuracy']:.4f} · {report[key]['macro_f1']:.4f} ({report[key]['fired']} fired)"
            for key in ("tau_0", "selected", "default")
        ]
        lines.append(f"| `{name}` | {report['tau_0']['tokens']} | " + " | ".join(cells) + " |")

    lines += ["", "## Calibration of the Stage 2 confidence", ""]
    sets = [("Validation", result["valid"]["calibration"])]
    if result.get("test"):
        sets.append(("Test", result["test"]["calibration"]))
    lines.append("| | " + " | ".join(name for name, _ in sets) + " |")
    lines.append("|---" * (len(sets) + 1) + "|")
    lines.append("| Mean confidence | " + " | ".join(f"{c['mean_confidence']:.4f}" for _, c in sets) + " |")
    lines.append("| Accuracy | " + " | ".join(f"{c['accuracy']:.4f}" for _, c in sets) + " |")
    lines.append("| ECE (15 bins) | " + " | ".join(f"{c['ece']:.4f}" for _, c in sets) + " |")
    lines += ["", "| Confidence band | " + " | ".join(f"{name}: words · accuracy" for name, _ in sets) + " |"]
    lines.append("|---" * (len(sets) + 1) + "|")
    for index, band in enumerate(sets[0][1]["bands"]):
        low, high = band["band"]
        bracket = "]" if index == len(sets[0][1]["bands"]) - 1 else ")"
        cells = []
        for _, report in sets:
            cell = report["bands"][index]
            accuracy = f"{cell['accuracy']:.3f}" if cell["accuracy"] is not None else "–"
            cells.append(f"{cell['words']} · {accuracy}")
        lines.append(f"| [{low:g}, {high:g}{bracket} | " + " | ".join(cells) + " |")
    scaled = result.get("calibration")
    if scaled:
        lines += [
            "",
            f"Temperature scaling (fitted on validation): **T = {scaled['temperature']:.3f}**. ECE after scaling: "
            f"validation {scaled['valid_scaled']['ece']:.4f}"
            + (f", test {scaled['test_scaled']['ece']:.4f}" if "test_scaled" in scaled else "")
            + f". Validation words below τ = 0.55 after scaling: {scaled['valid_scaled_below'][str(DEFAULT_TAU)]}. "
            f"Best τ on the rescaled validation confidences: {fmt_tau(scaled['valid_scaled_selected']['tau'])} "
            f"(macro-F1 {scaled['valid_scaled_selected']['macro_f1']:.4f}, "
            f"{scaled['valid_scaled_selected']['fixed']} fixed / {scaled['valid_scaled_selected']['broke']} broke; "
            f"Stage 2 alone {scaled['valid_scaled_stage2_alone']['macro_f1']:.4f}).",
        ]
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="local checkpoint folder or Hugging Face id")
    parser.add_argument("--valid", default="data/cache/comi-lingua-lid/roman/valid.csv", help="selects tau")
    parser.add_argument("--test", help="scored at the same tau values, reporting only")
    parser.add_argument("--report", nargs="*", default=list(REPORT_SETS), help="more sets, reporting only")
    parser.add_argument("--grid", type=float, default=0.01, help="grid step added to the change points (0: none)")
    parser.add_argument("--zoom-step", type=float, default=0.0001, help="step of the fine grid around the chosen tau")
    parser.add_argument("--zoom-width", type=float, default=0.05, help="the fine grid spans chosen tau +/- this")
    parser.add_argument("--calibration", action="store_true", help="also fit and report temperature scaling")
    parser.add_argument("--out-dir", default="results/stage3_tau", help="summary.md, sweep.json and sweep.csv go here")
    parser.add_argument("--no-save", action="store_true", help="do not write <checkpoint>/stage3.json")
    parser.add_argument("--resources", default=DEFAULT_RESOURCES)
    parser.add_argument("--device", default="auto", help="cuda, cpu or auto")
    args = parser.parse_args(argv)

    import torch

    from src.classifier.predict import TokenLIDPredictor

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    expander = CandidateExpander.from_resources(args.resources)
    predictor = TokenLIDPredictor(args.checkpoint, device=device)
    id2label = {int(index): label for index, label in predictor.id2label.items()}

    valid_path = resolve(args.valid)
    valid, valid_logits = run_stage2(predictor, expander, valid_path, args.calibration)
    taus = candidate_taus(valid, grid=args.grid or None)
    points = sweep(valid, taus)
    chosen = select_tau(points)
    change = [point for point in change_points(valid) if point < 1.0]
    just_above = evaluate_tau(valid, math.nextafter(chosen.tau, math.inf))
    print(
        f"validation: {len(taus)} candidate taus ({len(change)} change points); selected tau = {chosen.tau} "
        f"(macro-F1 {chosen.macro_f1:.4f} vs {points[0].macro_f1:.4f} at tau 0; "
        f"{chosen.fixed} fixed, {chosen.broke} broke; just above it: {just_above.fixed} fixed, "
        f"{just_above.broke} broke)",
        flush=True,
    )

    shown = shown_taus(change, chosen.tau)
    sweeps = {"valid": points}
    fine = zoom_taus(chosen.tau, args.zoom_step, args.zoom_width)
    zooms = {"valid": sweep(valid, fine)}
    result: dict = {
        "checkpoint": str(args.checkpoint),
        "device": device,
        "date": datetime.date.today().isoformat(),
        "selection": SELECTION,
        "candidates": {
            "count": len(taus), "change_points": len(change), "grid": args.grid, "change_point_values": change,
        },
        "selected": chosen.as_dict(),
        "valid": {
            "file": relative(valid_path),
            "sentences": len(valid),
            "sweep": [point.as_dict() for point in points if point.tau in shown],
            "just_above_selected": just_above.as_dict(),
            "forms_at_tau_1": firing_forms(valid, 1.0),
            "calibration": calibration_report(valid),
        },
    }
    test = test_logits = None
    if args.test:
        test_path = resolve(args.test)
        test, test_logits = run_stage2(predictor, expander, test_path, args.calibration)
        sweeps["test"] = sweep(test, taus)
        zooms["test"] = sweep(test, fine)
        result["test"] = {
            "file": relative(test_path),
            "sentences": len(test),
            "note": "reporting only: never used to select tau",
            "at_selected": evaluate_tau(test, chosen.tau).as_dict(),
            "change_point_values": [point for point in change_points(test) if point < 1.0],
            "sweep": [point.as_dict() for point in sweeps["test"] if point.tau in shown],
            "hindsight_best": select_tau(sweep(test, candidate_taus(test, grid=args.grid or None))).as_dict(),
            "forms_at_tau_1": firing_forms(test, 1.0),
            "calibration": calibration_report(test),
        }
    result["zoom"] = {"step": args.zoom_step, "width": args.zoom_width}
    for name, zoom_points in zooms.items():
        reference = chosen if name == "valid" else evaluate_tau(test, chosen.tau)
        result["zoom"][name] = zoom_check(zoom_points, reference)
    result["per_rule"] = per_rule(valid, test)
    result["reports"] = {}
    for report in args.report:
        path = resolve(report)
        sentences, _ = run_stage2(predictor, expander, path, False)
        result["reports"][relative(path)] = {
            "tau_0": evaluate_tau(sentences, 0.0).as_dict(),
            "selected": evaluate_tau(sentences, chosen.tau).as_dict(),
            "default": evaluate_tau(sentences, DEFAULT_TAU).as_dict(),
        }
    if args.calibration:
        result["calibration"] = calibration(valid, valid_logits, test, test_logits, id2label)

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sweep.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "summary.md").write_text(markdown(result), encoding="utf-8")
    write_csv(out_dir / "sweep.csv", sweeps)
    write_csv(out_dir / "zoom.csv", zooms)
    for name in ("summary.md", "sweep.json", "sweep.csv", "zoom.csv"):
        print(f"[written] {out_dir / name}")

    target = config_path(args.checkpoint)
    if args.no_save:
        print("[not saved] --no-save")
    elif target is None:
        print(f"[not saved] {args.checkpoint} is not a local folder; pass --tau {chosen.tau} to use this value")
    else:
        selected, alone = chosen.as_dict(), points[0].as_dict()
        keys = ("macro_f1", "accuracy", "fired", "fixed", "broke")
        save_tuned_tau(target, {
            "tau": chosen.tau,
            "selected_by": "scripts/tune_tau.py",
            "selection": result["selection"],
            "validation": result["valid"]["file"],
            "validation_scored_words": chosen.tokens,
            "validation_at_selected": {key: selected[key] for key in keys},
            "validation_stage2_alone": {key: alone[key] for key in keys[:2]},
            "validation_just_above": {key: just_above.as_dict()[key] for key in ("tau", *keys)},
            "candidates": len(taus),
            "default_tau": DEFAULT_TAU,
            "date": result["date"],
        })
        print(f"[written] {target} (used by the pipeline and scripts/evaluate.py when --tau is not given)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
