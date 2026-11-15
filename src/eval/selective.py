"""Measure the confidence gate and Stage 3, not just the final accuracy.

The pipeline makes a selective decision: Stage 2 keeps words it is sure about
(confidence >= tau) and hands the rest to Stage 3. Overall accuracy hides
whether that split works. This report answers three separate questions:

  1. Is the gate useful?    accuracy on accepted words should be much higher
                            than on deferred words (selective classification,
                            Geifman & El-Yaniv, NeurIPS 2017: risk vs coverage).
  2. Does Stage 3 help?     fixes (Stage 2 wrong -> final right) against breaks
                            (Stage 2 right -> final wrong), per rule.
  3. Where are the errors?  per Stage-1 category (ambiguous, variant, ...).
"""

from __future__ import annotations

from collections import Counter, defaultdict

from ..classifier.dataset import IGNORE_LABEL


def collect_rows(pipeline, sentences) -> list[dict]:
    """Run the pipeline on gold sentences; one row per scored word."""
    rows: list[dict] = []
    for sentence in sentences:
        pairs = pipeline.run(" ".join(sentence.words))
        if len(pairs) != len(sentence.words):
            raise ValueError(
                f"tokenization mismatch for {sentence.sentence_id}: "
                f"{len(sentence.words)} gold words, {len(pairs)} pipeline tokens"
            )
        for index, (info, decision) in enumerate(pairs):
            rows.append(
                {
                    "sentence_id": sentence.sentence_id,
                    "token": decision.token,
                    "gold": sentence.labels[index],
                    "pred": decision.label,
                    "stage2": decision.stage2_label or decision.label,
                    "deferred": decision.deferred,
                    "rules": decision.rules,
                    "stage1": info.category,
                    "category": (sentence.categories[index] if sentence.categories else info.category),
                }
            )
    return rows


def _share(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def selective_report(rows: list[dict]) -> dict:
    rows = [row for row in rows if row["gold"] != IGNORE_LABEL]
    accepted = [row for row in rows if not row["deferred"]]
    deferred = [row for row in rows if row["deferred"]]

    def correct(items, key):
        return sum(1 for row in items if row[key] == row["gold"])

    fixes = [row for row in rows if row["stage2"] != row["gold"] and row["pred"] == row["gold"]]
    breaks = [row for row in rows if row["stage2"] == row["gold"] and row["pred"] != row["gold"]]
    by_rule: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        if row["pred"] == row["stage2"]:
            continue
        rule = row["rules"][-1] if row["rules"] else "unattributed"
        by_rule[rule]["changed"] += 1
        by_rule[rule]["fixed"] += row["pred"] == row["gold"]
        by_rule[rule]["broke"] += row["stage2"] == row["gold"]

    by_stage1: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_stage1[row["stage1"]].append(row)

    return {
        "tokens": len(rows),
        "coverage": _share(len(accepted), len(rows)),
        "accepted_accuracy": _share(correct(accepted, "stage2"), len(accepted)),
        "deferred_tokens": len(deferred),
        "deferred_accuracy_stage2": _share(correct(deferred, "stage2"), len(deferred)),
        "deferred_accuracy_final": _share(correct(deferred, "pred"), len(deferred)),
        "stage2_accuracy": _share(correct(rows, "stage2"), len(rows)),
        "final_accuracy": _share(correct(rows, "pred"), len(rows)),
        "stage3_fixes": len(fixes),
        "stage3_breaks": len(breaks),
        "by_rule": {rule: dict(counts) for rule, counts in sorted(by_rule.items())},
        "by_stage1_category": {
            category: {
                "tokens": len(items),
                "share_deferred": _share(sum(row["deferred"] for row in items), len(items)),
                "stage2_accuracy": _share(correct(items, "stage2"), len(items)),
                "final_accuracy": _share(correct(items, "pred"), len(items)),
            }
            for category, items in sorted(by_stage1.items())
        },
    }


def selective_markdown(report: dict) -> list[str]:
    def fmt(value):
        return "n/a" if value is None else f"{value:.4f}"

    lines = [
        "## Confidence gate and Stage 3",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Coverage (accepted by Stage 2) | {fmt(report['coverage'])} |",
        f"| Accuracy on accepted words | {fmt(report['accepted_accuracy'])} |",
        f"| Deferred words | {report['deferred_tokens']} |",
        f"| Accuracy on deferred words, Stage 2 label | {fmt(report['deferred_accuracy_stage2'])} |",
        f"| Accuracy on deferred words, after Stage 3 | {fmt(report['deferred_accuracy_final'])} |",
        f"| Stage 3 fixes / breaks | {report['stage3_fixes']} / {report['stage3_breaks']} |",
        "",
        "| Stage-3 change | Changed | Fixed | Broke |",
        "|---|---|---|---|",
    ]
    for rule, counts in report["by_rule"].items():
        lines.append(f"| {rule} | {counts.get('changed', 0)} | {counts.get('fixed', 0)} | {counts.get('broke', 0)} |")
    lines += [
        "",
        "| Stage-1 category | Tokens | Deferred | Stage-2 acc. | Final acc. |",
        "|---|---|---|---|---|",
    ]
    for category, values in report["by_stage1_category"].items():
        lines.append(
            f"| {category} | {values['tokens']} | {fmt(values['share_deferred'])} | "
            f"{fmt(values['stage2_accuracy'])} | {fmt(values['final_accuracy'])} |"
        )
    lines.append("")
    return lines
