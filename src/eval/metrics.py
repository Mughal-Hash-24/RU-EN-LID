"""Token-level metrics and error attribution.

CMI follows Gambäck & Das (2014) with the length normalization used in the
code-mixing literature: CMI = 100 * (1 - sum(p_l^2)) / (1 - 1/n), 0 for n <= 1.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from ..classifier.dataset import IGNORE_LABEL


def accuracy(gold: list[str], pred: list[str]) -> float:
    if not gold:
        return 0.0
    return sum(1 for g, p in zip(gold, pred) if g == p) / len(gold)


def _f1(gold: list[str], pred: list[str], label: str) -> tuple[float, float, float]:
    tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
    fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
    fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def macro_f1(gold: list[str], pred: list[str]) -> float:
    labels = sorted(set(gold) | set(pred))
    if not labels:
        return 0.0
    return sum(_f1(gold, pred, label)[2] for label in labels) / len(labels)


def cmi(labels: list[str]) -> float:
    n = len(labels)
    if n <= 1:
        return 0.0
    counts = Counter(labels)
    sum_squares = sum((count / n) ** 2 for count in counts.values())
    return 100.0 * (1.0 - sum_squares) / (1.0 - 1.0 / n)


def summarize(rows: list[dict]) -> dict:
    """rows: dicts with token, gold, pred and category (gold category when available).

    Rows whose gold label is IGN are left out of every score and only counted.
    """
    ignored = sum(1 for row in rows if row["gold"] == IGNORE_LABEL)
    rows = [row for row in rows if row["gold"] != IGNORE_LABEL]
    gold = [row["gold"] for row in rows]
    pred = [row["pred"] for row in rows]
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_category[row.get("category") or "unknown"].append(row)

    category_report = {}
    for category, items in sorted(by_category.items()):
        category_report[category] = {
            "tokens": len(items),
            "accuracy": round(accuracy([i["gold"] for i in items], [i["pred"] for i in items]), 4),
            "macro_f1": round(macro_f1([i["gold"] for i in items], [i["pred"] for i in items]), 4),
        }

    errors = [row for row in rows if row["gold"] != row["pred"]]
    error_counts = Counter(row.get("category") or "unknown" for row in errors)
    error_share = {
        category: round(count / len(errors), 4) if errors else 0.0 for category, count in sorted(error_counts.items())
    }

    return {
        "tokens": len(rows),
        "ignored": ignored,
        "accuracy": round(accuracy(gold, pred), 4),
        "macro_f1": round(macro_f1(gold, pred), 4),
        "cmi": round(cmi(gold), 2),
        "per_category": category_report,
        "errors": len(errors),
        "error_share_by_cause": error_share,
    }
