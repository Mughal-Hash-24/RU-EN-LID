"""Token-level metrics and error attribution.

CMI is the Code-Mixing Index of Gambäck & Das (2014), computed per utterance:

    CMI = 100 * (1 - max_i w_i / (n - u))   if n > u,   else 0

where w_i counts the tokens of language i, n is the utterance length and u counts
language-independent tokens (OTH and IGN here). For two languages it ranges from
0 (monolingual) to 50 (evenly mixed). A corpus is summarised by the mean over its
sentences, because pooling all tokens into one "utterance" hides the mixing.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from ..classifier.dataset import IGNORE_LABEL

LANGUAGE_INDEPENDENT = frozenset({"OTH", IGNORE_LABEL})


def _check_lengths(gold: list[str], pred: list[str]) -> None:
    if len(gold) != len(pred):
        raise ValueError(f"gold and pred differ in length: {len(gold)} vs {len(pred)}")


def accuracy(gold: list[str], pred: list[str]) -> float:
    _check_lengths(gold, pred)
    if not gold:
        return 0.0
    return sum(g == p for g, p in zip(gold, pred, strict=True)) / len(gold)


def macro_f1(gold: list[str], pred: list[str]) -> float:
    """Unweighted mean F1 over every label seen in gold or pred, in one pass over the data.

    Per label, F1 = 2TP / (2TP + FP + FN) = 2TP / (gold count + predicted count).
    """
    _check_lengths(gold, pred)
    gold_counts, pred_counts = Counter(gold), Counter(pred)
    labels = gold_counts.keys() | pred_counts.keys()
    if not labels:
        return 0.0
    hits = Counter(g for g, p in zip(gold, pred, strict=True) if g == p)
    return sum(2 * hits[label] / (gold_counts[label] + pred_counts[label]) for label in labels) / len(labels)


def cmi(labels: list[str], independent: frozenset[str] = LANGUAGE_INDEPENDENT) -> float:
    """Gambäck & Das (2014) Code-Mixing Index of one utterance (see module docstring)."""
    language_counts = Counter(label for label in labels if label not in independent)
    tagged = sum(language_counts.values())  # n - u
    if not tagged:
        return 0.0
    return 100.0 * (1.0 - max(language_counts.values()) / tagged)


def mean_sentence_cmi(rows: list[dict], key: str = "gold") -> float:
    """Mean per-sentence CMI; rows without a ``sentence_id`` count as one utterance."""
    sentences: dict[object, list[str]] = defaultdict(list)
    for row in rows:
        sentences[row.get("sentence_id")].append(row[key])
    if not sentences:
        return 0.0
    return sum(cmi(labels) for labels in sentences.values()) / len(sentences)


def summarize(rows: list[dict]) -> dict:
    """rows: dicts with token, gold, pred, category (gold category when available) and,
    optionally, sentence_id. Rows whose gold label is IGN are left out of every score and
    only counted (CMI still sees them, as language-independent tokens)."""
    ignored = sum(row["gold"] == IGNORE_LABEL for row in rows)
    sentence_cmi = mean_sentence_cmi(rows)
    rows = [row for row in rows if row["gold"] != IGNORE_LABEL]
    gold = [row["gold"] for row in rows]
    pred = [row["pred"] for row in rows]

    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_category[row.get("category") or "unknown"].append(row)
    category_report = {}
    for category, items in sorted(by_category.items()):
        items_gold = [item["gold"] for item in items]
        items_pred = [item["pred"] for item in items]
        category_report[category] = {
            "tokens": len(items),
            "accuracy": round(accuracy(items_gold, items_pred), 4),
            "macro_f1": round(macro_f1(items_gold, items_pred), 4),
        }

    error_counts = Counter(row.get("category") or "unknown" for row in rows if row["gold"] != row["pred"])
    errors = sum(error_counts.values())
    return {
        "tokens": len(rows),
        "ignored": ignored,
        "accuracy": round(accuracy(gold, pred), 4),
        "macro_f1": round(macro_f1(gold, pred), 4),
        "cmi": round(sentence_cmi, 2),
        "per_category": category_report,
        "errors": errors,
        "error_share_by_cause": {
            category: round(count / errors, 4) for category, count in sorted(error_counts.items())
        },
    }
