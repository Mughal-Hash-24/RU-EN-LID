"""Confidence-calibration diagnostics for Stage 2 (standard library only).

Expected calibration error (ECE; Guo et al., 2017): words are binned by confidence into equal-width bins,
and the gap between each bin's mean confidence and its accuracy is averaged, weighted by bin size. A
calibrated model has ECE 0: among words with confidence 0.8, 80% are right. Temperature scaling divides the
logits by one scalar T fitted on validation data (minimum negative log-likelihood of the gold labels); it
never changes a predicted label, only its confidence.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def expected_calibration_error(confidences: Sequence[float], correct: Sequence[bool], bins: int = 15) -> float:
    if not confidences:
        return 0.0
    totals = [[0, 0.0, 0] for _ in range(bins)]  # words, summed confidence, correct words
    for confidence, right in zip(confidences, correct):
        cell = totals[min(int(confidence * bins), bins - 1)]
        cell[0] += 1
        cell[1] += confidence
        cell[2] += bool(right)
    return sum(abs(summed - hits) for _, summed, hits in totals) / len(confidences)


def reliability(confidences: Sequence[float], correct: Sequence[bool], edges: Sequence[float]) -> list[dict]:
    """Words, mean confidence and accuracy per confidence band [edges[i], edges[i+1]); the last band is closed."""
    rows = []
    for index, (low, high) in enumerate(zip(edges, edges[1:])):
        last = index == len(edges) - 2
        members = [
            (confidence, right)
            for confidence, right in zip(confidences, correct)
            if low <= confidence < high or (last and confidence == high)
        ]
        rows.append({
            "band": [low, high],
            "words": len(members),
            "mean_confidence": sum(c for c, _ in members) / len(members) if members else None,
            "accuracy": sum(bool(r) for _, r in members) / len(members) if members else None,
        })
    return rows


def softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    scaled = [value / temperature for value in logits]
    top = max(scaled)
    exps = [math.exp(value - top) for value in scaled]
    total = sum(exps)
    return [value / total for value in exps]


def negative_log_likelihood(rows: Sequence[Sequence[float]], gold: Sequence[int], temperature: float) -> float:
    """Mean -log p(gold) under softmax(logits / temperature)."""
    total = 0.0
    for logits, label in zip(rows, gold):
        scaled = [value / temperature for value in logits]
        top = max(scaled)
        total += top + math.log(sum(math.exp(value - top) for value in scaled)) - scaled[label]
    return total / len(rows)


def fit_temperature(
    rows: Sequence[Sequence[float]], gold: Sequence[int], low: float = 0.05, high: float = 20.0, steps: int = 60
) -> float:
    """The temperature with the lowest validation NLL (golden-section search on log T; NLL is unimodal in T)."""
    ratio = (math.sqrt(5) - 1) / 2
    a, b = math.log(low), math.log(high)
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc = negative_log_likelihood(rows, gold, math.exp(c))
    fd = negative_log_likelihood(rows, gold, math.exp(d))
    for _ in range(steps):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = negative_log_likelihood(rows, gold, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = negative_log_likelihood(rows, gold, math.exp(d))
    return math.exp((a + b) / 2)
