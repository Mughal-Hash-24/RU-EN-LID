"""Temperature scaling for Stage 2's confidences (fit on validation, never on test).

A fine-tuned encoder keeps getting more confident after its labels stop improving, so the raw
softmax maximum is a poor "am I sure?" signal. One scalar T > 0, fitted to minimise validation
NLL, rescales the logits (softmax(z / T)); labels never change, only confidences do. T is
stored in the checkpoint's config and applied by ``predict.py``; tau must be re-tuned afterwards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fit_temperature(logits, labels, steps: int = 200) -> float:
    """logits [n, classes], labels [n] (already filtered to scored words)."""
    import torch

    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=steps, line_search_fn="strong_wolfe")
    loss_fn = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.detach().exp())


def expected_calibration_error(confidences, correct, bins: int = 10) -> float:
    import torch

    edges = torch.linspace(0, 1, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        inside = (confidences > low) & (confidences <= high)
        if inside.any():
            error += float(inside.float().mean() * (confidences[inside].mean() - correct[inside].float().mean()).abs())
    return error


def main(argv: list[str] | None = None) -> int:
    import torch

    from ..candidate_expansion.expander import CandidateExpander
    from ..pipeline import resolve_resources
    from .dataset import IGNORE_LABEL, read_any
    from .predict import TokenLIDPredictor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--valid", required=True, help="validation CSV (never the test split)")
    parser.add_argument(
        "--resources", help="Stage-1 lexicon folder (default: the one saved with --checkpoint, else data/resources)"
    )
    args = parser.parse_args(argv)

    predictor = TokenLIDPredictor(args.checkpoint)
    predictor.temperature = 1.0
    expander = CandidateExpander.from_resources(resolve_resources(args.checkpoint, args.resources)[0])
    label2id = {label: index for index, label in predictor.id2label.items()}
    all_logits, all_labels = [], []
    for sentence in read_any(args.valid):
        candidates = [expander.expand_token(word) for word in sentence.words]
        logits = predictor.word_logits(sentence.words, candidates)
        for word_logits, label in zip(logits, sentence.labels, strict=True):
            if word_logits is not None and label != IGNORE_LABEL:
                all_logits.append(word_logits)
                all_labels.append(label2id[label])
    logits = torch.stack(all_logits)
    labels = torch.tensor(all_labels)
    temperature = fit_temperature(logits, labels)

    def report(t: float) -> dict:
        probabilities = torch.softmax(logits / t, dim=-1)
        confidences, predictions = probabilities.max(dim=-1)
        return {
            "nll": round(float(torch.nn.functional.cross_entropy(logits / t, labels)), 4),
            "ece": round(expected_calibration_error(confidences, predictions == labels), 4),
        }

    summary = {"temperature": round(temperature, 4), "before": report(1.0), "after": report(temperature)}
    config_path = Path(args.checkpoint, "config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["temperature"] = temperature
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
