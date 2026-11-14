"""End-to-end pipeline: Stage 1 expansion -> Stage 2 classifier -> Stage 3 verification."""

from __future__ import annotations

import argparse
import json
import os

from .candidate_expansion.expander import CandidateExpander, TokenCandidates
from .verification.rules import DEFAULT_TAU, apply_context, apply_rules

DEFAULT_RESOURCES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "resources")


def heuristic_predictions(tokens: list[TokenCandidates]) -> list[tuple[str, float]]:
    """Model-free fallback used for demos/tests until a checkpoint is trained."""
    predictions: list[tuple[str, float]] = []
    for info in tokens:
        if info.category in {"variant", "both"}:
            predictions.append(("UR", 0.40))
        elif info.category == "ambiguous":
            predictions.append(("EN", 0.30))
        elif info.english_reading and not info.urdu_canonicals:
            predictions.append(("EN", 0.90))
        elif info.urdu_canonicals and not info.english_reading:
            predictions.append(("UR", 0.90))
        else:
            predictions.append(("OTH", 0.0))
    return predictions


def heuristic_distributions(tokens: list[TokenCandidates]) -> list[dict[str, float]]:
    """Stand-in Stage-2 distributions for the model-free fallback (ambiguous = no preference)."""
    distributions: list[dict[str, float]] = []
    for info in tokens:
        if info.category == "ambiguous":
            distributions.append({"UR": 0.5, "EN": 0.5, "OTH": 0.0})
        elif info.category in {"variant", "both"}:
            distributions.append({"UR": 0.6, "EN": 0.4, "OTH": 0.0})
        elif info.english_reading and not info.urdu_canonicals:
            distributions.append({"UR": 0.05, "EN": 0.9, "OTH": 0.05})
        elif info.urdu_canonicals and not info.english_reading:
            distributions.append({"UR": 0.9, "EN": 0.05, "OTH": 0.05})
        else:
            distributions.append({"UR": 0.0, "EN": 0.0, "OTH": 1.0})
    return distributions


class LIDPipeline:
    def __init__(
        self,
        expander: CandidateExpander,
        predictor=None,
        tau: float = DEFAULT_TAU,
        context_penalty: float = 0.0,
    ) -> None:
        self.expander = expander
        self.predictor = predictor
        self.tau = tau
        # 0 = context decoder off (iteration-1 behaviour). Tune on validation with scripts/tune_stage3.py.
        self.context_penalty = context_penalty

    @classmethod
    def from_resources(
        cls,
        resources_dir: str = DEFAULT_RESOURCES,
        checkpoint: str | None = None,
        tau: float = DEFAULT_TAU,
        context_penalty: float = 0.0,
    ) -> "LIDPipeline":
        expander = CandidateExpander.from_resources(resources_dir)
        predictor = None
        if checkpoint:
            from .classifier.predict import TokenLIDPredictor

            predictor = TokenLIDPredictor(checkpoint)
        return cls(expander, predictor, tau, context_penalty)

    def run(self, text: str):
        tokens = self.expander.expand(text)
        if self.predictor is not None:
            outputs = self.predictor.predict_words([token.surface for token in tokens], candidates=tokens)
            predictions = [(output.label, output.confidence) for output in outputs]
            distributions = [output.probs or {output.label: 1.0} for output in outputs]
        else:
            predictions = heuristic_predictions(tokens)
            distributions = heuristic_distributions(tokens)
        decisions = apply_rules(tokens, predictions, self.tau)
        decisions = apply_context(tokens, decisions, distributions, self.context_penalty)
        return list(zip(tokens, decisions))

    def run_records(self, text: str) -> list[dict]:
        return [
            {
                "token": decision.token,
                "label": decision.label,
                "confidence": round(decision.confidence, 3),
                "category": info.category,
                "english_reading": info.english_reading,
                "urdu_canonicals": list(info.urdu_canonicals),
                "rules": list(decision.rules),
                "deferred": decision.deferred,
                "stage2_label": decision.stage2_label,
            }
            for info, decision in self.run(text)
        ]


def format_records(records: list[dict]) -> str:
    header = (
        f"{'token':<16}{'category':<12}{'readings':<24}{'label':<7}{'conf':<6}rules"
    )
    lines = [header, "-" * len(header)]
    for record in records:
        readings = "EN" if record["english_reading"] else ""
        if record["urdu_canonicals"]:
            readings += ("/" if readings else "") + ",".join(record["urdu_canonicals"])
        lines.append(
            f"{record['token']:<16}{record['category']:<12}{readings:<24}{record['label']:<7}"
            f"{record['confidence']:<6}{','.join(record['rules'])}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the RU-EN LID pipeline on text.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--checkpoint", help="trained token-classification checkpoint (optional)")
    parser.add_argument("--resources", default=DEFAULT_RESOURCES)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument(
        "--context-penalty", type=float, default=0.0, help="Stage-3 switch penalty for deferred words (0 = off)"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    pipeline = LIDPipeline.from_resources(
        args.resources, checkpoint=args.checkpoint, tau=args.tau, context_penalty=args.context_penalty
    )
    records = pipeline.run_records(args.text)
    if args.json:
        print(json.dumps(records, indent=2, ensure_ascii=False))
    else:
        if args.checkpoint is None:
            print(
                "[note] heuristic fallback (no --checkpoint): ambiguous tokens default to EN, "
                "then Stage-3 rules apply.\n"
            )
        print(format_records(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
