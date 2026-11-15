"""End-to-end pipeline: Stage 1 expansion -> Stage 2 classifier -> Stage 3 verification."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .candidate_expansion.expander import CandidateExpander, TokenCandidates
from .verification.rules import DEFAULT_TAU, apply_context, apply_rules
from .verification.tau import resolve_tau

DEFAULT_RESOURCES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "resources")
# A candidate-conditioned checkpoint keeps a copy of the Stage-1 lexicons it was trained with here (train.py):
# its Stage-1 tags mean at inference what they meant in training only if the same lexicons produce them.
CHECKPOINT_RESOURCES = "stage1_resources"
RESOURCE_FILES = ("english_seed.txt", "roman_urdu_seed.txt", "variant_pairs.tsv")


def resolve_resources(checkpoint: str | None, resources_dir: str | None = None) -> tuple[str, str]:
    """Stage-1 lexicon folder for a run, and where it came from: explicit > saved with the checkpoint > seeds."""
    if resources_dir:
        return resources_dir, "explicit"
    if checkpoint and os.path.isdir(os.path.join(checkpoint, CHECKPOINT_RESOURCES)):
        return os.path.join(checkpoint, CHECKPOINT_RESOURCES), "saved with the checkpoint"
    return DEFAULT_RESOURCES, "default"


# Model-free stand-in for Stage 2, used for demos and tests until a checkpoint is trained.
# Per kind of word: (label, confidence) for the gate and the lexicon rules, and the
# distribution the context decoder sees. The confidences are set below the default tau on
# purpose for variants and ambiguous words, so the fallback exercises Stage 3; they are not
# the max of the distribution, and the decoder uses only the distribution.
_HEURISTIC: dict[str, tuple[str, float, dict[str, float]]] = {
    "variant": ("UR", 0.40, {"UR": 0.6, "EN": 0.4, "OTH": 0.0}),
    "ambiguous": ("EN", 0.30, {"UR": 0.5, "EN": 0.5, "OTH": 0.0}),
    "english": ("EN", 0.90, {"UR": 0.05, "EN": 0.9, "OTH": 0.05}),
    "urdu": ("UR", 0.90, {"UR": 0.9, "EN": 0.05, "OTH": 0.05}),
    "none": ("OTH", 0.0, {"UR": 0.0, "EN": 0.0, "OTH": 1.0}),
}


def _heuristic_kind(info: TokenCandidates) -> str:
    if info.category in {"variant", "both"}:
        return "variant"
    if info.category == "ambiguous":
        return "ambiguous"
    if info.english_reading and not info.urdu_canonicals:
        return "english"
    if info.urdu_canonicals and not info.english_reading:
        return "urdu"
    return "none"


def heuristic_predictions(tokens: list[TokenCandidates]) -> list[tuple[str, float]]:
    """Model-free fallback: (label, confidence) per word."""
    return [_HEURISTIC[_heuristic_kind(info)][:2] for info in tokens]


def heuristic_distributions(tokens: list[TokenCandidates]) -> list[dict[str, float]]:
    """Model-free fallback: the distribution the context decoder sees (ambiguous = no preference)."""
    return [dict(_HEURISTIC[_heuristic_kind(info)][2]) for info in tokens]


class LIDPipeline:
    def __init__(
        self,
        expander: CandidateExpander,
        predictor=None,
        tau: float = DEFAULT_TAU,
        context_penalty: float = 0.0,
        tau_source: str = "explicit",
    ) -> None:
        self.expander = expander
        self.predictor = predictor
        self.tau = tau
        self.tau_source = tau_source
        self.resources_dir: str | None = None  # set by from_resources()
        self.resources_source = "explicit"
        # 0 = context decoder off (iteration-1 behaviour). Tune on validation with scripts/tune_stage3.py.
        self.context_penalty = context_penalty

    @classmethod
    def from_resources(
        cls,
        resources_dir: str | None = None,
        checkpoint: str | None = None,
        tau: float | None = None,
        context_penalty: float = 0.0,
    ) -> LIDPipeline:
        """tau=None: the value tuned for the checkpoint (its stage3.json, local or on the hub), else DEFAULT_TAU.
        resources_dir=None: the lexicons saved with the checkpoint, else data/resources."""
        tau, source = resolve_tau(tau, checkpoint)
        resources_dir, resources_source = resolve_resources(checkpoint, resources_dir)
        expander = CandidateExpander.from_resources(resources_dir)
        predictor = None
        if checkpoint:
            from .classifier.predict import TokenLIDPredictor

            predictor = TokenLIDPredictor(checkpoint)
        pipeline = cls(expander, predictor, tau, context_penalty, source)
        pipeline.resources_dir, pipeline.resources_source = resources_dir, resources_source
        return pipeline

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
        return list(zip(tokens, decisions, strict=True))

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


def probability(text: str) -> float:
    """argparse type for tau: a number in [0, 1]."""
    value = float(text)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1, got {value}")
    return value


def non_negative(text: str) -> float:
    """argparse type for the context penalty: a number >= 0."""
    value = float(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the RU-EN LID pipeline on text.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--checkpoint", help="trained token-classification checkpoint (optional)")
    parser.add_argument(
        "--resources", help="Stage-1 lexicon folder (default: the one saved with --checkpoint, else data/resources)"
    )
    parser.add_argument(
        "--tau",
        type=probability,
        help=f"Stage 3 threshold (default: the value tuned for --checkpoint, else {DEFAULT_TAU})",
    )
    parser.add_argument(
        "--context-penalty", type=non_negative, default=0.0, help="Stage-3 switch penalty for deferred words (0 = off)"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    pipeline = LIDPipeline.from_resources(
        args.resources, checkpoint=args.checkpoint, tau=args.tau, context_penalty=args.context_penalty
    )
    print(f"[stage1] resources = {pipeline.resources_dir} ({pipeline.resources_source})", file=sys.stderr)
    print(f"[stage3] tau = {pipeline.tau} ({pipeline.tau_source})", file=sys.stderr)
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
