"""Stage 3 rules: lexicon priors, confidence gating and consistency checks.

Every rule only fires when the classifier is below the confidence threshold, so
verification cannot override a confident contextual decision.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..candidate_expansion.expander import TokenCandidates

DEFAULT_TAU = 0.55


@dataclass(frozen=True)
class Decision:
    token: str
    label: str
    confidence: float
    rules: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "token": self.token,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "rules": list(self.rules),
        }


def apply_rules(
    tokens: list[TokenCandidates],
    predictions: list[tuple[str, float]],
    tau: float = DEFAULT_TAU,
) -> list[Decision]:
    decisions: list[Decision] = []
    for info, (label, confidence) in zip(tokens, predictions):
        final = label
        rules: list[str] = []

        if info.category in {"variant", "both"} and final == "EN" and confidence < tau:
            final = "UR"
            rules.append("variant-consistency")
        elif (
            info.category == "standard"
            and info.english_reading
            and not info.urdu_canonicals
            and final == "UR"
            and confidence < tau
        ):
            final = "EN"
            rules.append("english-lexicon-prior")
        elif (
            info.category == "standard"
            and info.urdu_canonicals
            and not info.english_reading
            and final == "EN"
            and confidence < tau
        ):
            final = "UR"
            rules.append("urdu-lexicon-prior")

        decisions.append(Decision(token=info.surface, label=final, confidence=confidence, rules=tuple(rules)))
    return decisions
