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
    # True when Stage 2 was below tau: the label was handed to Stage 3 instead of being accepted.
    deferred: bool = False
    # Stage 2's label before any Stage-3 change (lets evaluation count fixes and breaks).
    stage2_label: str = ""

    def as_dict(self) -> dict:
        return {
            "token": self.token,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "rules": list(self.rules),
            "deferred": self.deferred,
            "stage2_label": self.stage2_label,
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

        decisions.append(
            Decision(
                token=info.surface,
                label=final,
                confidence=confidence,
                rules=tuple(rules),
                deferred=confidence < tau,
                stage2_label=label,
            )
        )
    return decisions


# Each Stage-1 category gets one Stage-3 resolver. Standard words have the lexicon priors and
# variants have variant-consistency (an Urdu prior); only ambiguous and unknown words have no
# lexicon evidence, so only they are re-decoded from context. Re-decoding variants as well was
# tried on the running example: "same issue he" turned he (= hai) into English.
CONTEXT_CATEGORIES = frozenset({"ambiguous", "unknown"})


def apply_context(
    tokens: list[TokenCandidates],
    decisions: list[Decision],
    probabilities: list[dict[str, float]],
    penalty: float,
) -> list[Decision]:
    """Second Stage-3 pass: re-decode deferred ambiguous/unknown words from their neighbours.

    Accepted words, rule-fixed words and all other categories are pinned; see ``context.py``.
    penalty <= 0 returns the decisions unchanged.
    """
    if penalty <= 0 or not decisions:
        return decisions
    from dataclasses import replace

    from .context import decode

    pinned = [
        None
        if decision.deferred and not decision.rules and info.category in CONTEXT_CATEGORIES
        else decision.label
        for info, decision in zip(tokens, decisions)
    ]
    labels = decode(probabilities, pinned, penalty)
    updated = []
    for decision, pin, label in zip(decisions, pinned, labels):
        if pin is None and label != decision.label:
            decision = replace(decision, label=label, rules=decision.rules + ("context",))
        updated.append(decision)
    return updated
