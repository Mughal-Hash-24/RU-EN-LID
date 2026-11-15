"""Stage 3 threshold τ: selection on validation data, and the tuned value saved with a checkpoint.

Every Stage 3 rule looks at one word at a time and fires only when its Stage 2 confidence is below τ, so the
Stage 3 decisions change only at the confidences of the words some rule would re-label. ``candidate_taus``
lists exactly those change points, each one twice (the confidence c itself, the largest τ that still leaves
that word alone, and c nudged just above, the smallest τ that re-labels it), plus τ = 0 and an optional grid;
``sweep`` re-applies ``apply_rules`` at every candidate; ``select_tau`` keeps the best validation macro-F1
(then accuracy) and, among the τ values that tie on it, the largest one before the score first drops: the
rules re-check as many low-confidence words as they can without costing anything on validation. Stage 2 runs
once, outside this module, so nothing here needs torch.

The selected τ is saved as ``stage3.json`` next to a local checkpoint; a model published on the Hugging Face
Hub carries the same file in its repository. ``resolve_tau`` is the precedence the pipeline and
``scripts/evaluate.py`` use: an explicit τ, else the tuned value of that checkpoint (local folder or hub id),
else ``DEFAULT_TAU``.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..candidate_expansion.expander import TokenCandidates
from ..classifier.dataset import IGNORE_LABEL
from ..eval.metrics import accuracy, macro_f1
from .rules import DEFAULT_TAU, apply_rules

STAGE3_CONFIG = "stage3.json"
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


@dataclass(frozen=True)
class ScoredSentence:
    """One sentence after Stages 1 and 2: what Stage 3 needs, plus the gold labels."""

    tokens: tuple[TokenCandidates, ...]
    predictions: tuple[tuple[str, float], ...]
    gold: tuple[str, ...]


@dataclass(frozen=True)
class SweepPoint:
    """Stage 2 + Stage 3 at one τ, on the scored words (gold IGN words are left out)."""

    tau: float
    tokens: int
    accuracy: float
    macro_f1: float
    below_tau: int  # words re-checked: Stage 2 confidence < τ
    fired: int  # words a rule re-labelled
    fixed: int  # ... from wrong to right
    broke: int  # ... from right to wrong
    by_rule: dict[str, tuple[int, int, int]] = field(default_factory=dict)  # rule -> (fired, fixed, broke)

    def as_dict(self) -> dict:
        return {
            "tau": self.tau,
            "tokens": self.tokens,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "below_tau": self.below_tau,
            "fired": self.fired,
            "fixed": self.fixed,
            "broke": self.broke,
            "by_rule": {rule: dict(zip(("fired", "fixed", "broke"), counts)) for rule, counts in self.by_rule.items()},
        }


def _kept(rules_applied: tuple[str, ...], rules: Collection[str] | None) -> tuple[str, ...]:
    """The applied rules, or () when ``rules`` restricts the sweep to other rules."""
    if rules is None or set(rules_applied) <= set(rules):
        return rules_applied
    return ()


def evaluate_tau(sentences: Sequence[ScoredSentence], tau: float, rules: Collection[str] | None = None) -> SweepPoint:
    """Score Stage 3 at ``tau`` against Stage 2 alone; ``rules`` keeps only those rules' changes (None: all)."""
    gold: list[str] = []
    pred: list[str] = []
    below = fired = fixed = broke = 0
    by_rule: dict[str, list[int]] = {}
    for sentence in sentences:
        decisions = apply_rules(list(sentence.tokens), list(sentence.predictions), tau)
        for (stage2, confidence), decision, truth in zip(sentence.predictions, decisions, sentence.gold):
            if truth == IGNORE_LABEL:
                continue
            applied = _kept(decision.rules, rules)
            final = decision.label if applied else stage2
            gold.append(truth)
            pred.append(final)
            below += confidence < tau
            if not applied:
                continue
            fired += 1
            counts = by_rule.setdefault("+".join(applied), [0, 0, 0])
            counts[0] += 1
            if final == truth and stage2 != truth:
                fixed += 1
                counts[1] += 1
            elif stage2 == truth and final != truth:
                broke += 1
                counts[2] += 1
    return SweepPoint(
        tau=tau,
        tokens=len(gold),
        accuracy=accuracy(gold, pred),
        macro_f1=macro_f1(gold, pred),
        below_tau=below,
        fired=fired,
        fixed=fixed,
        broke=broke,
        by_rule={rule: tuple(counts) for rule, counts in sorted(by_rule.items())},
    )


def change_points(sentences: Sequence[ScoredSentence], rules: Collection[str] | None = None) -> list[float]:
    """Confidences of the scored words that some rule would re-label if τ had no limit, ascending."""
    points: set[float] = set()
    for sentence in sentences:
        decisions = apply_rules(list(sentence.tokens), list(sentence.predictions), math.inf)
        for (_, confidence), decision, truth in zip(sentence.predictions, decisions, sentence.gold):
            if truth != IGNORE_LABEL and _kept(decision.rules, rules):
                points.add(confidence)
    return sorted(points)


def candidate_taus(
    sentences: Sequence[ScoredSentence], grid: float | None = 0.01, rules: Collection[str] | None = None
) -> list[float]:
    """τ = 0, every change point c and c nudged just above it, and an optional grid; all within [0, 1].

    A rule fires when the confidence is below τ, so the decisions are constant on (c_k, c_k+1]: these
    candidates cover every distinct Stage 3 outcome on ``sentences`` and both ends of every interval; the grid
    only adds readable rows to the sweep table.
    """
    taus = {0.0}
    for point in change_points(sentences, rules):
        if point < 1.0:
            taus.update((point, math.nextafter(point, math.inf)))
    if grid:
        if not 0.0 < grid <= 1.0:
            raise ValueError(f"grid step must be in (0, 1], got {grid}")
        taus.update(round(step * grid, 10) for step in range(1, round(1.0 / grid) + 1))
    return sorted(tau for tau in taus if tau <= 1.0)


def sweep(
    sentences: Sequence[ScoredSentence], taus: Sequence[float], rules: Collection[str] | None = None
) -> list[SweepPoint]:
    return [evaluate_tau(sentences, tau, rules) for tau in taus]


def score(point: SweepPoint) -> tuple[float, float]:
    return point.macro_f1, point.accuracy


def select_tau(points: Sequence[SweepPoint]) -> SweepPoint:
    """The largest τ that keeps the best score (macro-F1, then accuracy) before the score first drops.

    Walking up from τ = 0: find the first τ with the best score, then keep raising τ while the score stays
    the same; the last τ before it drops is selected. τ = 0 (no rule can fire) must be among the points, so
    the selected τ never scores below Stage 2 alone on the data the points were scored on. With the change
    points of ``candidate_taus`` among the points, the selected τ is exact: just above it, a rule re-labels a
    word and the score drops.
    """
    if not any(point.tau == 0.0 for point in points):
        raise ValueError("the candidate taus must include 0 (Stage 2 alone)")
    ordered = sorted(points, key=lambda point: point.tau)
    best = max(score(point) for point in ordered)
    index = next(position for position, point in enumerate(ordered) if score(point) == best)
    while index + 1 < len(ordered) and score(ordered[index + 1]) == best:
        index += 1
    return ordered[index]


def config_path(checkpoint: str | None) -> Path | None:
    """Where the tuned τ of a local checkpoint folder lives; None without a checkpoint or for a hub id."""
    if not checkpoint:
        return None
    folder = Path(checkpoint)
    return folder / STAGE3_CONFIG if folder.is_dir() else None


def hub_config_path(model_id: str) -> Path | None:
    """The tuned τ of a Hugging Face model id: its ``stage3.json``, downloaded once into the Hugging Face cache.

    Needs ``huggingface_hub``, which comes with ``transformers`` (a hub id can't be loaded without it). None when
    it isn't installed, the repository has no such file, the id isn't on the hub, or there is no network and no
    cached copy.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    try:
        return Path(hf_hub_download(model_id, STAGE3_CONFIG))
    except Exception:  # noqa: BLE001 (missing file, unknown or invalid id, offline: all mean "not tuned")
        return None


def save_tuned_tau(path: Path, record: dict) -> None:
    if not 0.0 <= record["tau"] <= 1.0:
        raise ValueError(f"tau must be in [0, 1], got {record['tau']}")
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def load_tuned_tau(path: Path) -> float:
    tau = json.loads(path.read_text(encoding="utf-8")).get("tau")
    if isinstance(tau, bool) or not isinstance(tau, (int, float)) or not 0.0 <= tau <= 1.0:
        raise ValueError(f"{path}: 'tau' must be a number in [0, 1], got {tau!r}")
    return float(tau)


def is_stale(path: Path) -> bool:
    """True when the checkpoint's weights were written after its tuned τ (the model was retrained).

    A folder downloaded with ``hf download --local-dir`` (it holds ``.cache/huggingface/``) is never stale: its
    files come from one repository revision, and their times only say in which order they were downloaded.
    """
    if (path.parent / ".cache" / "huggingface").is_dir():
        return False
    saved = path.stat().st_mtime
    return any(
        (path.parent / name).is_file() and (path.parent / name).stat().st_mtime > saved for name in WEIGHT_FILES
    )


def resolve_tau(tau: float | None, checkpoint: str | None) -> tuple[float, str]:
    """τ for a run, and where it came from: explicit τ > tuned value of the checkpoint > DEFAULT_TAU.

    The checkpoint is a local folder or a Hugging Face model id. A hub file is not checked for staleness: it
    comes from the same repository revision as the weights.
    """
    if tau is not None:
        return tau, "explicit"
    if not checkpoint:
        return DEFAULT_TAU, "default"
    path = config_path(checkpoint)
    if path is None:  # not a local folder: a Hugging Face model id
        hub_path = hub_config_path(checkpoint)
        if hub_path is None:
            return DEFAULT_TAU, "default"
        return load_tuned_tau(hub_path), f"tuned, {checkpoint}/{STAGE3_CONFIG} on the Hugging Face Hub"
    if not path.is_file():
        return DEFAULT_TAU, "default"
    if is_stale(path):
        print(
            f"[stage3] {path} is older than the model weights, so it was tuned for another model; "
            f"using the default tau {DEFAULT_TAU}. Re-run scripts/tune_tau.py.",
            file=sys.stderr,
        )
        return DEFAULT_TAU, "default (stale tuned value ignored)"
    return load_tuned_tau(path), f"tuned, {path}"
