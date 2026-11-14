"""Stage 3 context decoder: resolve deferred words using their neighbours.

The lexicon rules in ``rules.py`` cannot help an ambiguous word (to, main, is):
both lexicons accept it. What can help is the sentence. Language switches are
rare between neighbouring words, so a deferred word should usually take the
language of the confident words around it unless Stage 2 is clearly against it.

This is a two-state Viterbi decode over the "current language" (UR or EN):

  cost(word gets label y) = -log p2(y)          Stage 2's probability for y
  cost(UR <-> EN switch)  = penalty              charged once per switch
  OTH                     = keeps the current language (names, numbers, punctuation)

Words Stage 2 was confident about (>= tau), and words a lexicon rule already
fixed, are pinned to their label, so the decoder can only change deferred
words. With penalty = 0 every deferred word keeps Stage 2's argmax, i.e. the
decoder is off. The penalty is a single number tuned on validation
(scripts/tune_stage3.py), never on test.
"""

from __future__ import annotations

import math

LANGUAGES = ("UR", "EN")
LABELS = ("UR", "EN", "OTH")
_FLOOR = 1e-6
_START = "START"


def _cost(probs: dict[str, float], label: str) -> float:
    return -math.log(max(probs.get(label, 0.0), _FLOOR))


def decode(
    probabilities: list[dict[str, float]],
    pinned: list[str | None],
    penalty: float,
) -> list[str]:
    """Best label per word.

    probabilities: Stage 2's distribution over UR/EN/OTH for each word.
    pinned:        a label for words that must not change, None for deferred words.
    penalty:       cost of one UR<->EN switch (>= 0).
    """
    if len(probabilities) != len(pinned):
        raise ValueError("one probability dict and one pin per word")
    if not probabilities:
        return []
    states = (_START,) + LANGUAGES  # language of the last UR/EN word so far
    best = {state: (0.0 if state == _START else math.inf) for state in states}
    back: list[dict[str, tuple[str, str]]] = []  # per word: new_state -> (old_state, label)

    for probs, pin in zip(probabilities, pinned):
        options = (pin,) if pin is not None else LABELS
        step_best = {state: math.inf for state in states}
        step_back: dict[str, tuple[str, str]] = {}
        for old, old_cost in best.items():
            if old_cost == math.inf:
                continue
            for label in options:
                emit = 0.0 if pin is not None else _cost(probs, label)
                if label == "OTH":
                    new, switch = old, 0.0
                else:
                    new = label
                    switch = penalty if old in LANGUAGES and old != label else 0.0
                total = old_cost + emit + switch
                if total < step_best[new]:
                    step_best[new] = total
                    step_back[new] = (old, label)
        best = step_best
        back.append(step_back)

    state = min(best, key=best.get)
    labels: list[str] = []
    for step_back in reversed(back):
        state, label = step_back[state]
        labels.append(label)
    return labels[::-1]
