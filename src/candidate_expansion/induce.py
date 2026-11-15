"""Grow the Stage-1 lexicons from labelled TRAINING data.

The hand-built seeds cover 177 English and 98 Urdu words, so most test words
reach Stage 2 tagged UNK. This module counts how each word form is labelled in
a training split and sorts it:

  UR   if at least ``purity`` of its UR/EN labels are UR
  EN   if at least ``purity`` of its UR/EN labels are EN
  AMB  if both languages each hold at least ``ambiguous_share`` (written to both lists)
  -    otherwise (too rare, or noisy in-between) - left out

This makes "ambiguous" an observed property of the corpus instead of a hand-picked
list, which is also what the paper needs to report as its ambiguous-word set.

Leakage: build the lexicon from the training split only. A test word that the
lexicon knows only because it was counted in test would inflate every Stage-1
number. ``scripts/build_lexicon.py`` reports how many test words the induced
lexicon covers, so the effect can be stated.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

LANGUAGES = ("UR", "EN")


@dataclass
class InducedLexicon:
    english: set[str] = field(default_factory=set)
    roman_urdu: set[str] = field(default_factory=set)
    ambiguous: set[str] = field(default_factory=set)
    counts: dict[str, Counter] = field(default_factory=dict)
    skipped_rare: int = 0
    skipped_mixed: int = 0


def induce(
    sentences,
    min_count: int = 3,
    purity: float = 0.9,
    ambiguous_share: float = 0.2,
) -> InducedLexicon:
    """sentences: objects with ``.words`` and ``.labels`` (see classifier/dataset.py)."""
    if not 0.5 < purity <= 1.0:
        raise ValueError("purity must be in (0.5, 1]")
    if not 0.0 < ambiguous_share <= 0.5:
        raise ValueError("ambiguous_share must be in (0, 0.5]")
    counts: dict[str, Counter] = defaultdict(Counter)
    for sentence in sentences:
        for word, label in zip(sentence.words, sentence.labels, strict=True):
            if label in LANGUAGES and any(ch.isalpha() for ch in word):
                counts[word.lower()][label] += 1

    result = InducedLexicon(counts=dict(counts))
    for word, label_counts in counts.items():
        total = label_counts["UR"] + label_counts["EN"]
        if total < min_count:
            result.skipped_rare += 1
            continue
        urdu_share = label_counts["UR"] / total
        if urdu_share >= purity:
            result.roman_urdu.add(word)
        elif 1.0 - urdu_share >= purity:
            result.english.add(word)
        elif min(urdu_share, 1.0 - urdu_share) >= ambiguous_share:
            result.ambiguous.add(word)
        else:
            result.skipped_mixed += 1
    return result
