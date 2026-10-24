"""Stage-2 augmentation: synthetic spelling variants plus ambiguity oversampling.

The design mirrors proposal §7.2: spelling variation is synthesizable
(rule-based re-spellings), whereas ambiguity is lexical and is handled by
oversampling real sentences that contain known ambiguous forms.
"""

from __future__ import annotations

import random
import re

from .dataset import Sentence

RULES: tuple[tuple[str, str], ...] = (
    ("aa", "a"),
    ("ee", "i"),
    ("oo", "u"),
    ("bh", "b"),
    ("dh", "d"),
    ("ph", "f"),
    ("th", "t"),
    ("gh", "g"),
    ("kh", "k"),
    ("ch", "c"),
    ("q", "k"),
    ("x", "ks"),
    ("c", "k"),
    ("y", "i"),
    ("w", "v"),
)

VOWELS = set("aeiou")


def _drop_internal_vowels(word: str) -> str:
    if len(word) <= 3:
        return word
    return word[0] + "".join(char for char in word[1:] if char not in VOWELS)


def synthetic_variants(word: str, limit: int = 4) -> list[str]:
    """Deterministic rule-based re-spellings of a Roman Urdu word."""
    base = word.lower()
    variants: set[str] = set()
    for source, target in RULES:
        if source in base:
            variants.add(base.replace(source, target))
    variants.add(_drop_internal_vowels(base))
    variants.add(re.sub(r"(.)\1+", r"\1", base))
    variants.discard(base)
    return sorted(variants)[:limit]


def augment_sentence(
    words: list[str],
    labels: list[str],
    urdu_label: str = "UR",
    probability: float = 0.25,
    rng: random.Random | None = None,
) -> tuple[list[str], list[str]]:
    """Randomly replace Urdu tokens with rule-based spelling variants."""
    generator = rng or random.Random(0)
    new_words: list[str] = []
    for word, label in zip(words, labels):
        if label == urdu_label:
            options = synthetic_variants(word)
            if options and generator.random() < probability:
                new_words.append(generator.choice(options))
                continue
        new_words.append(word)
    return new_words, list(labels)


def oversample_ambiguous(
    sentences: list[Sentence],
    ambiguous_forms: set[str],
    factor: int = 3,
) -> list[Sentence]:
    """Duplicate sentences containing ambiguous forms (up to ``factor`` copies)."""
    augmented: list[Sentence] = []
    for sentence in sentences:
        augmented.append(sentence)
        if any(word.lower() in ambiguous_forms for word in sentence.words):
            for copy in range(1, factor):
                augmented.append(
                    Sentence(
                        sentence_id=f"{sentence.sentence_id}#{copy}",
                        words=list(sentence.words),
                        labels=list(sentence.labels),
                        categories=list(sentence.categories) if sentence.categories else None,
                    )
                )
    return augmented
