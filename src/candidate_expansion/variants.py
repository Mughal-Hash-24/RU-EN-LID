"""UrduPhone-style phonetic encoding and variant clustering (Stage 1).

Simplified after the UrduPhone scheme (Rafae et al., Natural Language
Engineering): keep the first letter, drop vowels and duplicate letters, encode
the remaining consonants and pad to six slots. Variants are then found by
grouping spellings with the same code and clustering them by string similarity.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

VOWELS = set("aeiouy")


def urduphone(word: str) -> str:
    letters = re.sub(r"[^a-z]", "", word.lower())
    if not letters:
        return ""
    code = [letters[0].upper()]
    previous = ""
    for char in letters[1:]:
        if char == previous:
            continue
        previous = char
        if char in VOWELS:
            continue
        code.append(char)
        if len(code) == 6:
            break
    while len(code) < 6:
        code.append("0")
    return "".join(code)


def similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio()


def cluster_variants(words, threshold: float = 0.6) -> list[list[str]]:
    """Group spelling variants sharing a phonetic code and string similarity."""
    buckets: dict[str, list[str]] = {}
    for word in sorted({w.lower() for w in words}):
        buckets.setdefault(urduphone(word), []).append(word)
    clusters: list[list[str]] = []
    for members in buckets.values():
        if len(members) < 2:
            continue
        groups: list[list[str]] = []
        for word in members:
            for group in groups:
                if any(similarity(word, member) >= threshold for member in group):
                    group.append(word)
                    break
            else:
                groups.append([word])
        clusters.extend(group for group in groups if len(group) > 1)
    return clusters


def canonical_of(cluster: list[str]) -> str:
    """Longest spelling wins; ties resolve alphabetically (baar > bar, hai > he, main = main)."""
    return min(cluster, key=lambda word: (-len(word), word))


def load_dakshina_lexicon(path: str) -> list[tuple[str, str]]:
    """Read a Dakshina-style lexicon (``romanized<native_script_word`` per line)."""
    pairs: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or "<" not in line:
                continue
            romanized, native = line.split("<", 1)
            pairs.append((romanized.strip().lower(), native.strip()))
    return pairs


def load_variant_pairs(path: str) -> dict[str, list[str]]:
    """Read attested variant pairs (``canonical<TAB>variant`` per line, ``#`` starts a comment).

    A variant may belong to one canonical only: two canonicals would make its Urdu reading
    depend on file order, so that is rejected with the offending line number.
    """
    pairs: dict[str, list[str]] = {}
    owner: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            parts = line.split("#", 1)[0].lower().split()
            if len(parts) < 2:
                continue
            canonical, variant = parts[0], parts[1]
            if owner.setdefault(variant, canonical) != canonical:
                raise ValueError(
                    f"{path}:{number}: variant {variant!r} already belongs to {owner[variant]!r}, not {canonical!r}"
                )
            if variant not in pairs.setdefault(canonical, []):
                pairs[canonical].append(variant)
    return pairs

