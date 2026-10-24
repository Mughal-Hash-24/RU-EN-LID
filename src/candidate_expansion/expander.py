"""Stage 1 driver: tokenization + candidate reading sets (no destructive rewriting)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .ambiguous import Taxonomy, build_taxonomy
from .lexicons import Lexicons, load_lexicons
from .variants import load_variant_pairs

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019-][A-Za-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Whitespace/punctuation-aware word tokens; punctuation is not labelled."""
    return WORD_RE.findall(text)


@dataclass(frozen=True)
class TokenCandidates:
    surface: str
    normalized: str
    category: str
    english_reading: bool
    urdu_canonicals: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "surface": self.surface,
            "category": self.category,
            "english_reading": self.english_reading,
            "urdu_canonicals": list(self.urdu_canonicals),
        }


class CandidateExpander:
    def __init__(self, lexicons: Lexicons, taxonomy: Taxonomy) -> None:
        self.lexicons = lexicons
        self.taxonomy = taxonomy

    @classmethod
    def from_resources(
        cls,
        resources_dir: str,
        english_path: str | None = None,
        urdu_path: str | None = None,
    ) -> "CandidateExpander":
        lexicons = load_lexicons(resources_dir, english_path=english_path, urdu_path=urdu_path)
        pairs_path = os.path.join(resources_dir, "variant_pairs.tsv")
        pairs = load_variant_pairs(pairs_path) if os.path.exists(pairs_path) else {}
        return cls(lexicons, build_taxonomy(lexicons, pairs))

    def _category(self, word: str) -> str:
        if word in self.taxonomy.both_forms:
            return "both"
        if word in self.taxonomy.ambiguous:
            return "ambiguous"
        if word in self.taxonomy.variant_forms:
            return "variant"
        if word in self.lexicons.english or word in self.lexicons.roman_urdu:
            return "standard"
        return "unknown"

    def expand_token(self, surface: str) -> TokenCandidates:
        word = surface.lower()
        english = word in self.lexicons.english
        canonical = self.taxonomy.canonical.get(word)
        if canonical is None and word in self.lexicons.roman_urdu:
            canonical = word
        urdu = (canonical,) if canonical else ()
        return TokenCandidates(
            surface=surface,
            normalized=word,
            category=self._category(word),
            english_reading=english,
            urdu_canonicals=urdu,
        )

    def expand(self, text: str) -> list[TokenCandidates]:
        return [self.expand_token(token) for token in tokenize(text)]
