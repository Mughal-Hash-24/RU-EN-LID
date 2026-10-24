"""The four-category token taxonomy: standard / ambiguous / variant / both."""

from __future__ import annotations

from dataclasses import dataclass

from .lexicons import Lexicons


@dataclass(frozen=True)
class Taxonomy:
    ambiguous: frozenset[str]
    variant_forms: frozenset[str]
    both_forms: frozenset[str]
    canonical: dict[str, str]


def build_taxonomy(lexicons: Lexicons, variant_pairs: dict[str, list[str]]) -> Taxonomy:
    canonical: dict[str, str] = {}
    variant_forms: set[str] = set()
    both_forms: set[str] = set()
    for representative, variants in variant_pairs.items():
        representative = representative.lower()
        canonical[representative] = representative
        for variant in variants:
            variant = variant.lower()
            canonical[variant] = representative
            variant_forms.add(variant)
            if variant in lexicons.english:
                both_forms.add(variant)
    return Taxonomy(
        ambiguous=lexicons.ambiguous,
        variant_forms=frozenset(variant_forms),
        both_forms=frozenset(both_forms),
        canonical=canonical,
    )
