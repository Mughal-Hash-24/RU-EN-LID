"""Lexicon loading and validation for Stage 1.

The hard guard list below follows the proposal's audit warning: naive English
word lists contain Roman Urdu function words (hai, ko, ki, se, ...), so any such
form is removed from the English side before the ambiguous-form intersection is
computed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

EXCLUSIVELY_ROMAN_URDU = frozenset(
    """
    aap ab abhi acha agar ajj ammi aur baat barish bhai bhaiya bhookh bht bohat
    bachay chaar chai chalo cheez didi din dost ek ghar haan hai hain hamare ho
    hoga hoon houn hua hue hui hum jana jao ji ka kal kar karo karna karte karta
    kaha kuch kya kyun ke ki ko lagi larka larki lekin log magar mein mujhe nahi
    nhi pani paanch paisa par pata phir raat raha rahe rahi sab samajh se teen
    theek thi tha tum usne waqt warna yaar ye
    """.split()
)


def read_wordlist(path: str) -> set[str]:
    """Read a word list, accepting one-word-per-line or whitespace-separated files."""
    words: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            for token in line.replace(",", " ").split():
                token = token.strip().lower()
                if token and not token.startswith("#"):
                    words.add(token)
    return words


@dataclass(frozen=True)
class Lexicons:
    english: frozenset[str]
    roman_urdu: frozenset[str]

    @property
    def ambiguous(self) -> frozenset[str]:
        return frozenset(self.english & self.roman_urdu)


def load_lexicons(
    resources_dir: str,
    english_path: str | None = None,
    urdu_path: str | None = None,
    extra_english: set[str] | None = None,
    extra_urdu: set[str] | None = None,
) -> Lexicons:
    en_path = english_path or os.path.join(resources_dir, "english_seed.txt")
    ur_path = urdu_path or os.path.join(resources_dir, "roman_urdu_seed.txt")
    english = read_wordlist(en_path) - EXCLUSIVELY_ROMAN_URDU
    roman_urdu = read_wordlist(ur_path)
    if extra_english:
        english |= {w.strip().lower() for w in extra_english if w.strip()}
    if extra_urdu:
        roman_urdu |= {w.strip().lower() for w in extra_urdu if w.strip()}
    return Lexicons(english=frozenset(english), roman_urdu=frozenset(roman_urdu))
