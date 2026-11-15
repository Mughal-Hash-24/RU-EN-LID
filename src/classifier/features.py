"""Stage 1 -> Stage 2 handoff: turn candidate readings into classifier inputs.

Each word gets (a) one evidence tag saying what the lexicons know about it and
(b) optionally its Urdu canonical reading when that differs from the surface
form (bht -> bohat, he -> hai). The surface word is never rewritten; both are
*added* to the word's input embedding in ``candidate_model.py``.
"""

from __future__ import annotations

from ..candidate_expansion.expander import CandidateExpander, TokenCandidates

# id 0 is padding (special tokens and padded positions). Order is part of the checkpoint.
CAND_TAGS = ["PAD", "UNK", "EN", "UR", "AMB", "VAR", "BOTH"]
CAND2ID = {tag: index for index, tag in enumerate(CAND_TAGS)}
MAX_READING_PIECES = 4
CAND_MODES = ("none", "tag", "tag+reading")


def stage1_tag(info: TokenCandidates) -> str:
    """Collapse Stage 1's category + readings into one evidence tag.

    EN / UR  : in exactly one lexicon          (lexicon says: one reading)
    AMB      : in both lexicons                (two readings, context must decide)
    VAR      : attested spelling variant       (Urdu reading via a canonical form)
    BOTH     : variant that is also English    (he -> hai, bar -> baar)
    UNK      : in no lexicon                   (Stage 1 has no evidence)
    """
    if info.category == "both":
        return "BOTH"
    if info.category == "ambiguous":
        return "AMB"
    if info.category == "variant":
        return "VAR"
    if info.english_reading and not info.urdu_canonicals:
        return "EN"
    if info.urdu_canonicals and not info.english_reading:
        return "UR"
    return "UNK"


def reading_of(info: TokenCandidates) -> str | None:
    """The Urdu canonical reading, only when it adds something the surface form lacks."""
    for canonical in info.urdu_canonicals:
        if canonical != info.normalized:
            return canonical
    return None


def word_features(expander: CandidateExpander, words: list[str]) -> tuple[list[str], list[str | None]]:
    """Per-word tags and readings. Uses expand_token so the result is 1:1 with ``words``."""
    infos = [expander.expand_token(word) for word in words]
    return [stage1_tag(info) for info in infos], [reading_of(info) for info in infos]


def piece_features(
    word_ids: list[int | None],
    tags: list[str],
    readings: list[str | None],
    tokenizer,
    use_readings: bool,
) -> dict[str, list]:
    """Spread word-level features over subword pieces (every piece of a word gets them)."""
    cache: dict[str, list[int]] = {}
    cand_ids: list[int] = []
    reading_ids: list[list[int]] = []
    reading_mask: list[list[int]] = []
    pad = tokenizer.pad_token_id
    for word_id in word_ids:
        pieces: list[int] = []
        if word_id is None:
            cand_ids.append(CAND2ID["PAD"])
        else:
            cand_ids.append(CAND2ID[tags[word_id]])
            reading = readings[word_id] if use_readings else None
            if reading:
                if reading not in cache:
                    cache[reading] = tokenizer(reading, add_special_tokens=False)["input_ids"][:MAX_READING_PIECES]
                pieces = cache[reading]
        if use_readings:
            reading_ids.append(pieces + [pad] * (MAX_READING_PIECES - len(pieces)))
            reading_mask.append([1] * len(pieces) + [0] * (MAX_READING_PIECES - len(pieces)))
    features: dict[str, list] = {"cand_ids": cand_ids}
    if use_readings:
        features["reading_ids"] = reading_ids
        features["reading_mask"] = reading_mask
    return features
