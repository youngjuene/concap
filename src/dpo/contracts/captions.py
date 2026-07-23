"""Deterministic caption-contract compliance checks shared by both tracks.

These checks are the automatable subset of the caption contracts: length,
sentence form, and lexical modality screens. They are used to gate candidate
construction, to audit generated captions, and to compute the shared automated
evaluation metrics. They never overrule a human audit; a lexical hit is a flag
for the audit queue, not a verdict about meaning.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from dpo.contracts.study_contract import CaptionContract

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*")
_TERMINAL_RE = re.compile(r"[.!?]")


def caption_words(text: str) -> list[str]:
    """Tokenize a caption into contract words (alphanumeric cores)."""
    return _WORD_RE.findall(unicodedata.normalize("NFC", text))


def word_count(text: str) -> int:
    return len(caption_words(text))


def is_single_sentence(text: str) -> bool:
    """One line, at most one terminal mark, and that mark only at the end."""
    stripped = text.strip()
    if not stripped or "\n" in stripped or "\r" in stripped:
        return False
    marks = list(_TERMINAL_RE.finditer(stripped))
    if len(marks) > 1:
        return False
    return not marks or marks[0].end() == len(stripped)


@dataclass(frozen=True)
class ComplianceReport:
    """Deterministic contract-compliance outcome for one caption."""

    track: str
    text: str
    word_count: int
    empty: bool
    single_sentence: bool
    within_length: bool
    modality_flags: tuple[str, ...]

    @property
    def compliant(self) -> bool:
        return not self.empty and self.single_sentence and self.within_length and not self.modality_flags

    def document(self) -> dict[str, object]:
        return {
            "track": self.track,
            "word_count": self.word_count,
            "empty": self.empty,
            "single_sentence": self.single_sentence,
            "within_length": self.within_length,
            "modality_flags": list(self.modality_flags),
            "compliant": self.compliant,
        }


def lexicon_flags(text: str, lexicon: frozenset[str]) -> tuple[str, ...]:
    """Case-folded whole-word hits of a cross-modal lexicon, sorted and deduplicated."""
    tokens = {token.casefold() for token in caption_words(text)}
    return tuple(sorted(tokens & lexicon))


def check_caption(
    text: str,
    contract: CaptionContract,
    *,
    cross_modal_lexicon: frozenset[str],
) -> ComplianceReport:
    stripped = text.strip()
    count = word_count(stripped)
    return ComplianceReport(
        track=contract.track,
        text=stripped,
        word_count=count,
        empty=not stripped,
        single_sentence=is_single_sentence(stripped),
        within_length=contract.min_words <= count <= contract.max_words,
        modality_flags=lexicon_flags(stripped, cross_modal_lexicon),
    )
