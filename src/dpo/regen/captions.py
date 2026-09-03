"""§9.6: one caption track schema, used by both conditions and validated once.

A track is a list of cues. A cue is a pre-set window — ``start_ms``, ``end_ms``
— and the text shown in it. The timings belong to the track, not to whatever
produced the text: §6 generates "text only for fixed cue slots (timings
pre-set)", and the prepared track is authored against the same slots. So the
slots are the shape both conditions share, and the text is the only thing that
differs between them.

That is the whole point of the schema being identical. If the regenerated track
could carry a different number of cues, or cues at different moments, then a
difference in the ART scores between §2 and §7 would be a difference in caption
density or timing as much as in provenance. Validation therefore applies the
*same* rules to both: §9.4's slot count, and the per-slot character and line
caps from the same calibration.

:func:`validate_track` is what a document's prepared and fallback tracks pass
on load. :func:`validate_generated` is what §6 applies to model output, and it
is the same check plus the two failures only generation can produce — empty
text, and text in the wrong language. An authored track cannot be empty,
because the document schema refuses an empty string before it gets here.

Language detection is deliberately crude. It answers one question — did the
model answer in a different script from the study's — and a study running in a
Latin-script language gets no script signal at all, so the check passes rather
than inventing a verdict it cannot support. Detecting *which* Latin-script
language a 96-character caption is in is not something this can do honestly,
and a check that guesses would reject good captions.

Section numbers cite ``docs/v3-regen/spec-behavior.md``.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from dpo.regen.config import Calibration

TRACK_SCHEMA = "dpo.caption-regen-track/v1"

# Scripts that a language tag implies, for the one check language validation can
# make honestly: text in the wrong script is the wrong language. A tag absent
# from here has no script requirement, and its captions are accepted on length
# alone.
SCRIPTS: Mapping[str, str] = {
    "ko": "HANGUL",
    "ja": "CJK",
    "zh": "CJK",
    "ru": "CYRILLIC",
    "el": "GREEK",
    "ar": "ARABIC",
    "he": "HEBREW",
    "th": "THAI",
}


class TrackError(ValueError):
    """A caption track that could not be shown as it stands."""


@dataclass(frozen=True)
class Cue:
    """One slot: when it is on screen, and what it says."""

    index: int
    start_ms: int
    end_ms: int
    text: str

    def record(self) -> dict[str, Any]:
        """The stored form. §2 and §7 log "per-cue text + start/end times"."""
        return {"index": self.index, "start_ms": self.start_ms, "end_ms": self.end_ms, "text": self.text}


def _script_of(text: str) -> set[str]:
    """The scripts the text's letters belong to, by Unicode name prefix."""
    names = set()
    for character in text:
        if not character.isalpha():
            continue
        name = unicodedata.name(character, "")
        for script in ("HANGUL", "CJK", "CYRILLIC", "GREEK", "ARABIC", "HEBREW", "THAI"):
            if name.startswith(script):
                names.add(script)
                break
        else:
            names.add("LATIN" if name.startswith("LATIN") else "OTHER")
    return names


def in_language(text: str, language: str) -> bool:
    """Whether the text could be in the study's language.

    True whenever the language implies no script, so this can only ever reject
    text in a demonstrably different script — never accept-by-guess.
    """
    required = SCRIPTS.get(language.split("-")[0].lower())
    if required is None:
        return True
    return required in _script_of(text)


def cues_of(raw: Sequence[Mapping[str, Any]]) -> tuple[Cue, ...]:
    """Read a stored track back into cues, without validating it."""
    return tuple(
        Cue(
            index=int(cue["index"]),
            start_ms=int(cue["start_ms"]),
            end_ms=int(cue["end_ms"]),
            text=str(cue["text"]),
        )
        for cue in raw
    )


def validate_track(cues: Sequence[Cue], calibration: Calibration, *, path: str) -> None:
    """The rules both tracks obey (§9.4, §9.6). Raises naming the failing slot.

    Timings must not overlap, because two cues on screen at once is a third
    caption the study did not design. They are checked in the order given, and
    the order must be the order of the clock: a track whose slots are shuffled
    would still render, and would render wrong.
    """
    if len(cues) != calibration.cue_slots:
        raise TrackError(
            f"{path}: must have exactly {calibration.cue_slots} cue slots, has {len(cues)}; "
            "the prepared and regenerated tracks are validated against one slot count (§9.4)"
        )
    previous_end = None
    for position, cue in enumerate(cues):
        where = f"{path}[{position}]"
        if cue.index != position:
            raise TrackError(f"{where}.index: must be {position}, is {cue.index}")
        if cue.end_ms <= cue.start_ms:
            raise TrackError(f"{where}: end_ms must be after start_ms")
        if cue.start_ms < 0:
            raise TrackError(f"{where}.start_ms: must not be negative")
        if previous_end is not None and cue.start_ms < previous_end:
            raise TrackError(
                f"{where}: starts at {cue.start_ms} before the previous cue ends at {previous_end}"
            )
        previous_end = cue.end_ms
        _validate_text(cue.text, calibration, where=where)


def _validate_text(text: str, calibration: Calibration, *, where: str) -> None:
    if len(text) > calibration.slot_max_chars:
        raise TrackError(f"{where}.text: must be at most {calibration.slot_max_chars} characters")
    lines = text.splitlines() or [text]
    if len(lines) > calibration.slot_max_lines:
        raise TrackError(f"{where}.text: must be at most {calibration.slot_max_lines} lines")


def validate_generated(cues: Sequence[Cue], calibration: Calibration) -> None:
    """§6's output validation: the track rules, plus empty and wrong-language.

    Raised rather than returned, because the caller's response to any of these
    is the same — fall back to the default track — and a boolean would lose
    which slot failed and why, which is the thing the log needs.
    """
    for position, cue in enumerate(cues):
        where = f"generated[{position}]"
        if not cue.text.strip():
            raise TrackError(f"{where}.text: the model returned nothing for this slot")
        if not in_language(cue.text, calibration.language):
            raise TrackError(f"{where}.text: is not in the study's language ({calibration.language})")
    validate_track(cues, calibration, path="generated")


def retimed(cues: Sequence[Cue], texts: Sequence[str]) -> tuple[Cue, ...]:
    """The slots of ``cues`` carrying ``texts``: §6 writes text, never timings."""
    if len(texts) != len(cues):
        raise TrackError(f"expected {len(cues)} slot texts, got {len(texts)}")
    return tuple(
        Cue(index=cue.index, start_ms=cue.start_ms, end_ms=cue.end_ms, text=text.strip())
        for cue, text in zip(cues, texts, strict=True)
    )


def record_of(cues: Sequence[Cue]) -> list[dict[str, Any]]:
    """The full text of a displayed track, as §2 and §7 log it."""
    return [cue.record() for cue in cues]
