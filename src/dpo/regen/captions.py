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
    """One slot: when it is on screen, and what it says, in each language offered.

    ``text`` is keyed by language tag. A study offering one language has one
    key; the slot, its timings and its position are the same either way, which
    is what §9.6 asks for — the schema does not change with the language, only
    which key is read.
    """

    index: int
    start_ms: int
    end_ms: int
    text: Mapping[str, str]

    def say(self, language: str) -> str:
        """What this cue reads as on screen, in one language."""
        try:
            return self.text[language]
        except KeyError:
            raise TrackError(
                f"cue {self.index} has no text in {language!r}; it has {sorted(self.text)}"
            ) from None

    def record(self, language: str) -> dict[str, Any]:
        """The stored form: what was displayed, and which language it was in.

        §2 and §7 log "per-cue text + start/end times". The text is the text a
        participant actually read, so a row is legible without the document —
        and ``language`` says which, because a study offering two would
        otherwise leave that to be inferred from the characters.
        """
        return {
            "index": self.index,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "text": self.say(language),
            "language": language,
        }


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


def text_of(raw: object, default: str) -> Mapping[str, str]:
    """A cue's text, however the document wrote it.

    A study in one language may write ``"text": "..."`` and mean that language;
    one offering two writes ``{"en": "...", "ko": "..."}``. Both arrive here as
    a mapping, so nothing downstream has to know which shape was used.
    """
    if isinstance(raw, Mapping):
        return {str(tag): str(value) for tag, value in raw.items()}
    return {default: str(raw)}


def cues_of(raw: Sequence[Mapping[str, Any]], default: str = "en") -> tuple[Cue, ...]:
    """Read a stored track back into cues, without validating it."""
    return tuple(
        Cue(
            index=int(cue["index"]),
            start_ms=int(cue["start_ms"]),
            end_ms=int(cue["end_ms"]),
            text=text_of(cue["text"], default),
        )
        for cue in raw
    )


def _validate_timings(cues: Sequence[Cue], calibration: Calibration, *, path: str) -> None:
    """The shape both tracks obey (§9.4, §9.6). Raises naming the failing slot.

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


def validate_track(cues: Sequence[Cue], calibration: Calibration, *, path: str) -> None:
    """An authored track: the shape, and the text in every language offered.

    Every language, because the participant picks one before §2 and a slot
    missing the language they picked is a gap they would read, discovered with
    them sitting in front of it rather than at ``dpo regen validate``.
    """
    _validate_timings(cues, calibration, path=path)
    for position, cue in enumerate(cues):
        where = f"{path}[{position}]"
        missing = [tag for tag in calibration.languages if tag not in cue.text]
        if missing:
            raise TrackError(
                f"{where}.text: the study offers {list(calibration.languages)} and this slot "
                f"has no text in {missing}; a participant reading that language would find a gap"
            )
        for tag in calibration.languages:
            _validate_text(cue.text[tag], calibration, where=f"{where}.text[{tag}]")


def _validate_text(text: str, calibration: Calibration, *, where: str) -> None:
    if len(text) > calibration.slot_max_chars:
        raise TrackError(f"{where}.text: must be at most {calibration.slot_max_chars} characters")
    lines = text.splitlines() or [text]
    if len(lines) > calibration.slot_max_lines:
        raise TrackError(f"{where}.text: must be at most {calibration.slot_max_lines} lines")


def validate_generated(cues: Sequence[Cue], calibration: Calibration, language: str) -> None:
    """§6's output validation: the track rules, plus empty and wrong-language.

    Checked in the one language the track was generated in — the language the
    participant is reading — rather than in every language the study offers: §6
    writes for the session in front of it, not for the study's whole menu.

    Raised rather than returned, because the caller's response to any of these
    is the same — fall back to the default track — and a boolean would lose
    which slot failed and why, which is the thing the log needs.
    """
    for position, cue in enumerate(cues):
        where = f"generated[{position}]"
        written = cue.text.get(language, "")
        if not written.strip():
            raise TrackError(f"{where}.text: the model returned nothing for this slot")
        if not in_language(written, language):
            raise TrackError(f"{where}.text: is not in the language on screen ({language})")
        _validate_text(written, calibration, where=where)
    _validate_timings(cues, calibration, path="generated")


def retimed(cues: Sequence[Cue], texts: Sequence[str], language: str) -> tuple[Cue, ...]:
    """The slots of ``cues`` carrying ``texts``: §6 writes text, never timings.

    The result speaks one language — the one the session is being read in. A
    regenerated track is written for the participant in front of it and there
    is no second reading to offer.
    """
    if len(texts) != len(cues):
        raise TrackError(f"expected {len(cues)} slot texts, got {len(texts)}")
    return tuple(
        Cue(
            index=cue.index,
            start_ms=cue.start_ms,
            end_ms=cue.end_ms,
            text={language: text.strip()},
        )
        for cue, text in zip(cues, texts, strict=True)
    )


def record_of(cues: Sequence[Cue], language: str) -> list[dict[str, Any]]:
    """The full text of a displayed track, as §2 and §7 log it."""
    return [cue.record(language) for cue in cues]
