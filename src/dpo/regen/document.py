"""The session document: two segments of one scene, and everything §10 prepares.

One document describes one study. It carries the frozen configuration artifact,
and the two segments — A and B — that a participant watches one of under each
condition. Which participant gets which is not in here: that is
:mod:`dpo.regen.assignment`, computed per person from their sequence number, so
one document serves every participant and no document encodes a condition.

Each segment carries what §10 says is prepared in advance: the clip, the
five-second still, the segmentation masks with their object labels, the
separated stems with their waveforms and labels, the prepared caption track and
the default fallback track. Both tracks are validated against the same
calibration on load, which is what §9.6's "identical schema" amounts to in
practice — a document whose prepared track has five cues and whose fallback has
four is refused before a participant ever sees either.

Both segments carry a prepared *and* a fallback track because either segment
can be either condition. A document that only prepared segment A would work for
even-numbered participants and fail for odd ones, and it would fail at the
moment a participant is sitting in front of it.

What is *not* here: anything computed. The matched objects of §4, the listening
times of §5, the regenerated track of §6 — all of those are results, and they
live in the log. The document is what the study brings; the log is what the
session produces.

Validation names the offending path, because a researcher authors this by hand
and a bare "invalid document" costs them a hunt.

Section numbers cite ``docs/v3-regen/spec-behavior.md``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dpo.regen.assignment import SEGMENTS
from dpo.regen.captions import Cue, TrackError, cues_of, validate_track
from dpo.regen.config import Configuration, load_configuration
from dpo.regen.points import MaskObject

REGEN_SCHEMA = "dpo.caption-regen/v1"
ID_RE = re.compile(r"[A-Za-z0-9_-]+\Z")
COLOUR_RE = re.compile(r"#[0-9A-Fa-f]{6}\Z")
# §5 draws the lanes stacked over one ten-second timeline. Past this the lanes
# are thinner than a playback control and the waveform stops being readable.
MAX_STEMS = 8
# Enough envelope to draw a ten-second lane at kiosk width without the browser
# resampling a longer array down to the same picture.
MIN_WAVEFORM = 64
MAX_WAVEFORM = 4096
TRACKS = ("prepared_track", "fallback_track")


class RegenDocumentError(ValueError):
    """The document violates the schema; the message names the path."""


def _fail(path: str, message: str) -> RegenDocumentError:
    return RegenDocumentError(f"{path}: {message}")


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(path, "must be an object")
    return value


def _sequence(value: object, path: str, *, minimum: int = 0, maximum: int | None = None) -> Sequence[Any]:
    if not isinstance(value, list):
        raise _fail(path, "must be a list")
    if len(value) < minimum:
        raise _fail(path, f"must have at least {minimum} entr{'y' if minimum == 1 else 'ies'}")
    if maximum is not None and len(value) > maximum:
        raise _fail(path, f"must have at most {maximum} entries")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(path, "must be a non-empty string")
    return value


def _identifier(value: object, path: str) -> str:
    text = _string(value, path)
    if not ID_RE.fullmatch(text):
        raise _fail(path, "must match [A-Za-z0-9_-]+")
    return text


def _relative(value: object, path: str) -> str:
    """A media reference, resolved under ``--media-dir`` and never above it."""
    text = _string(value, path)
    candidate = Path(text)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise _fail(path, "must be a relative path inside the media directory")
    return text


def _integer(value: object, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(path, "must be an integer")
    if minimum is not None and value < minimum:
        raise _fail(path, f"must be at least {minimum}")
    return value


def _number(value: object, path: str, *, low: float = 0.0, high: float = 1.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _fail(path, "must be a number")
    number = float(value)
    if not low <= number <= high:
        raise _fail(path, f"must lie in [{low}, {high}]")
    return number


def _validate_object(raw: object, path: str) -> str:
    """One segmented object of the still: what §4 matches a point against."""
    entry = _mapping(raw, path)
    object_id = _identifier(entry.get("id"), f"{path}.id")
    _string(entry.get("label"), f"{path}.label")
    _relative(entry.get("mask"), f"{path}.mask")
    return object_id


def _validate_stem(raw: object, path: str) -> str:
    """One lane of §5: its audio, its envelope, its colour and its level."""
    entry = _mapping(raw, path)
    stem_id = _identifier(entry.get("id"), f"{path}.id")
    _string(entry.get("label"), f"{path}.label")
    _relative(entry.get("audio"), f"{path}.audio")
    colour = _string(entry.get("colour"), f"{path}.colour")
    if not COLOUR_RE.fullmatch(colour):
        raise _fail(f"{path}.colour", "must be #rrggbb")
    # §5 requires the stems normalised against the original mix. The gain that
    # achieves it is measured when the stems are cut and recorded here, so the
    # page applies one number rather than deciding a level per lane at play
    # time — which would make loudness a property of the order lanes were
    # clicked in.
    _number(entry.get("gain"), f"{path}.gain", low=0.0, high=4.0)
    waveform = _sequence(
        entry.get("waveform"), f"{path}.waveform", minimum=MIN_WAVEFORM, maximum=MAX_WAVEFORM
    )
    for index, sample in enumerate(waveform):
        _number(sample, f"{path}.waveform[{index}]")
    return stem_id


def _validate_track(raw: object, path: str, configuration: Configuration) -> None:
    entries = _sequence(raw, path, minimum=1)
    for index, entry in enumerate(entries):
        cue = _mapping(entry, f"{path}[{index}]")
        _integer(cue.get("index"), f"{path}[{index}].index", minimum=0)
        _integer(cue.get("start_ms"), f"{path}[{index}].start_ms", minimum=0)
        _integer(cue.get("end_ms"), f"{path}[{index}].end_ms", minimum=1)
        _string(cue.get("text"), f"{path}[{index}].text")
    try:
        validate_track(cues_of(entries), configuration.calibration, path=path)
    except TrackError as exc:
        raise RegenDocumentError(str(exc)) from exc


def _validate_segment(raw: object, path: str, name: str, configuration: Configuration) -> str:
    segment = _mapping(raw, path)
    if segment.get("segment") != name:
        raise _fail(f"{path}.segment", f"must be {name!r}, to match the key it is filed under")
    clip_id = _identifier(segment.get("clip_id"), f"{path}.clip_id")
    _relative(segment.get("video"), f"{path}.video")
    _relative(segment.get("still"), f"{path}.still")
    duration = _integer(segment.get("duration_ms"), f"{path}.duration_ms", minimum=1)

    objects = _sequence(segment.get("objects"), f"{path}.objects", minimum=1)
    seen_objects: set[str] = set()
    for index, entry in enumerate(objects):
        object_id = _validate_object(entry, f"{path}.objects[{index}]")
        if object_id in seen_objects:
            raise _fail(f"{path}.objects[{index}].id", f"{object_id!r} is declared twice")
        seen_objects.add(object_id)

    stems = _sequence(segment.get("stems"), f"{path}.stems", minimum=1, maximum=MAX_STEMS)
    seen_stems: set[str] = set()
    for index, entry in enumerate(stems):
        stem_id = _validate_stem(entry, f"{path}.stems[{index}]")
        if stem_id in seen_stems:
            raise _fail(f"{path}.stems[{index}].id", f"{stem_id!r} is declared twice")
        seen_stems.add(stem_id)

    for track in TRACKS:
        _validate_track(segment.get(track), f"{path}.{track}", configuration)
        last = cues_of(segment[track])[-1]
        if last.end_ms > duration:
            raise _fail(f"{path}.{track}", f"ends at {last.end_ms}ms, past the segment's {duration}ms")
    return clip_id


def validate_regen_document(document: Mapping[str, Any]) -> None:
    """Refuse a document a session could not be run from. Raises on the path."""
    root = _mapping(document, "document")
    if root.get("schema") != REGEN_SCHEMA:
        raise _fail("schema", f"must be {REGEN_SCHEMA!r}")
    _identifier(root.get("session_id"), "session_id")
    configuration = configuration_of(root)
    segments = _mapping(root.get("segments"), "segments")
    if set(segments) != set(SEGMENTS):
        raise _fail("segments", f"must be exactly {list(SEGMENTS)}; both are needed whichever way §1 assigns")
    clips = {}
    for name in SEGMENTS:
        clips[name] = _validate_segment(segments[name], f"segments.{name}", name, configuration)
    if clips["A"] == clips["B"]:
        raise _fail(
            "segments.B.clip_id",
            f"is the same clip as segment A ({clips['A']!r}); the two viewings must be different footage",
        )


def configuration_of(document: Mapping[str, Any]) -> Configuration:
    """The frozen configuration this document is served under."""
    raw = document.get("config")
    if not isinstance(raw, Mapping):
        raise _fail("config", "must be the configuration artifact")
    try:
        return load_configuration(raw)
    except ValueError as exc:
        raise _fail("config", str(exc)) from exc


def load_regen_document(path: Path | str) -> dict[str, Any]:
    """Read and validate a document from disk."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _fail(str(path), f"is not JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise _fail(str(path), "must be an object")
    validate_regen_document(raw)
    return raw


def segment_of(document: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    segments = document.get("segments")
    if not isinstance(segments, Mapping) or name not in segments:
        raise _fail("segments", f"has no segment {name!r}")
    return _mapping(segments[name], f"segments.{name}")


def objects_of(document: Mapping[str, Any], media_dir: Path, name: str) -> tuple[MaskObject, ...]:
    """The masks §4 matches points against, resolved under the media directory."""
    segment = segment_of(document, name)
    return tuple(
        MaskObject(id=str(entry["id"]), label=str(entry["label"]), path=Path(media_dir) / str(entry["mask"]))
        for entry in segment["objects"]
    )


def track_of(document: Mapping[str, Any], name: str, which: str) -> tuple[Cue, ...]:
    """A segment's prepared or fallback track, as cues."""
    if which not in TRACKS:
        raise _fail("track", f"must be one of {list(TRACKS)}")
    return cues_of(segment_of(document, name)[which])


def participant_document(document: Mapping[str, Any], name: str) -> dict[str, Any]:
    """What the browser is given for one segment.

    Mask files and the fallback track stay on the server. The masks because §4
    matches once, on submission, and a browser holding the masks could match
    continuously — the thing that rule exists to prevent. The fallback track
    because a page that already has it could show it before §6 has decided
    whether it is needed, and a participant would be reading the default policy
    under the label of their own regeneration.
    """
    segment = segment_of(document, name)
    return {
        "segment": name,
        "clip_id": segment["clip_id"],
        "duration_ms": segment["duration_ms"],
        "objects": [{"id": entry["id"]} for entry in segment["objects"]],
        "stems": [
            {
                "id": stem["id"],
                "label": stem["label"],
                "colour": stem["colour"],
                "gain": stem["gain"],
                "waveform": list(stem["waveform"]),
            }
            for stem in segment["stems"]
        ],
    }
