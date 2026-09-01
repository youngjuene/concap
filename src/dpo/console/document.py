"""The console session document: one clip, its shots, and what each shot measures.

The document is the configuration. It carries the frozen artifact (§10), the
clip's shots with their measured sources, and the two captions each shot needs
besides the participant's own: the raw model prose the first viewing plays, and
the default-policy caption the check compares against.

Three objects per shot, and they are not interchangeable (§11). The raw caption
is the audio-language model's own prose, and it is *not* a point in the
console's reachable space — a participant who preferred it cannot steer back to
it, and that frustration is what the probe is for. The default-policy caption
is what the fitted policy would produce, which is the thing the study is trying
to beat. The committed caption is the participant's. Together they form the
three-object gradient the analysis reads.

What is *not* here: ``v_g``, ``ŵ_g``, the regimes, the bands, the registers,
and every §9 measurement. All of those are computed from ``r_g``, ``p_g``,
``c_g`` and ``e_g`` under the configuration in force, so a document cannot
disagree with the calibration it was served under. The browser gets less still
— see :func:`participant_document`.

Validation names the offending path, because a researcher authors this by hand
and a bare "invalid document" costs them a hunt.

Section numbers cite ``spec-system.md`` except §3.2, §4.1 and
§4.4, which are ``spec-uiux.md``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dpo.caption.writer import CAPTION_MAX_CHARS
from dpo.console.config import Configuration, load_configuration
from dpo.console.quantities import GRAINS, Field, Source

CONSOLE_SCHEMA = "dpo.caption-console/v1"
ID_RE = re.compile(r"[A-Za-z0-9_-]+\Z")
# §4.1 renders sources as a token strip, and §7's regime count grows as the
# square of the source count. Beyond this a strip wraps to three lines and the
# crossfader loses every segment worth aiming at.
MAX_SOURCES = 8


class ConsoleDocumentError(ValueError):
    """The document violates the schema; the message names the path."""


def _fail(path: str, message: str) -> ConsoleDocumentError:
    return ConsoleDocumentError(f"{path}: {message}")


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(path, "must be an object")
    return value


def _sequence(value: object, path: str, *, minimum: int = 0) -> Sequence[Any]:
    if not isinstance(value, list):
        raise _fail(path, "must be a list")
    if len(value) < minimum:
        raise _fail(path, f"must have at least {minimum} entr{'y' if minimum == 1 else 'ies'}")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise _fail(path, "must be a string")
    if not value.strip():
        raise _fail(path, "must not be empty")
    return value


def _identifier(value: object, path: str) -> str:
    text = _string(value, path)
    if not ID_RE.fullmatch(text):
        raise _fail(path, "must match [A-Za-z0-9_-]+")
    return text


def _integer(value: object, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(path, "must be an integer")
    if minimum is not None and value < minimum:
        raise _fail(path, f"must be at least {minimum}")
    return value


def _number(value: object, path: str, *, low: float = 0.0, high: float | None = 1.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _fail(path, "must be a number")
    number = float(value)
    if number < low or (high is not None and number > high):
        bound = f"[{low}, {high}]" if high is not None else f"at least {low}"
        raise _fail(path, f"must lie in {bound}")
    return number


def _caption(value: object, path: str) -> str:
    """Prose placed or returned verbatim, so it must fit the caption band."""
    text = _string(value, path)
    if len(text) > CAPTION_MAX_CHARS:
        raise _fail(path, f"must be at most {CAPTION_MAX_CHARS} characters; the band reserves two lines")
    return text


def _validate_source(raw: object, path: str) -> str:
    source = _mapping(raw, path)
    source_id = _identifier(source.get("id"), f"{path}.id")
    labels = _sequence(source.get("labels"), f"{path}.labels", minimum=1)
    for index, label in enumerate(labels):
        _string(label, f"{path}.labels[{index}]")
    _string(source.get("prose"), f"{path}.prose")
    _number(source.get("share"), f"{path}.share")
    _number(source.get("presence"), f"{path}.presence")
    # c_g and e_g are separate because w_g = c_g · e_g separates detection from
    # prominence (§5.1). A document that carries only one of them is a document
    # whose audio axis is detectional, and the schema will not hide that by
    # defaulting the other to one.
    _number(source.get("confidence"), f"{path}.confidence")
    _number(source.get("energy"), f"{path}.energy", high=None)
    return source_id


def _validate_shot(raw: object, path: str, expected_start: int) -> tuple[str, int]:
    shot = _mapping(raw, path)
    shot_id = _identifier(shot.get("shot_id"), f"{path}.shot_id")
    start = _integer(shot.get("start_ms"), f"{path}.start_ms", minimum=0)
    end = _integer(shot.get("end_ms"), f"{path}.end_ms", minimum=start + 1)
    if start != expected_start:
        raise _fail(f"{path}.start_ms", f"must be {expected_start}; shots are contiguous from zero")
    _caption(shot.get("raw_caption"), f"{path}.raw_caption")
    _caption(shot.get("default_caption"), f"{path}.default_caption")
    sources = _sequence(shot.get("sources"), f"{path}.sources", minimum=1)
    if len(sources) > MAX_SOURCES:
        raise _fail(f"{path}.sources", f"must carry at most {MAX_SOURCES} sources")
    seen: set[str] = set()
    for index, raw_source in enumerate(sources):
        source_id = _validate_source(raw_source, f"{path}.sources[{index}]")
        if source_id in seen:
            raise _fail(f"{path}.sources[{index}].id", f"duplicates {source_id!r}")
        seen.add(source_id)
    return shot_id, end


def _validate_opening(raw: object, path: str, source_ids: Sequence[str]) -> None:
    """The start state, randomized per trial and recorded as a covariate (§11)."""
    opening = _mapping(raw, path)
    grain = _string(opening.get("grain"), f"{path}.grain")
    if grain not in GRAINS:
        raise _fail(f"{path}.grain", f"must be one of {list(GRAINS)}")
    _number(opening.get("alpha"), f"{path}.alpha")
    admitted = opening.get("admitted")
    if admitted is None:
        return
    known = set(source_ids)
    for index, source_id in enumerate(_sequence(admitted, f"{path}.admitted", minimum=1)):
        if source_id not in known:
            raise _fail(f"{path}.admitted[{index}]", f"names unknown source {source_id!r}")


def _validate_clip(raw: object, path: str) -> str:
    clip = _mapping(raw, path)
    clip_id = _identifier(clip.get("clip_id"), f"{path}.clip_id")
    shots = _sequence(clip.get("shots"), f"{path}.shots", minimum=1)
    expected_start = 0
    seen: set[str] = set()
    all_sources: list[str] = []
    for index, raw_shot in enumerate(shots):
        shot_id, expected_start = _validate_shot(raw_shot, f"{path}.shots[{index}]", expected_start)
        if shot_id in seen:
            raise _fail(f"{path}.shots[{index}].shot_id", f"duplicates {shot_id!r}")
        seen.add(shot_id)
        all_sources.extend(str(source["id"]) for source in raw_shot["sources"])
    _validate_opening(clip.get("opening"), f"{path}.opening", all_sources)
    return clip_id


def validate_console_document(document: Mapping[str, Any]) -> None:
    """Raise :class:`ConsoleDocumentError` naming the first offending path."""
    if document.get("schema") != CONSOLE_SCHEMA:
        raise _fail("schema", f"must be {CONSOLE_SCHEMA!r}")
    _identifier(document.get("session_id"), "session_id")
    try:
        load_configuration(_mapping(document.get("config"), "config"))
    except ValueError as exc:
        raise _fail("config", str(exc)) from exc
    clips = _sequence(document.get("clips"), "clips", minimum=1)
    seen: set[str] = set()
    for index, clip in enumerate(clips):
        clip_id = _validate_clip(clip, f"clips[{index}]")
        if clip_id in seen:
            raise _fail(f"clips[{index}].clip_id", f"duplicates {clip_id!r}")
        seen.add(clip_id)


def load_console_document(path: str | Path) -> dict[str, Any]:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConsoleDocumentError(f"{path}: not JSON ({exc})") from exc
    if not isinstance(document, dict):
        raise ConsoleDocumentError(f"{path}: the document must be a JSON object")
    validate_console_document(document)
    return document


# ---- reading the document ---------------------------------------------------


def configuration_of(document: Mapping[str, Any]) -> Configuration:
    return load_configuration(document["config"])


def clip_by_id(document: Mapping[str, Any], clip_id: str) -> Mapping[str, Any] | None:
    return next((clip for clip in document["clips"] if clip["clip_id"] == clip_id), None)


def shot_by_id(clip: Mapping[str, Any], shot_id: str) -> Mapping[str, Any] | None:
    return next((shot for shot in clip["shots"] if shot["shot_id"] == shot_id), None)


def field_of(shot: Mapping[str, Any], configuration: Configuration) -> Field:
    """The shot's full source set, normalized under the configuration in force.

    Built from the document's raw quantities every time rather than stored, so
    a session served under a different ``r0`` cannot show a participant numbers
    computed under the old one.
    """
    return Field(
        sources=tuple(
            Source(
                id=str(source["id"]),
                labels=tuple(str(label) for label in source["labels"]),
                prose=str(source["prose"]),
                share=float(source["share"]),
                presence=float(source["presence"]),
                confidence=float(source["confidence"]),
                energy=float(source["energy"]),
            )
            for source in shot["sources"]
        ),
        half_saturation=configuration.calibration.half_saturation,
        min_regime=configuration.calibration.min_regime,
    )


def participant_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """What the browser may hold: shot timings, and nothing measured.

    No ``r_g``, ``p_g``, ``c_g``, ``e_g``, no normalized axis, and none of §9.
    The sources, their bands and registers, and the regimes the crossfader
    draws arrive only through ``/api/shot``, computed server-side, as phrases
    and spans. Scene structure must not cue responses (§3.1), and a measured
    value must never reach a screen as a number (§4.4).

    The raw captions do travel with the clip, because the first viewing plays
    them (§3.2) and they are the model's prose, not a measurement. The
    default-policy captions do not: they are one side of the check, and the
    check follows the endpoint (§11), so they arrive from ``/api/check`` after
    every shot has a committed caption.
    """
    configuration = configuration_of(document)
    return {
        "schema": document["schema"],
        "session_id": document["session_id"],
        "config_hash": configuration.hash,
        "clips": [
            {
                "clip_id": clip["clip_id"],
                "shots": [
                    {
                        "shot_id": shot["shot_id"],
                        "start_ms": shot["start_ms"],
                        "end_ms": shot["end_ms"],
                        "raw_caption": shot["raw_caption"],
                    }
                    for shot in clip["shots"]
                ],
            }
            for clip in document["clips"]
        ],
    }
