"""The session document: what a session IS, and what of it a participant may see.

The document is the configuration (spec 1: "Configuration assigns, per clip,
whether the first viewing carries captions, whether the clip is shaped or given
the control task, and the settings the skeleton opens at"). Clip order is the
list order; there is no separate ordering field, so nothing can disagree with
it. Everything the skeleton draws — sources, their measured phrases, their
weights, roles, the scene and atmosphere rows — is authored here per shot.

Two readers with different rights. The server reads everything. The browser
reads ``participant_document``: weights never leave the server (spec 2: no
measured value is ever shown as a number, and the orderings the browser draws
are precomputed from them, not the weights themselves), and the whole
sources/scene/atmosphere block only travels through the gated inventory route
(spec 2 last paragraph, spec 7: "the skeleton cannot appear early by any
path"). The follow-up answers stay out for the same reason.

Validation names the offending path (``clips[2].shots[0].sources[3].weights``)
because a researcher authors this by hand and a bare "invalid document" costs
them a hunt through several hundred lines.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dpo.caption.writer import CAPTION_MAX_CHARS

SESSION_SCHEMA = "dpo.caption-session/v1"
ID_RE = re.compile(r"[A-Za-z0-9_-]+\Z")
LEVELS = ("itemized", "grouped", "scene", "atmospheric")
TASKS = ("shaped", "control")
BALANCE_CONTROLS = ("orderings", "crossfader")

# Spec Table 6, verbatim: the copy the document must carry unchanged so the
# browser can render it without owning a second copy of the strings.
POLES = {"audio": ["Eye", "Ear"], "visual": ["Near", "Far"]}
AUDIO_ROLE_HEADS = {"underneath": "Underneath", "stands_out": "Stands out", "only_here": "Only here"}
ATTENTION_POLES = ["Mostly the image", "Mostly the sound"]
ATTENTION_TEXT = "Where was your attention?"
CRITERION_TEXT = "Is this the caption you'd want for this moment?"
OPEN_ITEM_TEXT = "What makes this place sound like itself?"
MAX_SOURCES = 8
# The strip and the placed box hold two lines of 22px on the kiosk's stage
# (kiosk.css ``--stage-w``, 640 wide): about 96 characters of this face. A
# longer caption clips behind an ellipsis in both positions (spec 4.4 "same
# two-line box"; spec 2 "the read caption never vanishes"), and a caption the
# participant cannot read cannot be kept or compared. Authored captions —
# the automatic caption, the scene and atmosphere prose — are held to it
# here, against the one budget both instruments share
# (dpo.caption.writer.CAPTION_MAX_CHARS).


class SessionDocumentError(ValueError):
    """The document violates the schema; the message names the path."""


def load_session_document(path: str | Path) -> dict[str, Any]:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SessionDocumentError(f"{path}: not JSON ({exc})") from exc
    if not isinstance(document, dict):
        raise SessionDocumentError(f"{path}: the document must be a JSON object")
    validate_session_document(document)
    return document


# ---- validation ------------------------------------------------------------


def _fail(path: str, message: str) -> SessionDocumentError:
    return SessionDocumentError(f"{path}: {message}")


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


def _string(value: object, path: str, *, empty: bool = False) -> str:
    if not isinstance(value, str):
        raise _fail(path, "must be a string")
    if not empty and not value.strip():
        raise _fail(path, "must not be empty")
    return value


def _integer(value: object, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(path, "must be an integer")
    if minimum is not None and value < minimum:
        raise _fail(path, f"must be at least {minimum}")
    return value


def _unit(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _fail(path, "must be a number")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise _fail(path, "must lie in [0, 1]")
    return number


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise _fail(path, "must be true or false")
    return value


def _identifier(value: object, path: str) -> str:
    text = _string(value, path)
    if not ID_RE.fullmatch(text):
        raise _fail(path, "must match [A-Za-z0-9_-]+")
    return text


def _choice(value: object, path: str, choices: Sequence[str]) -> str:
    text = _string(value, path)
    if text not in choices:
        raise _fail(path, f"must be one of {list(choices)}")
    return text


def _token(value: object, path: str) -> str:
    """Uppercase words, no brackets: the UI adds the brackets (contract §2)."""
    text = _string(value, path)
    if "[" in text or "]" in text:
        raise _fail(path, "must not contain brackets; the interface draws them")
    if text != text.upper():
        raise _fail(path, "must be uppercase")
    return text


def _phrase(value: object, path: str) -> str:
    """A measured quantity as words: never numbers (spec 2, spec-identity.md Register)."""
    text = _string(value, path)
    if any(char.isdigit() for char in text):
        raise _fail(path, "must carry no numerals; measured quantities appear only as phrases")
    return text


def _caption(value: object, path: str) -> str:
    """Authored caption prose: placed or written verbatim, so it must fit the two-line box."""
    text = _string(value, path)
    if len(text) > CAPTION_MAX_CHARS:
        raise _fail(path, f"must be at most {CAPTION_MAX_CHARS} characters; longer clips in the two-line box")
    return text


def _validate_measures(value: object, path: str) -> None:
    measures = _mapping(value, path)
    items = _sequence(measures.get("items"), f"{path}.items", minimum=1)
    seen: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(items):
        item_path = f"{path}.items[{index}]"
        item = _mapping(raw, item_path)
        item_id = _identifier(item.get("id"), f"{item_path}.id")
        if item_id in seen:
            raise _fail(f"{item_path}.id", f"duplicates item {item_id!r}")
        seen[item_id] = item
        _string(item.get("text"), f"{item_path}.text")
        _integer(item.get("boxes"), f"{item_path}.boxes", minimum=2)
        poles = _sequence(item.get("poles"), f"{item_path}.poles")
        if len(poles) != 2:
            raise _fail(f"{item_path}.poles", "must be [left, right]")
        for side, pole in enumerate(poles):
            _string(pole, f"{item_path}.poles[{side}]")
    attention = seen.get("attention")
    if attention is None:
        raise _fail(f"{path}.items", "must contain the item with id 'attention'")
    if attention["boxes"] != 7:
        raise _fail(f"{path}.items[attention].boxes", "must be 7")
    if list(attention["poles"]) != ATTENTION_POLES:
        raise _fail(f"{path}.items[attention].poles", f"must be {ATTENTION_POLES}")
    if attention["text"] != ATTENTION_TEXT:
        raise _fail(f"{path}.items[attention].text", f"must be {ATTENTION_TEXT!r}")
    criterion = seen.get("criterion")
    if criterion is None:
        raise _fail(f"{path}.items", "must contain the item with id 'criterion'")
    if criterion["boxes"] != 2:
        raise _fail(f"{path}.items[criterion].boxes", "must be 2")
    if criterion["text"] != CRITERION_TEXT:
        raise _fail(f"{path}.items[criterion].text", f"must be {CRITERION_TEXT!r}")
    if measures.get("open_item") != OPEN_ITEM_TEXT:
        raise _fail(f"{path}.open_item", f"must be {OPEN_ITEM_TEXT!r}")


def _validate_source(raw: object, path: str, roles: Mapping[str, str]) -> str:
    source = _mapping(raw, path)
    source_id = _identifier(source.get("id"), f"{path}.id")
    _token(source.get("token"), f"{path}.token")
    _string(source.get("prose"), f"{path}.prose")
    phrases = _sequence(source.get("phrases"), f"{path}.phrases")
    if len(phrases) != 2:
        raise _fail(f"{path}.phrases", "must be [left phrase, right phrase]")
    for side, phrase in enumerate(phrases):
        _phrase(phrase, f"{path}.phrases[{side}]")
    weights = _sequence(source.get("weights"), f"{path}.weights")
    if len(weights) != 2:
        raise _fail(f"{path}.weights", "must be [left, right]")
    for side, weight in enumerate(weights):
        _unit(weight, f"{path}.weights[{side}]")
    _choice(source.get("role"), f"{path}.role", list(roles))
    return source_id


def _validate_shot(raw: object, path: str, roles: Mapping[str, str], expected_start: int) -> tuple[str, int]:
    shot = _mapping(raw, path)
    shot_id = _identifier(shot.get("shot_id"), f"{path}.shot_id")
    start = _integer(shot.get("start_ms"), f"{path}.start_ms", minimum=0)
    end = _integer(shot.get("end_ms"), f"{path}.end_ms", minimum=0)
    if start != expected_start:
        raise _fail(f"{path}.start_ms", f"must be {expected_start}; shots are contiguous from 0")
    if end <= start:
        raise _fail(f"{path}.end_ms", "must exceed start_ms")
    _caption(shot.get("automatic_caption"), f"{path}.automatic_caption")
    scene = _mapping(shot.get("scene"), f"{path}.scene")
    _token(scene.get("token"), f"{path}.scene.token")
    _caption(scene.get("prose"), f"{path}.scene.prose")
    atmosphere = _mapping(shot.get("atmosphere"), f"{path}.atmosphere")
    _phrase(atmosphere.get("phrase"), f"{path}.atmosphere.phrase")
    _caption(atmosphere.get("prose"), f"{path}.atmosphere.prose")
    sources = _sequence(shot.get("sources"), f"{path}.sources", minimum=1)
    if len(sources) > MAX_SOURCES:
        raise _fail(f"{path}.sources", f"must have at most {MAX_SOURCES} entries")
    seen: set[str] = set()
    for index, source in enumerate(sources):
        source_id = _validate_source(source, f"{path}.sources[{index}]", roles)
        if source_id in seen:
            raise _fail(f"{path}.sources[{index}].id", f"duplicates source {source_id!r}")
        seen.add(source_id)
    return shot_id, end


def _validate_clip(raw: object, path: str, role_heads: Mapping[str, str]) -> tuple[str, str, int]:
    clip = _mapping(raw, path)
    clip_id = _identifier(clip.get("clip_id"), f"{path}.clip_id")
    task = _choice(clip.get("task"), f"{path}.task", TASKS)
    _boolean(clip.get("first_viewing_captions"), f"{path}.first_viewing_captions")
    _choice(clip.get("balance_control"), f"{path}.balance_control", BALANCE_CONTROLS)
    opening = _mapping(clip.get("opening"), f"{path}.opening")
    _choice(opening.get("level"), f"{path}.opening.level", LEVELS)
    _unit(opening.get("balance"), f"{path}.opening.balance")
    if task == "control":
        visual_roles = _mapping(clip.get("visual_roles"), f"{path}.visual_roles")
        if len(visual_roles) != 3:
            raise _fail(f"{path}.visual_roles", "must name exactly three role heads")
        for role_id, head in visual_roles.items():
            _identifier(role_id, f"{path}.visual_roles")
            _string(head, f"{path}.visual_roles[{role_id}]")
        roles: Mapping[str, str] = visual_roles
    else:
        if "visual_roles" in clip:
            raise _fail(
                f"{path}.visual_roles", "belongs to control clips only; shaped clips use role_heads.audio"
            )
        roles = role_heads
    shots = _sequence(clip.get("shots"), f"{path}.shots", minimum=1)
    seen: set[str] = set()
    cursor = 0
    for index, shot in enumerate(shots):
        shot_id, cursor = _validate_shot(shot, f"{path}.shots[{index}]", roles, cursor)
        if shot_id in seen:
            raise _fail(f"{path}.shots[{index}].shot_id", f"duplicates shot {shot_id!r}")
        seen.add(shot_id)
    return clip_id, task, cursor


def _validate_followup(value: object, path: str, clip_ends: Mapping[str, int], shaped: Sequence[str]) -> None:
    followup = _mapping(value, path)
    recognition = _sequence(followup.get("recognition"), f"{path}.recognition")
    covered: set[str] = set()
    for index, raw in enumerate(recognition):
        entry_path = f"{path}.recognition[{index}]"
        entry = _mapping(raw, entry_path)
        clip_id = _choice(entry.get("clip_id"), f"{entry_path}.clip_id", list(clip_ends))
        if clip_id in covered:
            raise _fail(f"{entry_path}.clip_id", f"duplicates clip {clip_id!r}")
        covered.add(clip_id)
        sounds = _sequence(entry.get("sounds"), f"{entry_path}.sounds", minimum=1)
        for sound_index, sound in enumerate(sounds):
            _token(sound, f"{entry_path}.sounds[{sound_index}]")
    missing = [clip_id for clip_id in clip_ends if clip_id not in covered]
    if missing:
        raise _fail(f"{path}.recognition", f"must cover every clip; missing {missing[0]!r}")
    excerpts: set[str] = set()
    for index, raw in enumerate(_sequence(followup.get("sound_only"), f"{path}.sound_only")):
        entry_path = f"{path}.sound_only[{index}]"
        entry = _mapping(raw, entry_path)
        excerpt_id = _identifier(entry.get("excerpt_id"), f"{entry_path}.excerpt_id")
        if excerpt_id in excerpts:
            raise _fail(f"{entry_path}.excerpt_id", f"duplicates excerpt {excerpt_id!r}")
        excerpts.add(excerpt_id)
        clip_id = _choice(entry.get("clip_id"), f"{entry_path}.clip_id", list(clip_ends))
        start = _integer(entry.get("start_ms"), f"{entry_path}.start_ms", minimum=0)
        end = _integer(entry.get("end_ms"), f"{entry_path}.end_ms", minimum=0)
        if end <= start:
            raise _fail(f"{entry_path}.end_ms", "must exceed start_ms")
        if end > clip_ends[clip_id]:
            raise _fail(f"{entry_path}.end_ms", f"exceeds the end of clip {clip_id!r} ({clip_ends[clip_id]})")
    checked: set[str] = set()
    for index, raw in enumerate(_sequence(followup.get("check"), f"{path}.check")):
        entry_path = f"{path}.check[{index}]"
        entry = _mapping(raw, entry_path)
        clip_id = _choice(entry.get("clip_id"), f"{entry_path}.clip_id", list(shaped))
        if clip_id in checked:
            raise _fail(f"{entry_path}.clip_id", f"duplicates clip {clip_id!r}")
        checked.add(clip_id)
        _choice(entry.get("a"), f"{entry_path}.a", ("own", "automatic"))
    missing = [clip_id for clip_id in shaped if clip_id not in checked]
    if missing:
        raise _fail(f"{path}.check", f"must cover every shaped clip; missing {missing[0]!r}")


def validate_session_document(document: Mapping[str, Any]) -> None:
    """Raise ``SessionDocumentError`` naming the first offending path."""
    if document.get("schema") != SESSION_SCHEMA:
        raise _fail("schema", f"must be {SESSION_SCHEMA!r}")
    _string(document.get("session_id"), "session_id")
    if document.get("poles") != POLES:
        raise _fail("poles", f"must be {POLES} (spec Table 6)")
    role_heads = _mapping(document.get("role_heads"), "role_heads")
    if role_heads.get("audio") != AUDIO_ROLE_HEADS:
        raise _fail("role_heads.audio", f"must be {AUDIO_ROLE_HEADS} (spec Table 6)")
    _validate_measures(document.get("measures"), "measures")
    clips = _sequence(document.get("clips"), "clips", minimum=1)
    clip_ends: dict[str, int] = {}
    tasks: dict[str, str] = {}
    for index, clip in enumerate(clips):
        clip_id, task, end = _validate_clip(clip, f"clips[{index}]", AUDIO_ROLE_HEADS)
        if clip_id in clip_ends:
            raise _fail(f"clips[{index}].clip_id", f"duplicates clip {clip_id!r}")
        clip_ends[clip_id] = end
        tasks[clip_id] = task
    for task in TASKS:
        if task not in tasks.values():
            raise _fail("clips", f"must contain at least one {task} clip")
    shaped = [clip_id for clip_id, task in tasks.items() if task == "shaped"]
    _validate_followup(document.get("followup"), "followup", clip_ends, shaped)


# ---- lookups ---------------------------------------------------------------


def clip_by_id(document: Mapping[str, Any], clip_id: str) -> Mapping[str, Any] | None:
    clips: Sequence[Mapping[str, Any]] = document["clips"]
    return next((clip for clip in clips if clip["clip_id"] == clip_id), None)


def shot_by_id(clip: Mapping[str, Any], shot_id: str) -> Mapping[str, Any] | None:
    shots: Sequence[Mapping[str, Any]] = clip["shots"]
    return next((shot for shot in shots if shot["shot_id"] == shot_id), None)


def roles_for(document: Mapping[str, Any], clip: Mapping[str, Any]) -> dict[str, str]:
    """The role heads a clip's skeleton groups under, in document order.

    Shaped clips share the three soundscape roles (spec 4.3); a control clip
    carries its own three visual roles with the visual inventory (spec 4.6).
    """
    if clip["task"] == "control":
        return dict(clip["visual_roles"])
    return dict(document["role_heads"]["audio"])


def poles_for(document: Mapping[str, Any], clip: Mapping[str, Any]) -> list[str]:
    return list(document["poles"]["visual" if clip["task"] == "control" else "audio"])


# ---- participant narrowing -------------------------------------------------


def participant_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """What the browser may hold from the first request onward.

    Per shot only its bounds and the automatic caption: the sources, their
    phrases, the scene and atmosphere rows all wait behind the listing gate
    (spec 2: "Show nothing the tool knows before the participant has said what
    they noticed"). So do a control clip's visual role heads — they are
    "supplied with the visual inventory" (spec 4.6) and the inventory route
    returns them as ``role_heads``; the audio heads stay because they are
    Table 6 copy, the same on every clip. Weights never travel at all. The
    follow-up keeps the recognition lists (those are stimuli) and the excerpt
    ids, but not which clip an excerpt came from nor which caption set is A —
    both are answers.
    """
    return {
        "schema": document["schema"],
        "session_id": document["session_id"],
        "poles": {key: list(value) for key, value in document["poles"].items()},
        "role_heads": {key: dict(value) for key, value in document["role_heads"].items()},
        "measures": {
            "items": [
                {
                    "id": item["id"],
                    "text": item["text"],
                    "boxes": item["boxes"],
                    "poles": list(item["poles"]),
                }
                for item in document["measures"]["items"]
            ],
            "open_item": document["measures"]["open_item"],
        },
        "clips": [
            {
                "clip_id": clip["clip_id"],
                "task": clip["task"],
                "first_viewing_captions": clip["first_viewing_captions"],
                "balance_control": clip["balance_control"],
                "opening": dict(clip["opening"]),
                "shots": [
                    {
                        "shot_id": shot["shot_id"],
                        "start_ms": shot["start_ms"],
                        "end_ms": shot["end_ms"],
                        "automatic_caption": shot["automatic_caption"],
                    }
                    for shot in clip["shots"]
                ],
            }
            for clip in document["clips"]
        ],
        "followup": {
            "recognition": [
                {"clip_id": entry["clip_id"], "sounds": list(entry["sounds"])}
                for entry in document["followup"]["recognition"]
            ],
            "sound_only": [
                {"excerpt_id": entry["excerpt_id"]} for entry in document["followup"]["sound_only"]
            ],
            "check": [{"clip_id": entry["clip_id"]} for entry in document["followup"]["check"]],
        },
    }
