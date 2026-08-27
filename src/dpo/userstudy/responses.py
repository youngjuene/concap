"""Validated human-study responses: what a participant's saved file must be.

``dpo study serve`` writes one ``responses-<participant>.json`` per participant
in ``dpo.userstudy-responses/v2``. This module is the reader: it checks each
file against the study export it was collected under and turns it into typed
rows. A response naming a caption the export does not carry at the position it
claims, a clip the export never had, or a rating off the instrument's scale is
refused rather than stored — the raw files stay on disk as the append-only
record, and what enters the artifact store is exactly what the instrument could
have produced.

Participants enter the store hashed, exactly as annotators do.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dpo.core.identity import sha256_bytes

# v2 records presentation_index, and times the first slider move from when
# the slider appeared rather than from when the clip did. A v1 reader would
# take the new latency for the old one and never notice.
RESPONSES_SCHEMA = "dpo.userstudy-responses/v2"
STUDY_EXPORT_SCHEMA = "dpo.study-export/v1"
MATCH_RATING_RANGE = (1, 5)

_RESPONSE_FIELDS = frozenset(
    {
        "clip_id",
        "presentation_index",
        "congruency_position",
        "congruency_index",
        "caption_shown",
        "match_rating",
        "heard_freetext",
        "slider_moves",
        "time_to_first_move_ms",
        "response_time_ms",
        "replay_count",
    }
)


class StudyResponseError(ValueError):
    """Raised when a responses document is not one the study instrument could have written."""


@dataclass(frozen=True)
class StudyResponse:
    """One participant's answer for one clip, bound to the rung the export carries."""

    participant_hash: str
    clip_id: str
    presentation_index: int
    congruency_index: int
    congruency_position: float
    # The rung's measured congruency, copied from the export: the placement on
    # the axis this response is an observation of.
    congruency: float
    caption_shown: str
    match_rating: int
    heard_freetext: str
    slider_moves: int
    time_to_first_move_ms: int | None
    response_time_ms: int
    replay_count: int

    def document(self) -> dict[str, object]:
        return {
            "participant_hash": self.participant_hash,
            "clip_id": self.clip_id,
            "presentation_index": self.presentation_index,
            "congruency_index": self.congruency_index,
            "congruency_position": self.congruency_position,
            "congruency": self.congruency,
            "caption_shown": self.caption_shown,
            "match_rating": self.match_rating,
            "heard_freetext": self.heard_freetext,
            "slider_moves": self.slider_moves,
            "time_to_first_move_ms": self.time_to_first_move_ms,
            "response_time_ms": self.response_time_ms,
            "replay_count": self.replay_count,
        }


def participant_hash(name: str) -> str:
    return sha256_bytes(f"participant:{name.strip()}".encode())


def export_ladders(export: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    """clip_id -> ascending rungs of a study export, all ladders of one width."""
    if export.get("schema") != STUDY_EXPORT_SCHEMA:
        raise StudyResponseError(f"study export schema must be {STUDY_EXPORT_SCHEMA!r}")
    clips = export.get("clips")
    if not isinstance(clips, list) or not clips:
        raise StudyResponseError("study export carries no clips")
    ladders: dict[str, list[Mapping[str, Any]]] = {}
    widths: set[int] = set()
    for clip in clips:
        if not isinstance(clip, Mapping):
            raise StudyResponseError("study export clips must be objects")
        clip_id = str(clip.get("clip_id", ""))
        levels = clip.get("levels")
        if not clip_id or not isinstance(levels, list) or len(levels) < 2:
            raise StudyResponseError(f"study export clip {clip_id!r} has no ladder")
        for level in levels:
            if not isinstance(level, Mapping) or not {"position", "congruency", "text"} <= set(level):
                raise StudyResponseError(f"study export clip {clip_id!r} has a malformed rung")
        if clip_id in ladders:
            raise StudyResponseError(f"study export lists clip {clip_id!r} twice")
        ladders[clip_id] = list(levels)
        widths.add(len(levels))
    if len(widths) != 1:
        raise StudyResponseError(f"study export ladders differ in width: {sorted(widths)}")
    return ladders


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise StudyResponseError(f"response {field} must be an integer >= {minimum}")
    return int(value)


def parse_study_responses(
    document: Mapping[str, Any], export: Mapping[str, Any]
) -> tuple[StudyResponse, ...]:
    """One participant's saved document, checked rung by rung against the export."""
    if document.get("schema") != RESPONSES_SCHEMA:
        raise StudyResponseError(f"responses document schema must be {RESPONSES_SCHEMA!r}")
    if set(document) != {"schema", "participant", "responses"}:
        raise StudyResponseError("responses document has an unexpected field set")
    participant = str(document.get("participant") or "").strip()
    if not participant:
        raise StudyResponseError("responses document requires a non-empty participant")
    rows = document.get("responses")
    if not isinstance(rows, list) or not rows:
        raise StudyResponseError("responses document must hold a non-empty responses list")
    ladders = export_ladders(export)
    identity = participant_hash(participant)
    parsed: list[StudyResponse] = []
    seen_clips: set[str] = set()
    seen_order: set[int] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _RESPONSE_FIELDS:
            unexpected = (
                sorted(set(row) ^ _RESPONSE_FIELDS) if isinstance(row, Mapping) else ["<not an object>"]
            )
            raise StudyResponseError(
                f"response has an unexpected field set (first difference {unexpected[0]!r})"
            )
        clip_id = str(row["clip_id"])
        levels = ladders.get(clip_id)
        if levels is None:
            raise StudyResponseError(
                f"response names clip {clip_id!r}, which the study export does not carry"
            )
        if clip_id in seen_clips:
            raise StudyResponseError(f"participant answered clip {clip_id!r} twice")
        seen_clips.add(clip_id)
        index = _integer(row["congruency_index"], "congruency_index")
        if index >= len(levels):
            raise StudyResponseError(
                f"response on clip {clip_id!r} names rung {index}; the ladder has {len(levels)}"
            )
        rung = levels[index]
        if str(row["caption_shown"]) != str(rung["text"]):
            raise StudyResponseError(
                f"response on clip {clip_id!r} shows a caption that is not rung {index} of the export"
            )
        position = row["congruency_position"]
        if not isinstance(position, (int, float)) or isinstance(position, bool):
            raise StudyResponseError("response congruency_position must be a number")
        if abs(float(position) - float(rung["position"])) > 1e-9:
            raise StudyResponseError(
                f"response on clip {clip_id!r} places rung {index} at {float(position):.4f};"
                f" the export places it at {float(rung['position']):.4f}"
            )
        rating = _integer(row["match_rating"], "match_rating", minimum=MATCH_RATING_RANGE[0])
        if rating > MATCH_RATING_RANGE[1]:
            raise StudyResponseError(f"response match_rating must be <= {MATCH_RATING_RANGE[1]}")
        order = _integer(row["presentation_index"], "presentation_index")
        if order in seen_order:
            raise StudyResponseError(f"participant has two responses at presentation index {order}")
        seen_order.add(order)
        first_move = row["time_to_first_move_ms"]
        parsed.append(
            StudyResponse(
                participant_hash=identity,
                clip_id=clip_id,
                presentation_index=order,
                congruency_index=index,
                congruency_position=float(position),
                congruency=float(rung["congruency"]),
                caption_shown=str(rung["text"]),
                match_rating=rating,
                heard_freetext=str(row["heard_freetext"]),
                slider_moves=_integer(row["slider_moves"], "slider_moves"),
                time_to_first_move_ms=(
                    None if first_move is None else _integer(first_move, "time_to_first_move_ms")
                ),
                response_time_ms=_integer(row["response_time_ms"], "response_time_ms"),
                replay_count=_integer(row["replay_count"], "replay_count"),
            )
        )
    return tuple(sorted(parsed, key=lambda response: response.presentation_index))


def load_study_responses(paths: Sequence[Path], export: Mapping[str, Any]) -> tuple[StudyResponse, ...]:
    """Every participant's file, parsed; one file per participant, no participant twice."""
    if not paths:
        raise StudyResponseError("study ingest needs at least one responses file")
    responses: list[StudyResponse] = []
    participants: dict[str, Path] = {}
    for path in paths:
        try:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StudyResponseError(f"responses file {str(path)!r} is not valid JSON") from exc
        if not isinstance(document, Mapping):
            raise StudyResponseError(f"responses file {str(path)!r} must hold an object")
        rows = parse_study_responses(document, export)
        identity = rows[0].participant_hash
        if identity in participants:
            raise StudyResponseError(
                f"responses files {str(participants[identity])!r} and {str(path)!r} are the same participant"
            )
        participants[identity] = Path(path)
        responses.extend(rows)
    return tuple(responses)
