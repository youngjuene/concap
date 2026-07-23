"""Raw annotation records (PRD section 8.7) and their canonical resolution."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from dpo.candidates.freeze import FrozenCandidatePool
from dpo.contracts.study_contract import (
    CHOICES,
    PRESENTATIONS,
    REASON_TAGS,
    TIE_SUBTYPES,
    TRACKS,
)

HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


class AnnotationError(ValueError):
    """Raised when a raw annotation is structurally invalid."""


@dataclass(frozen=True)
class RawAnnotation:
    """One human judgment exactly as collected.

    ``choice`` refers to the DISPLAYED positions; ``display_order`` maps the
    displayed left/right slots back to canonical candidate ids. The canonical
    outcome is only ever derived through :meth:`canonical_choice`, which is
    what keeps display randomization out of the science.
    """

    annotation_id: str
    pair_id: str
    clip_id: str
    track: str
    candidate_a: str
    candidate_b: str
    display_order: tuple[str, str]
    choice: str
    tie_subtype: str | None
    preference_strength: int | None
    confidence: int
    reason_tags: tuple[str, ...]
    annotator_id_hash: str
    replay_count: int
    response_time_ms: int
    collection_version: str
    presentation: str
    is_attention_check: bool = False
    repeat_of: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("annotation_id", self.annotation_id),
            ("pair_id", self.pair_id),
            ("clip_id", self.clip_id),
            ("collection_version", self.collection_version),
        ):
            if not value.strip():
                raise AnnotationError(f"annotation {name} must be non-empty")
        if self.track not in TRACKS:
            raise AnnotationError(f"annotation track must be one of {sorted(TRACKS)}")
        if self.candidate_a >= self.candidate_b:
            raise AnnotationError("annotation candidates must be in canonical id order (a < b)")
        if set(self.display_order) != {self.candidate_a, self.candidate_b}:
            raise AnnotationError("display_order must be a permutation of the pair candidates")
        if self.choice not in CHOICES:
            raise AnnotationError(f"choice must be one of {sorted(CHOICES)}")
        if self.choice in {"a_better", "b_better"}:
            if self.tie_subtype is not None:
                raise AnnotationError("a decisive choice cannot carry a tie subtype")
            if self.preference_strength is None or not 1 <= self.preference_strength <= 5:
                raise AnnotationError("a decisive choice requires preference_strength in 1-5")
        else:
            if self.preference_strength is not None:
                raise AnnotationError("tie/both-unacceptable outcomes carry no preference strength")
            if (
                self.choice == "tie"
                and self.tie_subtype is not None
                and (self.tie_subtype not in TIE_SUBTYPES)
            ):
                raise AnnotationError(f"tie_subtype must be one of {sorted(TIE_SUBTYPES)}")
            if self.choice == "both_unacceptable" and self.tie_subtype not in {None, "both_bad"}:
                raise AnnotationError("both_unacceptable maps only to the both_bad subtype")
        if not 1 <= self.confidence <= 5:
            raise AnnotationError("confidence must be in 1-5")
        unknown_tags = sorted(set(self.reason_tags) - set(REASON_TAGS))
        if unknown_tags:
            raise AnnotationError(f"unknown reason tag {unknown_tags[0]!r}")
        if len(set(self.reason_tags)) != len(self.reason_tags):
            raise AnnotationError("reason tags must be unique")
        if not HASH_RE.fullmatch(self.annotator_id_hash):
            raise AnnotationError("annotator_id_hash must be a sha256 hash")
        if self.replay_count < 0 or self.response_time_ms < 0:
            raise AnnotationError("replay_count and response_time_ms must be non-negative")
        if self.presentation not in PRESENTATIONS:
            raise AnnotationError(f"presentation must be one of {sorted(PRESENTATIONS)}")
        if self.track == "visual" and self.presentation != "muted_video":
            raise AnnotationError("visual-track judgments are always made on the muted video")
        if self.track == "audio" and self.presentation == "muted_video":
            raise AnnotationError("audio-track judgments use audio_only or unmuted_video")

    def canonical_choice(self) -> str:
        """Resolve the displayed choice against display_order to canonical ids.

        Returns one of ``a_better``/``b_better`` (canonical: `a` is
        ``candidate_a``), ``tie``, or ``both_unacceptable``.
        """
        if self.choice in {"tie", "both_unacceptable"}:
            return self.choice
        displayed_winner = self.display_order[0] if self.choice == "a_better" else self.display_order[1]
        return "a_better" if displayed_winner == self.candidate_a else "b_better"

    def winner_id(self) -> str | None:
        canonical = self.canonical_choice()
        if canonical == "a_better":
            return self.candidate_a
        if canonical == "b_better":
            return self.candidate_b
        return None

    def document(self) -> dict[str, object]:
        return {
            "annotation_id": self.annotation_id,
            "pair_id": self.pair_id,
            "clip_id": self.clip_id,
            "track": self.track,
            "candidate_a": self.candidate_a,
            "candidate_b": self.candidate_b,
            "display_order": list(self.display_order),
            "choice": self.choice,
            "tie_subtype": self.tie_subtype,
            "preference_strength": self.preference_strength,
            "confidence": self.confidence,
            "reason_tags": list(self.reason_tags),
            "annotator_id_hash": self.annotator_id_hash,
            "replay_count": self.replay_count,
            "response_time_ms": self.response_time_ms,
            "collection_version": self.collection_version,
            "presentation": self.presentation,
            "is_attention_check": self.is_attention_check,
            "repeat_of": self.repeat_of,
        }


_FIELDS = {
    "annotation_id",
    "pair_id",
    "clip_id",
    "track",
    "candidate_a",
    "candidate_b",
    "display_order",
    "choice",
    "tie_subtype",
    "preference_strength",
    "confidence",
    "reason_tags",
    "annotator_id_hash",
    "replay_count",
    "response_time_ms",
    "collection_version",
    "presentation",
    "is_attention_check",
    "repeat_of",
}


def parse_annotation(value: Mapping[str, object]) -> RawAnnotation:
    if set(value) != _FIELDS:
        raise AnnotationError("raw annotation has an unexpected field set")
    display = value["display_order"]
    if not isinstance(display, list) or len(display) != 2:
        raise AnnotationError("display_order must be a two-element array")
    tags = value["reason_tags"]
    if not isinstance(tags, list):
        raise AnnotationError("reason_tags must be an array")
    strength = value["preference_strength"]
    if strength is not None and type(strength) is not int:
        raise AnnotationError("preference_strength must be an integer or null")
    tie_subtype = value["tie_subtype"]
    if tie_subtype is not None and not isinstance(tie_subtype, str):
        raise AnnotationError("tie_subtype must be a string or null")
    repeat_of = value["repeat_of"]
    if repeat_of is not None and not isinstance(repeat_of, str):
        raise AnnotationError("repeat_of must be a string or null")
    integer_fields: dict[str, int] = {}
    for key in ("confidence", "replay_count", "response_time_ms"):
        item = value[key]
        if type(item) is not int:
            raise AnnotationError(f"{key} must be an integer")
        integer_fields[key] = int(item)
    if type(value["is_attention_check"]) is not bool:
        raise AnnotationError("is_attention_check must be a boolean")
    return RawAnnotation(
        annotation_id=str(value["annotation_id"]),
        pair_id=str(value["pair_id"]),
        clip_id=str(value["clip_id"]),
        track=str(value["track"]),
        candidate_a=str(value["candidate_a"]),
        candidate_b=str(value["candidate_b"]),
        display_order=(str(display[0]), str(display[1])),
        choice=str(value["choice"]),
        tie_subtype=None if tie_subtype is None else str(tie_subtype),
        preference_strength=None if strength is None else int(strength),
        confidence=integer_fields["confidence"],
        reason_tags=tuple(str(tag) for tag in tags),
        annotator_id_hash=str(value["annotator_id_hash"]),
        replay_count=integer_fields["replay_count"],
        response_time_ms=integer_fields["response_time_ms"],
        collection_version=str(value["collection_version"]),
        presentation=str(value["presentation"]),
        is_attention_check=bool(value["is_attention_check"]),
        repeat_of=None if repeat_of is None else str(repeat_of),
    )


def load_annotations(payload: bytes | str) -> tuple[RawAnnotation, ...]:
    """Parse a JSONL export of the append-only raw annotation store."""
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    annotations: list[RawAnnotation] = []
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AnnotationError(f"line {line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, Mapping):
            raise AnnotationError(f"line {line_number}: expected a JSON object")
        annotation = parse_annotation(value)
        if annotation.annotation_id in seen:
            raise AnnotationError(
                f"line {line_number}: duplicate annotation id {annotation.annotation_id!r};"
                " the raw store is append-only and ids are unique"
            )
        seen.add(annotation.annotation_id)
        annotations.append(annotation)
    return tuple(annotations)


def validate_against_pool(annotations: Sequence[RawAnnotation], pool: FrozenCandidatePool) -> None:
    """Every non-attention annotation must reference the frozen pool exactly."""
    for annotation in annotations:
        if annotation.is_attention_check:
            continue
        pair = pool.pair(annotation.pair_id)
        if (
            pair.clip_id != annotation.clip_id
            or pair.track != annotation.track
            or pair.candidate_a != annotation.candidate_a
            or pair.candidate_b != annotation.candidate_b
        ):
            raise AnnotationError(
                f"annotation {annotation.annotation_id!r} disagrees with the frozen pair"
                f" {annotation.pair_id!r}"
            )
