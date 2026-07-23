"""Audio-track evidence schema: what the audio stream supports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field

from dpo.contracts.study_contract import AUDIO_CLAIM_TYPES
from dpo.evidence.records import EvidenceError, EvidenceItem, parse_item

AUDIO_ITEM_KINDS = frozenset(AUDIO_CLAIM_TYPES)


@dataclass(frozen=True)
class AudioEvidence:
    """Evidence extracted from the audio stream only. No visual field may exist."""

    clip_id: str
    provider: str
    speech_present: bool
    music_present: bool
    low_activity_spans_ms: tuple[tuple[int, int], ...]
    items: tuple[EvidenceItem, ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.clip_id.strip() or not self.provider.strip():
            raise EvidenceError("audio evidence requires clip_id and provider provenance")
        for span in self.low_activity_spans_ms:
            if span[0] < 0 or span[1] <= span[0]:
                raise EvidenceError("audio low-activity spans must satisfy 0 <= start < end")

    def document(self) -> dict[str, object]:
        return {
            "schema": "dpo.audio-evidence/v1",
            "clip_id": self.clip_id,
            "provider": self.provider,
            "speech_present": self.speech_present,
            "music_present": self.music_present,
            "low_activity_spans_ms": [list(span) for span in self.low_activity_spans_ms],
            "items": [item.document() for item in self.items],
        }


def parse_audio_evidence(payload: bytes | str | Mapping[str, object], *, provider: str) -> AudioEvidence:
    """Parse and strictly validate one audio evidence response."""
    if isinstance(payload, (bytes, str)):
        try:
            document = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise EvidenceError("audio evidence payload is not valid JSON") from exc
    else:
        document = dict(payload)
    if not isinstance(document, dict):
        raise EvidenceError("audio evidence payload must be an object")
    expected = {"language", "clip_id", "speech_present", "music_present", "low_activity_spans_ms", "items"}
    if set(document) != expected:
        raise EvidenceError("audio evidence payload has an unexpected field set")
    if document["language"] != "en":
        raise EvidenceError("audio evidence must declare language 'en'")
    if type(document["speech_present"]) is not bool or type(document["music_present"]) is not bool:
        raise EvidenceError("audio speech/music presence must be explicit booleans")
    spans = document["low_activity_spans_ms"]
    items = document["items"]
    if not isinstance(spans, list) or not all(
        isinstance(span, list) and len(span) == 2 and all(type(value) is int for value in span)
        for span in spans
    ):
        raise EvidenceError("audio low_activity_spans_ms must be an array of [start, end] pairs")
    if not isinstance(items, list):
        raise EvidenceError("audio items must be an array")
    return AudioEvidence(
        clip_id=str(document["clip_id"]),
        provider=provider,
        speech_present=bool(document["speech_present"]),
        music_present=bool(document["music_present"]),
        low_activity_spans_ms=tuple((int(span[0]), int(span[1])) for span in spans),
        items=tuple(
            parse_item(value, allowed_kinds=AUDIO_ITEM_KINDS, location=f"audio.items[{index}]")
            for index, value in enumerate(items)
        ),
    )
