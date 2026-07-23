"""Visual-track evidence schema: what the muted video supports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field

from dpo.contracts.study_contract import VISUAL_CLAIM_TYPES
from dpo.evidence.records import EvidenceError, EvidenceItem, parse_item

VISUAL_ITEM_KINDS = frozenset(VISUAL_CLAIM_TYPES)


@dataclass(frozen=True)
class VisualEvidence:
    """Evidence extracted from the visual stream only. No audio field may exist."""

    clip_id: str
    provider: str
    frame_timestamps_ms: tuple[int, ...]
    scene_boundaries_ms: tuple[int, ...]
    items: tuple[EvidenceItem, ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.clip_id.strip() or not self.provider.strip():
            raise EvidenceError("visual evidence requires clip_id and provider provenance")
        if not self.frame_timestamps_ms:
            raise EvidenceError("visual evidence requires sampled frame timestamps")
        if any(timestamp < 0 for timestamp in self.frame_timestamps_ms):
            raise EvidenceError("visual frame timestamps must be non-negative")
        if list(self.frame_timestamps_ms) != sorted(self.frame_timestamps_ms):
            raise EvidenceError("visual frame timestamps must be sorted")
        if any(boundary < 0 for boundary in self.scene_boundaries_ms):
            raise EvidenceError("visual scene boundaries must be non-negative")

    def document(self) -> dict[str, object]:
        return {
            "schema": "dpo.visual-evidence/v1",
            "clip_id": self.clip_id,
            "provider": self.provider,
            "frame_timestamps_ms": list(self.frame_timestamps_ms),
            "scene_boundaries_ms": list(self.scene_boundaries_ms),
            "items": [item.document() for item in self.items],
        }


def parse_visual_evidence(payload: bytes | str | Mapping[str, object], *, provider: str) -> VisualEvidence:
    """Parse and strictly validate one visual evidence response."""
    if isinstance(payload, (bytes, str)):
        try:
            document = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise EvidenceError("visual evidence payload is not valid JSON") from exc
    else:
        document = dict(payload)
    if not isinstance(document, dict):
        raise EvidenceError("visual evidence payload must be an object")
    expected = {"language", "clip_id", "frame_timestamps_ms", "scene_boundaries_ms", "items"}
    if set(document) != expected:
        raise EvidenceError("visual evidence payload has an unexpected field set")
    if document["language"] != "en":
        raise EvidenceError("visual evidence must declare language 'en'")
    frames = document["frame_timestamps_ms"]
    scenes = document["scene_boundaries_ms"]
    items = document["items"]
    if not isinstance(frames, list) or not all(type(value) is int for value in frames):
        raise EvidenceError("visual frame_timestamps_ms must be an integer array")
    if not isinstance(scenes, list) or not all(type(value) is int for value in scenes):
        raise EvidenceError("visual scene_boundaries_ms must be an integer array")
    if not isinstance(items, list):
        raise EvidenceError("visual items must be an array")
    return VisualEvidence(
        clip_id=str(document["clip_id"]),
        provider=provider,
        frame_timestamps_ms=tuple(int(value) for value in frames),
        scene_boundaries_ms=tuple(int(value) for value in scenes),
        items=tuple(
            parse_item(value, allowed_kinds=VISUAL_ITEM_KINDS, location=f"visual.items[{index}]")
            for index, value in enumerate(items)
        ),
    )
