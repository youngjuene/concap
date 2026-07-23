"""Typed evidence items shared by the visual and audio evidence schemas."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from dpo.core.textsafety import UntrustedTextError, validate_untrusted_text


class EvidenceError(ValueError):
    """Raised when an evidence record or modality boundary is invalid."""


@dataclass(frozen=True)
class EvidenceItem:
    """One evidence observation with provenance-free content.

    Provenance (which model or auditor produced it) lives on the enclosing
    evidence record and in the provider artifact lineage, not per item.
    """

    kind: str
    label: str
    start_ms: int
    end_ms: int
    confidence: float
    salience: float
    foreground: bool | None = None

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise EvidenceError("evidence item kind must be non-empty")
        try:
            validate_untrusted_text(self.label, field="evidence.item.label")
        except UntrustedTextError as exc:
            raise EvidenceError(str(exc)) from exc
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise EvidenceError("evidence item span must satisfy 0 <= start_ms < end_ms")
        for name, value in (("confidence", self.confidence), ("salience", self.salience)):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise EvidenceError(f"evidence item {name} must be in [0, 1]")

    def document(self) -> dict[str, object]:
        body: dict[str, object] = {
            "kind": self.kind,
            "label": self.label,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "confidence": self.confidence,
            "salience": self.salience,
        }
        if self.foreground is not None:
            body["foreground"] = self.foreground
        return body


def parse_item(value: object, *, allowed_kinds: frozenset[str], location: str) -> EvidenceItem:
    if not isinstance(value, Mapping):
        raise EvidenceError(f"{location} must be an object")
    keys = set(value)
    if keys not in (
        {"kind", "label", "start_ms", "end_ms", "confidence", "salience"},
        {"kind", "label", "start_ms", "end_ms", "confidence", "salience", "foreground"},
    ):
        raise EvidenceError(f"{location} has an unexpected field set")
    kind = str(value["kind"])
    if kind not in allowed_kinds:
        raise EvidenceError(f"{location}.kind {kind!r} is not valid for this track")
    start = value["start_ms"]
    end = value["end_ms"]
    if type(start) is not int or type(end) is not int:
        raise EvidenceError(f"{location} span must be integer milliseconds")
    confidence = value["confidence"]
    salience = value["salience"]
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise EvidenceError(f"{location}.confidence must be a number")
    if not isinstance(salience, (int, float)) or isinstance(salience, bool):
        raise EvidenceError(f"{location}.salience must be a number")
    foreground = value.get("foreground")
    if foreground is not None and type(foreground) is not bool:
        raise EvidenceError(f"{location}.foreground must be a boolean when present")
    try:
        return EvidenceItem(
            kind=kind,
            label=str(value["label"]),
            start_ms=int(start),
            end_ms=int(end),
            confidence=float(confidence),
            salience=float(salience),
            foreground=foreground,
        )
    except EvidenceError:
        raise
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{location} is invalid: {exc}") from exc
