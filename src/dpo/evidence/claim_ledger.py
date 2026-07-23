"""The per-clip, per-track claim ledger.

Claims are proposed from evidence with `human_status="uncertain"` and move to
any other status only through an explicit human audit. Provider outputs retain
their provenance in `support_sources` and are never silently promoted to
human-verified ground truth.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from dpo.contracts.study_contract import AUDIO_CLAIM_TYPES, CLAIM_STATUSES, TRACKS, VISUAL_CLAIM_TYPES
from dpo.core.identity import semantic_hash
from dpo.core.textsafety import UntrustedTextError, validate_untrusted_text
from dpo.evidence.audio_evidence import AudioEvidence
from dpo.evidence.records import EvidenceError
from dpo.evidence.visual_evidence import VisualEvidence

_TRACK_CLAIM_TYPES: dict[str, frozenset[str]] = {
    "visual": frozenset(VISUAL_CLAIM_TYPES),
    "audio": frozenset(AUDIO_CLAIM_TYPES),
}
_HUMAN_AUDITABLE_STATUSES = frozenset(CLAIM_STATUSES)


@dataclass(frozen=True)
class Claim:
    claim_id: str
    type: str
    canonical_form: str
    start_ms: int
    end_ms: int
    support_sources: tuple[str, ...]
    support_confidence: float
    human_status: str

    def __post_init__(self) -> None:
        if not self.claim_id.strip():
            raise EvidenceError("claim_id must be non-empty")
        try:
            validate_untrusted_text(self.canonical_form, field=f"claim.{self.claim_id}.canonical_form")
        except UntrustedTextError as exc:
            raise EvidenceError(str(exc)) from exc
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise EvidenceError(f"claim {self.claim_id}: span must satisfy 0 <= start_ms < end_ms")
        if not self.support_sources or any(not source.strip() for source in self.support_sources):
            raise EvidenceError(f"claim {self.claim_id}: support sources must be non-empty provenance")
        if not math.isfinite(self.support_confidence) or not 0.0 <= self.support_confidence <= 1.0:
            raise EvidenceError(f"claim {self.claim_id}: support confidence must be in [0, 1]")
        if self.human_status not in _HUMAN_AUDITABLE_STATUSES:
            raise EvidenceError(f"claim {self.claim_id}: unknown status {self.human_status!r}")

    def document(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "type": self.type,
            "canonical_form": self.canonical_form,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "support_sources": list(self.support_sources),
            "support_confidence": self.support_confidence,
            "human_status": self.human_status,
        }


@dataclass(frozen=True)
class ClaimAudit:
    """One human decision about one claim."""

    claim_id: str
    auditor_id: str
    status: str

    def __post_init__(self) -> None:
        if not self.claim_id.strip() or not self.auditor_id.strip():
            raise EvidenceError("claim audit requires claim_id and auditor_id")
        if self.status not in _HUMAN_AUDITABLE_STATUSES:
            raise EvidenceError(f"claim audit status {self.status!r} is unknown")


@dataclass(frozen=True)
class ClaimLedger:
    clip_id: str
    track: str
    claims: tuple[Claim, ...]
    prohibited_cross_modal_claims: tuple[str, ...]
    audit_version: str

    def __post_init__(self) -> None:
        if self.track not in TRACKS:
            raise EvidenceError(f"claim ledger track must be one of {sorted(TRACKS)}")
        if not self.clip_id.strip() or not self.audit_version.strip():
            raise EvidenceError("claim ledger requires clip_id and audit_version")
        allowed = _TRACK_CLAIM_TYPES[self.track]
        seen: set[str] = set()
        for claim in self.claims:
            if claim.type not in allowed:
                raise EvidenceError(
                    f"claim {claim.claim_id}: type {claim.type!r} is invalid for track {self.track!r}"
                )
            if claim.claim_id in seen:
                raise EvidenceError(f"duplicate claim id {claim.claim_id!r}")
            seen.add(claim.claim_id)
        for text in self.prohibited_cross_modal_claims:
            try:
                validate_untrusted_text(text, field="ledger.prohibited_cross_modal_claims")
            except UntrustedTextError as exc:
                raise EvidenceError(str(exc)) from exc

    def claim(self, claim_id: str) -> Claim:
        for claim in self.claims:
            if claim.claim_id == claim_id:
                return claim
        raise EvidenceError(f"unknown claim id {claim_id!r}")

    def supported_forms(self) -> tuple[str, ...]:
        return tuple(claim.canonical_form for claim in self.claims if claim.human_status == "supported")

    def document(self) -> dict[str, object]:
        return {
            "schema": "dpo.claim-ledger/v1",
            "clip_id": self.clip_id,
            "track": self.track,
            "claims": [claim.document() for claim in self.claims],
            "prohibited_cross_modal_claims": list(self.prohibited_cross_modal_claims),
            "audit_version": self.audit_version,
        }


def propose_claims(evidence: VisualEvidence | AudioEvidence, *, track: str) -> tuple[Claim, ...]:
    """Turn evidence items into uncertain claims that await human audit."""
    if (track == "visual") != isinstance(evidence, VisualEvidence):
        raise EvidenceError("evidence record does not match the requested track")
    claims = []
    for item in evidence.items:
        claim_suffix = semantic_hash(
            {
                "clip_id": evidence.clip_id,
                "track": track,
                "kind": item.kind,
                "label": item.label,
                "start_ms": item.start_ms,
                "end_ms": item.end_ms,
            }
        ).removeprefix("sha256:")[:12]
        claims.append(
            Claim(
                claim_id=f"claim-{claim_suffix}",
                type=item.kind,
                canonical_form=item.label,
                start_ms=item.start_ms,
                end_ms=item.end_ms,
                support_sources=(evidence.provider,),
                support_confidence=item.confidence,
                human_status="uncertain",
            )
        )
    return tuple(claims)


def apply_audits(ledger: ClaimLedger, audits: Sequence[ClaimAudit]) -> ClaimLedger:
    """Apply human audit decisions, appending auditor provenance."""
    decisions: dict[str, ClaimAudit] = {}
    for audit in audits:
        if audit.claim_id in decisions:
            raise EvidenceError(f"conflicting audits for claim {audit.claim_id!r}")
        decisions[audit.claim_id] = audit
    unknown = sorted(set(decisions) - {claim.claim_id for claim in ledger.claims})
    if unknown:
        raise EvidenceError(f"audit references unknown claim {unknown[0]!r}")
    updated = []
    for claim in ledger.claims:
        decision = decisions.get(claim.claim_id)
        if decision is None:
            updated.append(claim)
            continue
        updated.append(
            replace(
                claim,
                human_status=decision.status,
                support_sources=(*claim.support_sources, decision.auditor_id),
            )
        )
    return replace(ledger, claims=tuple(updated))


def parse_claim_ledger(payload: bytes | str | Mapping[str, object]) -> ClaimLedger:
    if isinstance(payload, (bytes, str)):
        try:
            document = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise EvidenceError("claim ledger payload is not valid JSON") from exc
    else:
        document = dict(payload)
    if not isinstance(document, dict) or set(document) != {
        "schema",
        "clip_id",
        "track",
        "claims",
        "prohibited_cross_modal_claims",
        "audit_version",
    }:
        raise EvidenceError("claim ledger payload has an unexpected field set")
    if document["schema"] != "dpo.claim-ledger/v1":
        raise EvidenceError("claim ledger schema is unknown")
    claims_value = document["claims"]
    prohibited_value = document["prohibited_cross_modal_claims"]
    if not isinstance(claims_value, list) or not isinstance(prohibited_value, list):
        raise EvidenceError("claim ledger claims/prohibited lists are invalid")
    claims = []
    for index, value in enumerate(claims_value):
        if not isinstance(value, Mapping) or set(value) != {
            "claim_id",
            "type",
            "canonical_form",
            "start_ms",
            "end_ms",
            "support_sources",
            "support_confidence",
            "human_status",
        }:
            raise EvidenceError(f"claims[{index}] has an unexpected field set")
        sources = value["support_sources"]
        if not isinstance(sources, list):
            raise EvidenceError(f"claims[{index}].support_sources must be an array")
        start = value["start_ms"]
        end = value["end_ms"]
        confidence = value["support_confidence"]
        if type(start) is not int or type(end) is not int:
            raise EvidenceError(f"claims[{index}] span must be integer milliseconds")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            raise EvidenceError(f"claims[{index}].support_confidence must be a number")
        claims.append(
            Claim(
                claim_id=str(value["claim_id"]),
                type=str(value["type"]),
                canonical_form=str(value["canonical_form"]),
                start_ms=int(start),
                end_ms=int(end),
                support_sources=tuple(str(source) for source in sources),
                support_confidence=float(confidence),
                human_status=str(value["human_status"]),
            )
        )
    return ClaimLedger(
        clip_id=str(document["clip_id"]),
        track=str(document["track"]),
        claims=tuple(claims),
        prohibited_cross_modal_claims=tuple(str(item) for item in prohibited_value),
        audit_version=str(document["audit_version"]),
    )
