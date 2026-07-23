"""Candidate-level factuality and modality audit.

The deterministic checks here (contract compliance, cross-modal lexicon
screen, ledger-form matching) are recall aids and hard structural gates. Claim
semantics stay with the human audit: a candidate whose deterministic screens
pass can still be rejected by a human decision, and a lexical flag can be
overturned by one. Human decisions arrive as `CandidateAuditDecision` records
with auditor provenance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from dpo.candidates.candidate_records import CandidateError, CandidateRecord
from dpo.contracts.audio_caption import check_audio_caption
from dpo.contracts.captions import ComplianceReport, caption_words
from dpo.contracts.study_contract import CaptionContract
from dpo.contracts.visual_caption import check_visual_caption
from dpo.evidence.claim_ledger import ClaimLedger

_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "are",
        "of",
        "on",
        "or",
        "the",
        "then",
        "to",
        "while",
        "with",
    }
)


def content_tokens(text: str) -> frozenset[str]:
    return frozenset(token.casefold() for token in caption_words(text)) - _STOP_WORDS


@dataclass(frozen=True)
class CandidateAudit:
    """Deterministic audit outcome for one candidate."""

    candidate_id: str
    compliance: ComplianceReport
    supported_tokens: tuple[str, ...]
    supported_overlap: int
    unsupported_token_count: int
    contradicted_overlap: int
    cross_modal_violation: bool

    @property
    def unsupported_claim_count(self) -> int:
        return self.unsupported_token_count

    def document(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "compliance": self.compliance.document(),
            "supported_tokens": list(self.supported_tokens),
            "supported_overlap": self.supported_overlap,
            "unsupported_token_count": self.unsupported_token_count,
            "contradicted_overlap": self.contradicted_overlap,
            "cross_modal_violation": self.cross_modal_violation,
            "audit_kind": "deterministic",
        }


@dataclass(frozen=True)
class CandidateAuditDecision:
    """One human decision about one candidate, with provenance."""

    candidate_id: str
    auditor_id: str
    acceptable: bool
    critical_hallucination: bool
    modality_violation: bool

    def __post_init__(self) -> None:
        if not self.candidate_id.strip() or not self.auditor_id.strip():
            raise CandidateError("candidate audit decision requires candidate and auditor ids")

    def document(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "auditor_id": self.auditor_id,
            "acceptable": self.acceptable,
            "critical_hallucination": self.critical_hallucination,
            "modality_violation": self.modality_violation,
            "audit_kind": "human",
        }


def audit_candidate(
    candidate: CandidateRecord,
    *,
    contract: CaptionContract,
    ledger: ClaimLedger,
) -> CandidateAudit:
    """Deterministic screen of one candidate against contract and ledger."""
    if candidate.track != contract.track:
        raise CandidateError("candidate track does not match the caption contract")
    if ledger.track != candidate.track or ledger.clip_id != candidate.clip_id:
        raise CandidateError("claim ledger does not belong to this candidate's clip/track")
    compliance = (
        check_visual_caption(candidate.text, contract)
        if candidate.track == "visual"
        else check_audio_caption(candidate.text, contract)
    )
    caption_tokens = content_tokens(candidate.text)
    supported_tokens: set[str] = set()
    contradicted_assertions = 0
    for claim in ledger.claims:
        claim_tokens = content_tokens(claim.canonical_form)
        if claim.human_status == "supported":
            supported_tokens |= claim_tokens
        elif (
            # A candidate asserts a contradicted claim only when it mentions the
            # claim's full canonical form; sharing one subject word with it is
            # not an assertion.
            claim.human_status == "contradicted" and claim_tokens and claim_tokens <= caption_tokens
        ):
            contradicted_assertions += 1
    prohibited_assertions = sum(
        1
        for prohibited in ledger.prohibited_cross_modal_claims
        if content_tokens(prohibited) and content_tokens(prohibited) <= caption_tokens
    )
    matched_tokens = caption_tokens & supported_tokens
    return CandidateAudit(
        candidate_id=candidate.candidate_id,
        compliance=compliance,
        supported_tokens=tuple(sorted(matched_tokens)),
        supported_overlap=len(matched_tokens),
        unsupported_token_count=len(caption_tokens - supported_tokens),
        contradicted_overlap=contradicted_assertions,
        cross_modal_violation=bool(compliance.modality_flags) or prohibited_assertions > 0,
    )


@dataclass(frozen=True)
class ResolvedCandidateAudit:
    """Deterministic screen merged with the authoritative human decision."""

    candidate_id: str
    deterministic: CandidateAudit
    human: CandidateAuditDecision | None

    @property
    def acceptable(self) -> bool:
        if self.human is not None:
            return self.human.acceptable
        return self.deterministic.compliance.compliant

    @property
    def critical_hallucination(self) -> bool:
        if self.human is not None:
            return self.human.critical_hallucination
        return self.deterministic.contradicted_overlap > 0

    @property
    def modality_violation(self) -> bool:
        if self.human is not None:
            return self.human.modality_violation
        return self.deterministic.cross_modal_violation


def resolve_audits(
    audits: Sequence[CandidateAudit],
    decisions: Sequence[CandidateAuditDecision],
) -> Mapping[str, ResolvedCandidateAudit]:
    by_candidate: dict[str, CandidateAuditDecision] = {}
    for decision in decisions:
        if decision.candidate_id in by_candidate:
            raise CandidateError(f"conflicting human audit decisions for candidate {decision.candidate_id!r}")
        by_candidate[decision.candidate_id] = decision
    audited_ids = {audit.candidate_id for audit in audits}
    unknown = sorted(set(by_candidate) - audited_ids)
    if unknown:
        raise CandidateError(f"human decision references unaudited candidate {unknown[0]!r}")
    return {
        audit.candidate_id: ResolvedCandidateAudit(
            candidate_id=audit.candidate_id,
            deterministic=audit,
            human=by_candidate.get(audit.candidate_id),
        )
        for audit in audits
    }
