"""Atomic claim evaluation against the claim ledger (PRD section 26.4).

The deterministic matcher assigns a provisional status to each ledger claim a
caption mentions; explicit human judgments override it. Reports must
distinguish "no unique reference caption exists" (always true here) from "no
auditable evidence exists" (an empty or unaudited ledger), so the report
carries the ledger's audit coverage alongside the caption verdicts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from dpo.candidates.audit import content_tokens
from dpo.evidence.claim_ledger import ClaimLedger

ATOMIC_STATUSES = ("supported", "unsupported", "uncertain", "contradicted")


@dataclass(frozen=True)
class HumanClaimJudgment:
    caption_id: str
    claim_text: str
    judge_id: str
    status: str

    def __post_init__(self) -> None:
        if self.status not in ATOMIC_STATUSES:
            raise ValueError(f"claim judgment status must be one of {sorted(ATOMIC_STATUSES)}")


@dataclass(frozen=True)
class CaptionFactuality:
    caption_id: str
    clip_id: str
    matched_claims: tuple[tuple[str, str], ...]  # (claim_id, status)
    unsupported_token_count: int
    supported_mention_count: int
    contradicted_mention_count: int
    uncertain_mention_count: int
    human_overrides: int

    def document(self) -> dict[str, object]:
        return {
            "caption_id": self.caption_id,
            "clip_id": self.clip_id,
            "matched_claims": [list(entry) for entry in self.matched_claims],
            "unsupported_token_count": self.unsupported_token_count,
            "supported_mention_count": self.supported_mention_count,
            "contradicted_mention_count": self.contradicted_mention_count,
            "uncertain_mention_count": self.uncertain_mention_count,
            "human_overrides": self.human_overrides,
        }


def evaluate_caption_claims(
    caption_id: str,
    text: str,
    ledger: ClaimLedger,
    *,
    human_judgments: Sequence[HumanClaimJudgment] = (),
) -> CaptionFactuality:
    caption_tokens = content_tokens(text)
    overrides: dict[str, str] = {}
    override_count = 0
    for judgment in human_judgments:
        if judgment.caption_id != caption_id:
            continue
        overrides[judgment.claim_text.strip().casefold()] = judgment.status
        override_count += 1
    matched: list[tuple[str, str]] = []
    supported = 0
    contradicted = 0
    uncertain = 0
    claimed_tokens: set[str] = set()
    for claim in ledger.claims:
        claim_tokens = content_tokens(claim.canonical_form)
        if not claim_tokens or not claim_tokens <= caption_tokens:
            continue
        status = overrides.get(claim.canonical_form.strip().casefold(), claim.human_status)
        if status == "not_applicable":
            continue
        matched.append((claim.claim_id, status))
        claimed_tokens |= claim_tokens
        if status == "supported":
            supported += 1
        elif status == "contradicted":
            contradicted += 1
        elif status in {"uncertain", "unsupported"}:
            uncertain += 1
    return CaptionFactuality(
        caption_id=caption_id,
        clip_id=ledger.clip_id,
        matched_claims=tuple(matched),
        unsupported_token_count=len(caption_tokens - claimed_tokens),
        supported_mention_count=supported,
        contradicted_mention_count=contradicted,
        uncertain_mention_count=uncertain,
        human_overrides=override_count,
    )


@dataclass(frozen=True)
class FactualityReport:
    caption_count: int
    unsupported_claim_rate: float
    contradicted_claim_rate: float
    mean_supported_mentions: float
    ledger_audit_coverage: float

    def document(self) -> dict[str, object]:
        return {
            "schema": "dpo.factuality-report/v1",
            "caption_count": self.caption_count,
            "unsupported_claim_rate": self.unsupported_claim_rate,
            "contradicted_claim_rate": self.contradicted_claim_rate,
            "mean_supported_mentions": self.mean_supported_mentions,
            "ledger_audit_coverage": self.ledger_audit_coverage,
            "note": (
                "there is no unique reference caption; rates are measured against"
                " audited evidence, and ledger_audit_coverage reports how much of"
                " that evidence is human-audited"
            ),
        }


def summarize_factuality(
    verdicts: Sequence[CaptionFactuality], ledgers: Mapping[str, ClaimLedger]
) -> FactualityReport:
    if not verdicts:
        raise ValueError("factuality summary requires at least one caption verdict")
    unsupported = sum(1 for verdict in verdicts if verdict.unsupported_token_count > 0)
    contradicted = sum(1 for verdict in verdicts if verdict.contradicted_mention_count > 0)
    total_claims = 0
    audited_claims = 0
    for ledger in ledgers.values():
        for claim in ledger.claims:
            total_claims += 1
            if claim.human_status != "uncertain":
                audited_claims += 1
    return FactualityReport(
        caption_count=len(verdicts),
        unsupported_claim_rate=unsupported / len(verdicts),
        contradicted_claim_rate=contradicted / len(verdicts),
        mean_supported_mentions=sum(verdict.supported_mention_count for verdict in verdicts) / len(verdicts),
        ledger_audit_coverage=audited_claims / total_claims if total_claims else 0.0,
    )
