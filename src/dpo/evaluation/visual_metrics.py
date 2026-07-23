"""Visual-track supporting metrics (PRD section 26.2).

Claim-grounded metrics are computed locally against the audited ledger.
Similarity metrics that need external checkpoints (frame-aggregated CLIPScore,
learned factuality checkers) are typed external boundaries and never silently
proxied. All of these are secondary to human preference and claim support.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from dpo.candidates.audit import content_tokens
from dpo.contracts.captions import caption_words
from dpo.evidence.claim_ledger import ClaimLedger

EXTERNAL_BOUNDARY = "blocked_pending_external_operation"


def claim_coverage(text: str, ledger: ClaimLedger) -> float:
    """Fraction of human-supported ledger claims the caption mentions."""
    supported = [claim for claim in ledger.claims if claim.human_status == "supported"]
    if not supported:
        return 0.0
    caption_tokens = content_tokens(text)
    mentioned = sum(
        1
        for claim in supported
        if content_tokens(claim.canonical_form) and content_tokens(claim.canonical_form) <= caption_tokens
    )
    return mentioned / len(supported)


def hallucinated_entity_rate(text: str, ledger: ClaimLedger) -> float:
    """Fraction of caption content tokens outside every supported claim."""
    caption_tokens = content_tokens(text)
    if not caption_tokens:
        return 0.0
    supported_tokens: set[str] = set()
    for claim in ledger.claims:
        if claim.human_status == "supported":
            supported_tokens |= content_tokens(claim.canonical_form)
    return len(caption_tokens - supported_tokens) / len(caption_tokens)


def temporal_coverage(text: str, ledger: ClaimLedger, *, clip_ms: int) -> float:
    """Fraction of the clip's timeline covered by mentioned supported claims."""
    if clip_ms <= 0:
        raise ValueError("clip duration must be positive")
    caption_tokens = content_tokens(text)
    covered: list[tuple[int, int]] = []
    for claim in ledger.claims:
        if claim.human_status != "supported":
            continue
        tokens = content_tokens(claim.canonical_form)
        if tokens and tokens <= caption_tokens:
            covered.append((claim.start_ms, claim.end_ms))
    if not covered:
        return 0.0
    covered.sort()
    merged = [list(covered[0])]
    for start, end in covered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return min(1.0, sum(end - start for start, end in merged) / clip_ms)


def mention_order_accuracy(text: str, ledger: ClaimLedger) -> float:
    """Pairwise agreement between mention order and evidence span order."""
    caption_tokens = [token.casefold() for token in caption_words(text)]
    first_positions = []
    for claim in sorted(ledger.claims, key=lambda claim: claim.claim_id):
        if claim.human_status != "supported":
            continue
        tokens = content_tokens(claim.canonical_form)
        if not tokens or not tokens <= set(caption_tokens):
            continue
        position = min(caption_tokens.index(token) for token in tokens)
        first_positions.append((position, claim.start_ms))
    if len(first_positions) < 2:
        return 1.0
    agreements = 0
    comparisons = 0
    for index in range(len(first_positions)):
        for other in range(index + 1, len(first_positions)):
            left, right = first_positions[index], first_positions[other]
            if left[1] == right[1]:
                continue
            comparisons += 1
            if (left[0] < right[0]) == (left[1] < right[1]):
                agreements += 1
    return agreements / comparisons if comparisons else 1.0


@dataclass(frozen=True)
class VisualMetricsReport:
    caption_count: int
    mean_claim_coverage: float
    mean_hallucinated_entity_rate: float
    mean_temporal_coverage: float
    mean_mention_order_accuracy: float
    clipscore: str = EXTERNAL_BOUNDARY
    learned_factuality_checker: str = EXTERNAL_BOUNDARY

    def document(self) -> dict[str, object]:
        return {
            "schema": "dpo.visual-metrics/v1",
            "caption_count": self.caption_count,
            "mean_claim_coverage": self.mean_claim_coverage,
            "mean_hallucinated_entity_rate": self.mean_hallucinated_entity_rate,
            "mean_temporal_coverage": self.mean_temporal_coverage,
            "mean_mention_order_accuracy": self.mean_mention_order_accuracy,
            "clipscore": self.clipscore,
            "learned_factuality_checker": self.learned_factuality_checker,
        }


def compute_visual_metrics(rows: Sequence[tuple[str, ClaimLedger, int]]) -> VisualMetricsReport:
    """rows: (caption text, ledger, clip duration ms)."""
    if not rows:
        raise ValueError("visual metrics require at least one caption")
    coverage = [claim_coverage(text, ledger) for text, ledger, _ in rows]
    hallucination = [hallucinated_entity_rate(text, ledger) for text, ledger, _ in rows]
    temporal = [temporal_coverage(text, ledger, clip_ms=clip_ms) for text, ledger, clip_ms in rows]
    order = [mention_order_accuracy(text, ledger) for text, ledger, _ in rows]
    count = len(rows)
    return VisualMetricsReport(
        caption_count=count,
        mean_claim_coverage=sum(coverage) / count,
        mean_hallucinated_entity_rate=sum(hallucination) / count,
        mean_temporal_coverage=sum(temporal) / count,
        mean_mention_order_accuracy=sum(order) / count,
    )
