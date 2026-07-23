"""Audio-track supporting metrics (PRD section 26.3).

Sound-event coverage, foreground/background ordering, and temporal-order
accuracy are computed against the audited ledger and evidence. FENSE and
CLAP-based similarity require external checkpoints and are typed external
boundaries. Image-caption metrics are never assumed valid for audio
captioning; nothing here silently reuses one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from dpo.candidates.audit import content_tokens
from dpo.evaluation.visual_metrics import (
    EXTERNAL_BOUNDARY,
    claim_coverage,
    hallucinated_entity_rate,
    mention_order_accuracy,
)
from dpo.evidence.audio_evidence import AudioEvidence
from dpo.evidence.claim_ledger import ClaimLedger


def sound_event_coverage(text: str, ledger: ClaimLedger) -> float:
    """Fraction of supported sound_event claims the caption mentions."""
    events = [
        claim for claim in ledger.claims if claim.type == "sound_event" and claim.human_status == "supported"
    ]
    if not events:
        return 0.0
    caption_tokens = content_tokens(text)
    mentioned = sum(
        1
        for claim in events
        if content_tokens(claim.canonical_form) and content_tokens(claim.canonical_form) <= caption_tokens
    )
    return mentioned / len(events)


def foreground_first_rate(text: str, evidence: AudioEvidence) -> float:
    """How often a mentioned foreground event precedes mentioned background ones."""
    caption_tokens = content_tokens(text)
    foreground_mentioned = []
    background_mentioned = []
    lowered = text.casefold()
    for item in evidence.items:
        tokens = content_tokens(item.label)
        if not tokens or not tokens <= caption_tokens:
            continue
        positions = [lowered.index(token) for token in tokens if token in lowered]
        if not positions:
            continue
        position = min(positions)
        if item.foreground is True:
            foreground_mentioned.append(position)
        elif item.foreground is False:
            background_mentioned.append(position)
    if not foreground_mentioned or not background_mentioned:
        return 1.0
    comparisons = 0
    ordered = 0
    for front in foreground_mentioned:
        for back in background_mentioned:
            comparisons += 1
            if front < back:
                ordered += 1
    return ordered / comparisons


@dataclass(frozen=True)
class AudioMetricsReport:
    caption_count: int
    mean_sound_event_coverage: float
    mean_claim_coverage: float
    mean_hallucinated_entity_rate: float
    mean_mention_order_accuracy: float
    mean_foreground_first_rate: float
    fense: str = EXTERNAL_BOUNDARY
    clap_similarity: str = EXTERNAL_BOUNDARY

    def document(self) -> dict[str, object]:
        return {
            "schema": "dpo.audio-metrics/v1",
            "caption_count": self.caption_count,
            "mean_sound_event_coverage": self.mean_sound_event_coverage,
            "mean_claim_coverage": self.mean_claim_coverage,
            "mean_hallucinated_entity_rate": self.mean_hallucinated_entity_rate,
            "mean_mention_order_accuracy": self.mean_mention_order_accuracy,
            "mean_foreground_first_rate": self.mean_foreground_first_rate,
            "fense": self.fense,
            "clap_similarity": self.clap_similarity,
        }


def compute_audio_metrics(rows: Sequence[tuple[str, ClaimLedger, AudioEvidence]]) -> AudioMetricsReport:
    """rows: (caption text, ledger, audio evidence)."""
    if not rows:
        raise ValueError("audio metrics require at least one caption")
    events = [sound_event_coverage(text, ledger) for text, ledger, _ in rows]
    coverage = [claim_coverage(text, ledger) for text, ledger, _ in rows]
    hallucination = [hallucinated_entity_rate(text, ledger) for text, ledger, _ in rows]
    order = [mention_order_accuracy(text, ledger) for text, ledger, _ in rows]
    foreground = [foreground_first_rate(text, evidence) for text, _, evidence in rows]
    count = len(rows)
    return AudioMetricsReport(
        caption_count=count,
        mean_sound_event_coverage=sum(events) / count,
        mean_claim_coverage=sum(coverage) / count,
        mean_hallucinated_entity_rate=sum(hallucination) / count,
        mean_mention_order_accuracy=sum(order) / count,
        mean_foreground_first_rate=sum(foreground) / count,
    )
