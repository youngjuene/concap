"""Per-pair aggregation of raw annotations into derived, versioned labels.

Aggregates never overwrite raw annotations; they are a separate derived
artifact. All probabilities are reported over all judgments, and the decisive
preference probability is additionally reported conditional on a decisive
outcome, because the two answer different questions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from dpo.annotation.raw_annotations import AnnotationError, RawAnnotation

AGGREGATION_VERSION = "aggregate/v1"


@dataclass(frozen=True)
class PairAggregate:
    pair_id: str
    clip_id: str
    track: str
    candidate_a: str
    candidate_b: str
    judgment_count: int
    a_better_count: int
    b_better_count: int
    tie_count: int
    both_bad_count: int
    agreement: float
    disagreement_entropy: float
    mean_confidence: float
    mean_strength: float | None
    difficulty: float
    aggregation_version: str = AGGREGATION_VERSION

    @property
    def a_probability(self) -> float:
        return self.a_better_count / self.judgment_count

    @property
    def b_probability(self) -> float:
        return self.b_better_count / self.judgment_count

    @property
    def tie_probability(self) -> float:
        return self.tie_count / self.judgment_count

    @property
    def both_bad_probability(self) -> float:
        return self.both_bad_count / self.judgment_count

    @property
    def decisive_count(self) -> int:
        return self.a_better_count + self.b_better_count

    @property
    def preference_probability(self) -> float:
        """P(candidate_a preferred | decisive outcome); 0.5 when never decisive."""
        if self.decisive_count == 0:
            return 0.5
        return self.a_better_count / self.decisive_count

    @property
    def modal_choice(self) -> str:
        counts = {
            "a_better": self.a_better_count,
            "b_better": self.b_better_count,
            "tie": self.tie_count,
            "both_unacceptable": self.both_bad_count,
        }
        best = max(counts.values())
        winners = sorted(choice for choice, count in counts.items() if count == best)
        return winners[0] if len(winners) == 1 else "tie"

    @property
    def winner_id(self) -> str | None:
        """The majority decisive winner, or None when there is no clear one."""
        if self.a_better_count > self.b_better_count and self.modal_choice == "a_better":
            return self.candidate_a
        if self.b_better_count > self.a_better_count and self.modal_choice == "b_better":
            return self.candidate_b
        return None

    def document(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "clip_id": self.clip_id,
            "track": self.track,
            "candidate_a": self.candidate_a,
            "candidate_b": self.candidate_b,
            "judgment_count": self.judgment_count,
            "a_better_count": self.a_better_count,
            "b_better_count": self.b_better_count,
            "tie_count": self.tie_count,
            "both_bad_count": self.both_bad_count,
            "a_probability": self.a_probability,
            "b_probability": self.b_probability,
            "tie_probability": self.tie_probability,
            "both_bad_probability": self.both_bad_probability,
            "preference_probability": self.preference_probability,
            "agreement": self.agreement,
            "disagreement_entropy": self.disagreement_entropy,
            "mean_confidence": self.mean_confidence,
            "mean_strength": self.mean_strength,
            "difficulty": self.difficulty,
            "aggregation_version": self.aggregation_version,
        }


def _entropy(counts: Sequence[int]) -> float:
    """Shannon entropy of the choice distribution, normalized to [0, 1]."""
    total = sum(counts)
    if total == 0 or len(counts) < 2:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count > 0:
            probability = count / total
            entropy -= probability * math.log(probability)
    return entropy / math.log(len(counts))


def aggregate_pair(annotations: Sequence[RawAnnotation]) -> PairAggregate:
    if not annotations:
        raise AnnotationError("cannot aggregate an empty annotation set")
    first = annotations[0]
    for annotation in annotations:
        if annotation.is_attention_check:
            raise AnnotationError("attention checks must be excluded before aggregation")
        if (
            annotation.pair_id != first.pair_id
            or annotation.candidate_a != first.candidate_a
            or annotation.candidate_b != first.candidate_b
        ):
            raise AnnotationError("aggregation requires annotations of exactly one pair")
    counts = {"a_better": 0, "b_better": 0, "tie": 0, "both_unacceptable": 0}
    strengths: list[int] = []
    confidences: list[int] = []
    for annotation in annotations:
        counts[annotation.canonical_choice()] += 1
        confidences.append(annotation.confidence)
        if annotation.preference_strength is not None:
            strengths.append(annotation.preference_strength)
    total = len(annotations)
    modal = max(counts.values())
    decisive = counts["a_better"] + counts["b_better"]
    margin = abs(counts["a_better"] - counts["b_better"]) / decisive if decisive else 0.0
    # Difficulty is high when decisive judgments split evenly or when ties and
    # both-bad outcomes dominate.
    difficulty = 1.0 - margin * (decisive / total)
    return PairAggregate(
        pair_id=first.pair_id,
        clip_id=first.clip_id,
        track=first.track,
        candidate_a=first.candidate_a,
        candidate_b=first.candidate_b,
        judgment_count=total,
        a_better_count=counts["a_better"],
        b_better_count=counts["b_better"],
        tie_count=counts["tie"],
        both_bad_count=counts["both_unacceptable"],
        agreement=modal / total,
        disagreement_entropy=_entropy(list(counts.values())),
        mean_confidence=sum(confidences) / total,
        mean_strength=sum(strengths) / len(strengths) if strengths else None,
        difficulty=difficulty,
    )


def aggregate_all(
    annotations: Sequence[RawAnnotation],
    *,
    minimum_judgments: int,
) -> tuple[PairAggregate, ...]:
    """Aggregate every primary (non-repeat, non-attention) judgment per pair."""
    by_pair: dict[str, list[RawAnnotation]] = {}
    for annotation in annotations:
        if annotation.is_attention_check or annotation.repeat_of is not None:
            continue
        by_pair.setdefault(annotation.pair_id, []).append(annotation)
    aggregates = []
    for pair_id in sorted(by_pair):
        pair_annotations = by_pair[pair_id]
        if len(pair_annotations) < minimum_judgments:
            raise AnnotationError(
                f"pair {pair_id!r} has {len(pair_annotations)} judgments;"
                f" the contract requires at least {minimum_judgments}"
            )
        aggregates.append(aggregate_pair(pair_annotations))
    return tuple(aggregates)
