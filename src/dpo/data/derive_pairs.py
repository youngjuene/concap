"""Pairwise training views derived from the frozen pool and raw aggregates.

`D_pair_strict` keeps only clear, confident, agreed decisive preferences and
orients them chosen/rejected. `D_pair_all` keeps every aggregate with its
outcome probabilities and difficulty for robust-method diagnostics. Both are
derived artifacts: regenerating them from the same pool and annotations is
bit-identical.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from dpo.annotation.aggregate import PairAggregate
from dpo.candidates.freeze import FrozenCandidatePool
from dpo.data.split import SplitManifest


class ViewError(ValueError):
    """Raised when a derived view would violate its construction rules."""


@dataclass(frozen=True)
class StrictPair:
    pair_id: str
    clip_id: str
    track: str
    chosen_id: str
    rejected_id: str
    chosen_text: str
    rejected_text: str
    category: str
    agreement: float
    mean_strength: float
    difficulty: float
    weight: float = 1.0

    def document(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "clip_id": self.clip_id,
            "track": self.track,
            "chosen_id": self.chosen_id,
            "rejected_id": self.rejected_id,
            "chosen_text": self.chosen_text,
            "rejected_text": self.rejected_text,
            "category": self.category,
            "agreement": self.agreement,
            "mean_strength": self.mean_strength,
            "difficulty": self.difficulty,
            "weight": self.weight,
        }


@dataclass(frozen=True)
class MetadataPair:
    pair_id: str
    clip_id: str
    track: str
    candidate_a: str
    candidate_b: str
    text_a: str
    text_b: str
    category: str
    preference_probability: float
    a_probability: float
    b_probability: float
    tie_probability: float
    both_bad_probability: float
    mean_confidence: float
    agreement: float
    disagreement_entropy: float
    difficulty: float
    weight: float = 1.0

    def document(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "clip_id": self.clip_id,
            "track": self.track,
            "candidate_a": self.candidate_a,
            "candidate_b": self.candidate_b,
            "text_a": self.text_a,
            "text_b": self.text_b,
            "category": self.category,
            "preference_probability": self.preference_probability,
            "a_probability": self.a_probability,
            "b_probability": self.b_probability,
            "tie_probability": self.tie_probability,
            "both_bad_probability": self.both_bad_probability,
            "mean_confidence": self.mean_confidence,
            "agreement": self.agreement,
            "disagreement_entropy": self.disagreement_entropy,
            "difficulty": self.difficulty,
            "weight": self.weight,
        }


def _check_membership(
    aggregates: Sequence[PairAggregate],
    pool: FrozenCandidatePool,
    manifest: SplitManifest,
    split: str,
) -> None:
    for aggregate in aggregates:
        pair = pool.pair(aggregate.pair_id)
        if pair.clip_id != aggregate.clip_id:
            raise ViewError(f"aggregate {aggregate.pair_id!r} disagrees with the frozen pair clip")
        if manifest.role_of(aggregate.clip_id) != split:
            raise ViewError(
                f"pair {aggregate.pair_id!r} belongs to split"
                f" {manifest.role_of(aggregate.clip_id)!r}, not {split!r}"
            )


def derive_pair_strict(
    aggregates: Sequence[PairAggregate],
    pool: FrozenCandidatePool,
    manifest: SplitManifest,
    *,
    split: str,
    min_agreement: float,
    min_strength: float,
) -> tuple[StrictPair, ...]:
    """D_pair_strict: clear preference, minimum strength/agreement, no ties, no both-bad."""
    _check_membership(aggregates, pool, manifest, split)
    rows = []
    for aggregate in sorted(aggregates, key=lambda item: item.pair_id):
        winner = aggregate.winner_id
        if winner is None:
            continue
        if aggregate.both_bad_count > 0:
            continue
        if aggregate.agreement < min_agreement:
            continue
        if aggregate.mean_strength is None or aggregate.mean_strength < min_strength:
            continue
        loser = aggregate.candidate_b if winner == aggregate.candidate_a else aggregate.candidate_a
        pair = pool.pair(aggregate.pair_id)
        rows.append(
            StrictPair(
                pair_id=aggregate.pair_id,
                clip_id=aggregate.clip_id,
                track=aggregate.track,
                chosen_id=winner,
                rejected_id=loser,
                chosen_text=pool.candidate(winner).text,
                rejected_text=pool.candidate(loser).text,
                category=pair.category,
                agreement=aggregate.agreement,
                mean_strength=aggregate.mean_strength,
                difficulty=aggregate.difficulty,
            )
        )
    return tuple(rows)


def derive_pair_all(
    aggregates: Sequence[PairAggregate],
    pool: FrozenCandidatePool,
    manifest: SplitManifest,
    *,
    split: str,
) -> tuple[MetadataPair, ...]:
    """D_pair_all: every aggregate with outcome probabilities and difficulty."""
    _check_membership(aggregates, pool, manifest, split)
    rows = []
    for aggregate in sorted(aggregates, key=lambda item: item.pair_id):
        pair = pool.pair(aggregate.pair_id)
        rows.append(
            MetadataPair(
                pair_id=aggregate.pair_id,
                clip_id=aggregate.clip_id,
                track=aggregate.track,
                candidate_a=aggregate.candidate_a,
                candidate_b=aggregate.candidate_b,
                text_a=pool.candidate(aggregate.candidate_a).text,
                text_b=pool.candidate(aggregate.candidate_b).text,
                category=pair.category,
                preference_probability=aggregate.preference_probability,
                a_probability=aggregate.a_probability,
                b_probability=aggregate.b_probability,
                tie_probability=aggregate.tie_probability,
                both_bad_probability=aggregate.both_bad_probability,
                mean_confidence=aggregate.mean_confidence,
                agreement=aggregate.agreement,
                disagreement_entropy=aggregate.disagreement_entropy,
                difficulty=aggregate.difficulty,
            )
        )
    return tuple(rows)


def with_weights(rows: Sequence[StrictPair], weights: Mapping[str, float]) -> tuple[StrictPair, ...]:
    """Attach clip-level weights, dropping pairs the weighting strategy removed."""
    from dataclasses import replace

    weighted = []
    for row in rows:
        weight = weights.get(row.pair_id)
        if weight is None:
            continue
        weighted.append(replace(row, weight=weight))
    return tuple(weighted)
