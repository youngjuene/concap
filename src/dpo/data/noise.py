"""Natural preference-noise calibration and the shared synthetic flip manifest.

The natural-noise estimate is computed from training-split annotations only;
robust-method hyperparameters must never be estimated from the test split.
Synthetic flips are materialized as one shared manifest so every method trains
against exactly the same flipped pair ids, train labels only.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from dpo.annotation.aggregate import PairAggregate
from dpo.annotation.raw_annotations import RawAnnotation
from dpo.core.identity import semantic_hash
from dpo.data.derive_pairs import StrictPair, ViewError


@dataclass(frozen=True)
class NoiseCalibration:
    """Separable components of the natural-noise estimate (PRD section 9.5)."""

    judgment_count: int
    repeat_inconsistency: float
    majority_mismatch: float
    low_confidence_fraction: float
    disagreement_fraction: float
    epsilon_estimate: float

    def document(self) -> dict[str, object]:
        return {
            "schema": "dpo.noise-calibration/v1",
            "judgment_count": self.judgment_count,
            "repeat_inconsistency": self.repeat_inconsistency,
            "majority_mismatch": self.majority_mismatch,
            "low_confidence_fraction": self.low_confidence_fraction,
            "disagreement_fraction": self.disagreement_fraction,
            "epsilon_estimate": self.epsilon_estimate,
        }


def estimate_natural_noise(
    annotations: Sequence[RawAnnotation],
    aggregates: Sequence[PairAggregate],
    *,
    low_confidence_threshold: int = 2,
) -> NoiseCalibration:
    """Estimate the decisive-label flip rate from internal inconsistency.

    ``epsilon_estimate`` is the majority-versus-individual mismatch rate over
    decisive judgments: an unbiased plug-in for a symmetric flip model when
    the majority is taken as the reference label. The other components are
    reported separately, never mixed into the estimate.
    """
    primary_by_id = {
        annotation.annotation_id: annotation
        for annotation in annotations
        if annotation.repeat_of is None and not annotation.is_attention_check
    }
    repeats = 0
    repeat_mismatches = 0
    for annotation in annotations:
        if annotation.repeat_of is None or annotation.is_attention_check:
            continue
        primary = primary_by_id.get(annotation.repeat_of)
        if primary is None:
            raise ViewError(f"repeat {annotation.annotation_id!r} references an unknown primary")
        repeats += 1
        if primary.canonical_choice() != annotation.canonical_choice():
            repeat_mismatches += 1
    aggregate_by_pair: Mapping[str, PairAggregate] = {
        aggregate.pair_id: aggregate for aggregate in aggregates
    }
    decisive_judgments = 0
    majority_mismatches = 0
    low_confidence = 0
    total_judgments = 0
    for annotation in primary_by_id.values():
        aggregate = aggregate_by_pair.get(annotation.pair_id)
        if aggregate is None:
            continue
        total_judgments += 1
        if annotation.confidence <= low_confidence_threshold:
            low_confidence += 1
        choice = annotation.canonical_choice()
        winner = aggregate.winner_id
        if winner is None or choice not in {"a_better", "b_better"}:
            continue
        decisive_judgments += 1
        individual_winner = annotation.candidate_a if choice == "a_better" else annotation.candidate_b
        if individual_winner != winner:
            majority_mismatches += 1
    disagreeing_pairs = sum(1 for aggregate in aggregates if aggregate.agreement < 1.0)
    return NoiseCalibration(
        judgment_count=total_judgments,
        repeat_inconsistency=repeat_mismatches / repeats if repeats else 0.0,
        majority_mismatch=majority_mismatches / decisive_judgments if decisive_judgments else 0.0,
        low_confidence_fraction=low_confidence / total_judgments if total_judgments else 0.0,
        disagreement_fraction=disagreeing_pairs / len(aggregates) if aggregates else 0.0,
        epsilon_estimate=majority_mismatches / decisive_judgments if decisive_judgments else 0.0,
    )


@dataclass(frozen=True)
class FlipManifest:
    """One shared flip-index manifest per (rate, seed), used by every method."""

    flip_rate: float
    seed: int
    pair_ids: tuple[str, ...]
    flipped_pair_ids: tuple[str, ...]

    def document(self) -> dict[str, object]:
        return {
            "schema": "dpo.flip-manifest/v1",
            "flip_rate": self.flip_rate,
            "seed": self.seed,
            "pair_ids": list(self.pair_ids),
            "flipped_pair_ids": list(self.flipped_pair_ids),
        }

    @property
    def sha256(self) -> str:
        return semantic_hash(self.document())


def make_flip_manifest(rows: Sequence[StrictPair], *, flip_rate: float, seed: int) -> FlipManifest:
    if not 0.0 <= flip_rate <= 0.5:
        raise ViewError("flip_rate must be in [0, 0.5]")
    pair_ids = tuple(sorted(row.pair_id for row in rows))
    if len(pair_ids) != len(set(pair_ids)):
        raise ViewError("flip manifest requires unique pair ids")
    ordered = sorted(pair_ids, key=lambda pair_id: semantic_hash({"seed": seed, "pair": pair_id}))
    flip_count = round(len(ordered) * flip_rate)
    flipped = tuple(sorted(ordered[:flip_count]))
    return FlipManifest(flip_rate=flip_rate, seed=seed, pair_ids=pair_ids, flipped_pair_ids=flipped)


def apply_flips(rows: Sequence[StrictPair], manifest: FlipManifest) -> tuple[StrictPair, ...]:
    """Swap chosen/rejected for the manifest's pairs. Train labels only."""
    if tuple(sorted(row.pair_id for row in rows)) != manifest.pair_ids:
        raise ViewError("flip manifest was built for a different pair set")
    flipped_ids = set(manifest.flipped_pair_ids)
    result = []
    for row in rows:
        if row.pair_id in flipped_ids:
            result.append(
                replace(
                    row,
                    chosen_id=row.rejected_id,
                    rejected_id=row.chosen_id,
                    chosen_text=row.rejected_text,
                    rejected_text=row.chosen_text,
                )
            )
        else:
            result.append(row)
    return tuple(result)


def parse_flip_manifest(payload: bytes | str) -> FlipManifest:
    try:
        document = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ViewError("flip manifest payload is not valid JSON") from exc
    if not isinstance(document, dict) or set(document) != {
        "schema",
        "flip_rate",
        "seed",
        "pair_ids",
        "flipped_pair_ids",
    }:
        raise ViewError("flip manifest payload has an unexpected field set")
    if document["schema"] != "dpo.flip-manifest/v1":
        raise ViewError("flip manifest schema is unknown")
    rate = document["flip_rate"]
    seed = document["seed"]
    if not isinstance(rate, (int, float)) or isinstance(rate, bool) or type(seed) is not int:
        raise ViewError("flip manifest rate/seed are invalid")
    pair_ids = document["pair_ids"]
    flipped = document["flipped_pair_ids"]
    if not isinstance(pair_ids, list) or not isinstance(flipped, list):
        raise ViewError("flip manifest pair lists are invalid")
    manifest = FlipManifest(
        flip_rate=float(rate),
        seed=int(seed),
        pair_ids=tuple(str(item) for item in pair_ids),
        flipped_pair_ids=tuple(str(item) for item in flipped),
    )
    if not set(manifest.flipped_pair_ids) <= set(manifest.pair_ids):
        raise ViewError("flip manifest flips pairs outside its pair set")
    return manifest
