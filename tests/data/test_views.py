"""Derived-view tests: orientation, gates, dedup, weighting (PRD section 32.2)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from dpo.annotation.aggregate import aggregate_all, aggregate_pair
from dpo.data.derive_pairs import derive_pair_all, derive_pair_strict, with_weights
from dpo.data.derive_sft import derive_sft
from dpo.data.noise import apply_flips, estimate_natural_noise, make_flip_manifest
from dpo.data.weighting import pair_weights
from tests.conftest import PreferenceWorld


def test_strict_pairs_orient_chosen_to_the_majority_winner(world: PreferenceWorld) -> None:
    strict = derive_pair_strict(
        world.aggregates,
        world.pool,
        world.manifest,
        split="train",
        min_agreement=0.6,
        min_strength=2.0,
    )
    assert strict
    by_pair = {aggregate.pair_id: aggregate for aggregate in world.aggregates}
    for row in strict:
        aggregate = by_pair[row.pair_id]
        assert row.chosen_id == aggregate.winner_id
        assert row.rejected_id != row.chosen_id
        assert {row.chosen_id, row.rejected_id} == {aggregate.candidate_a, aggregate.candidate_b}


def test_display_order_is_separated_from_the_canonical_label(world: PreferenceWorld) -> None:
    # Annotators saw randomized left/right order; the aggregate must be
    # invariant to it. Rebuild the aggregate with every display order flipped
    # (and choices re-expressed in display space) and require identical labels.
    sample_pair = world.aggregates[0]
    rows = [
        annotation
        for annotation in world.annotations
        if annotation.pair_id == sample_pair.pair_id and annotation.repeat_of is None
    ]
    flipped_rows = []
    for annotation in rows:
        flipped_display = (annotation.display_order[1], annotation.display_order[0])
        if annotation.choice == "a_better":
            flipped_choice = "b_better"
        elif annotation.choice == "b_better":
            flipped_choice = "a_better"
        else:
            flipped_choice = annotation.choice
        flipped_rows.append(replace(annotation, display_order=flipped_display, choice=flipped_choice))
    original = aggregate_pair(rows)
    flipped = aggregate_pair(flipped_rows)
    assert original.winner_id == flipped.winner_id
    assert original.a_better_count == flipped.a_better_count


def test_both_unacceptable_pairs_never_reach_sft(world: PreferenceWorld) -> None:
    pair = world.pool.pairs[0]
    bad_annotations = []
    for index, annotation in enumerate(world.annotations):
        if annotation.pair_id != pair.pair_id:
            continue
        bad_annotations.append(
            replace(
                annotation,
                annotation_id=f"bad-{index:04d}",
                choice="both_unacceptable",
                tie_subtype="both_bad",
                preference_strength=None,
            )
        )
    aggregates = aggregate_all(tuple(bad_annotations), minimum_judgments=3)
    rows = derive_sft(
        aggregates,
        tuple(bad_annotations),
        world.pool,
        world.manifest,
        world.audits,
        split="train",
        min_agreement=0.0,
        min_confidence=1.0,
    )
    assert rows == ()
    strict = derive_pair_strict(
        aggregates, world.pool, world.manifest, split="train", min_agreement=0.0, min_strength=1.0
    )
    assert strict == ()


def test_sft_targets_are_deduplicated_and_gated_by_audits(world: PreferenceWorld) -> None:
    rows = derive_sft(
        world.aggregates,
        world.annotations,
        world.pool,
        world.manifest,
        world.audits,
        split="train",
        min_agreement=0.6,
        min_confidence=3.0,
    )
    assert rows
    candidate_ids = [row.candidate_id for row in rows]
    assert len(candidate_ids) == len(set(candidate_ids))
    for row in rows:
        audit = world.audits[row.candidate_id]
        assert not audit.critical_hallucination
        assert not audit.modality_violation


def test_pair_all_view_carries_outcome_probabilities(world: PreferenceWorld) -> None:
    rows = derive_pair_all(world.aggregates, world.pool, world.manifest, split="train")
    assert len(rows) == len(world.aggregates)
    for row in rows:
        total = row.a_probability + row.b_probability + row.tie_probability + row.both_bad_probability
        assert total == pytest.approx(1.0)


def test_inverse_pair_count_weights_sum_to_one_per_clip(world: PreferenceWorld) -> None:
    strict = derive_pair_strict(
        world.aggregates,
        world.pool,
        world.manifest,
        split="train",
        min_agreement=0.6,
        min_strength=2.0,
    )
    weights = pair_weights(strict, strategy="inverse_pair_count", cap_per_clip=6)
    weighted = with_weights(strict, weights)
    per_clip: dict[str, float] = {}
    for row in weighted:
        per_clip[row.clip_id] = per_clip.get(row.clip_id, 0.0) + row.weight
    for total in per_clip.values():
        assert total == pytest.approx(1.0)


def test_flip_manifest_is_shared_train_only_and_reversible(world: PreferenceWorld) -> None:
    strict = derive_pair_strict(
        world.aggregates,
        world.pool,
        world.manifest,
        split="train",
        min_agreement=0.6,
        min_strength=2.0,
    )
    manifest = make_flip_manifest(strict, flip_rate=0.2, seed=13)
    assert manifest.sha256 == make_flip_manifest(strict, flip_rate=0.2, seed=13).sha256
    assert len(manifest.flipped_pair_ids) == round(0.2 * len(strict))
    flipped = apply_flips(strict, manifest)
    changed = [
        (original, mutated)
        for original, mutated in zip(strict, flipped, strict=True)
        if original.chosen_id != mutated.chosen_id
    ]
    assert len(changed) == len(manifest.flipped_pair_ids)
    for original, mutated in changed:
        assert original.chosen_id == mutated.rejected_id
        assert original.rejected_id == mutated.chosen_id
    restored = apply_flips(flipped, manifest)
    assert [row.chosen_id for row in restored] == [row.chosen_id for row in strict]


def test_natural_noise_estimate_is_zero_on_perfectly_consistent_labels(
    world: PreferenceWorld,
) -> None:
    calibration = estimate_natural_noise(world.annotations, world.aggregates)
    assert calibration.epsilon_estimate == 0.0
    assert calibration.repeat_inconsistency == 0.0
