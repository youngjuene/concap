"""Candidate construction, audit, and pair-sampler tests."""

from __future__ import annotations

import pytest

from dpo.candidates.candidate_records import (
    CandidateError,
    CollectionPolicy,
    GenerationConfig,
    build_candidate_records,
    validate_pool_mixture,
)
from dpo.candidates.pair_sampler import sample_pairs
from dpo.contracts.study_contract import PAIR_CATEGORIES
from dpo.core.identity import sha256_bytes
from tests.conftest import PreferenceWorld


def test_candidate_ids_are_content_derived() -> None:
    policy = CollectionPolicy(policy_id="C0", checkpoint_hash=sha256_bytes(b"c0"))
    config = GenerationConfig(temperature=0.0, top_p=1.0, max_new_tokens=32, seed=0)
    generations = [
        ("greedy", "A runner passes the fountain in the square.", config),
        ("sample", "Near the square, a runner circles the fountain.", config),
        ("sample", "A runner slows beside the fountain and stops.", config),
        ("controlled_error", "A runner waves at a parade beside the fountain.", config),
    ]
    first = build_candidate_records(clip_id="clip-x", track="visual", policy=policy, generations=generations)
    second = build_candidate_records(clip_id="clip-x", track="visual", policy=policy, generations=generations)
    assert [c.candidate_id for c in first] == [c.candidate_id for c in second]
    assert all(c.candidate_hash == d.candidate_hash for c, d in zip(first, second, strict=True))


def test_pool_requires_at_least_four_candidates() -> None:
    policy = CollectionPolicy(policy_id="C0", checkpoint_hash=sha256_bytes(b"c0"))
    config = GenerationConfig(temperature=0.0, top_p=1.0, max_new_tokens=32, seed=0)
    with pytest.raises(CandidateError, match="at least four"):
        build_candidate_records(
            clip_id="clip-x",
            track="visual",
            policy=policy,
            generations=[("greedy", "Only one candidate here.", config)],
        )


def test_mixture_composition_is_enforced(world: PreferenceWorld) -> None:
    validate_pool_mixture(
        world.pool.candidates,
        per_clip=4,
        challenge_fraction_min=0.2,
        challenge_fraction_max=0.3,
    )
    with pytest.raises(CandidateError, match="challenge"):
        validate_pool_mixture(
            world.pool.candidates,
            per_clip=4,
            challenge_fraction_min=0.4,
            challenge_fraction_max=0.5,
        )


def test_pairs_are_within_clip_canonically_ordered_and_categorized(
    world: PreferenceWorld,
) -> None:
    for pair in world.pool.pairs:
        assert pair.candidate_a < pair.candidate_b
        assert world.pool.candidate(pair.candidate_a).clip_id == pair.clip_id
        assert world.pool.candidate(pair.candidate_b).clip_id == pair.clip_id
        assert pair.category in PAIR_CATEGORIES


def test_sampler_is_deterministic_and_bounded(world: PreferenceWorld) -> None:
    pairs = sample_pairs(
        world.pool.candidates,
        world.audits,
        per_clip_min=3,
        per_clip_max=6,
        seed=7,
        near_duplicate_jaccard=0.9,
    )
    again = sample_pairs(
        world.pool.candidates,
        world.audits,
        per_clip_min=3,
        per_clip_max=6,
        seed=7,
        near_duplicate_jaccard=0.9,
    )
    assert pairs == again
    per_clip: dict[str, int] = {}
    for pair in pairs:
        per_clip[pair.clip_id] = per_clip.get(pair.clip_id, 0) + 1
    assert all(3 <= count <= 6 for count in per_clip.values())


def test_length_confounds_are_flagged_not_silently_kept(world: PreferenceWorld) -> None:
    flagged = [pair for pair in world.pool.pairs if pair.features.length_confounded]
    for pair in flagged:
        assert pair.features.length_difference >= 5
