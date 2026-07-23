"""Trainer invariants: frozen references, weighting, diagnostics (PRD 14.3, 31)."""

from __future__ import annotations

import pytest
import torch

from dpo.data.derive_pairs import derive_pair_strict
from dpo.models.base import MediaBatch
from dpo.models.tiny import TinyAdapter, synthetic_media
from dpo.objectives.dpo import DPOObjective
from dpo.trainers.callbacks import REQUIRED_PREFERENCE_KEYS
from dpo.trainers.preference_trainer import (
    PreferenceTrainer,
    TrainerConfig,
    TrainerError,
    build_pair_batches,
    precompute_reference_logps,
)
from tests.conftest import PreferenceWorld

MEDIA_DIM = 16


def _media_provider(track: str, clip_ids: list[str]) -> MediaBatch:
    return synthetic_media(track, clip_ids, media_dim=MEDIA_DIM)


def _strict_rows(world: PreferenceWorld) -> list:
    return list(
        derive_pair_strict(
            world.aggregates,
            world.pool,
            world.manifest,
            split="train",
            min_agreement=0.6,
            min_strength=2.0,
        )
    )


def _adapter(world: PreferenceWorld) -> TinyAdapter:
    return TinyAdapter(
        track="visual",
        prompt=world.contract.tracks["visual"].prompt,
        media_dim=MEDIA_DIM,
        seed=0,
    )


def test_reference_receives_no_gradients_and_never_changes(world: PreferenceWorld) -> None:
    rows = _strict_rows(world)
    policy = _adapter(world)
    reference = policy.clone_frozen()
    reference_signature = reference.state_signature()
    batches = build_pair_batches(rows, _media_provider, batch_size=4)
    trainer = PreferenceTrainer(
        policy,
        DPOObjective(beta=0.1),
        TrainerConfig(learning_rate=0.01, total_steps=4, max_grad_norm=1.0, seed=1),
        reference=reference,
    )
    before = policy.state_signature()
    trainer.train(batches)
    assert policy.state_signature() != before
    assert reference.state_signature() == reference_signature
    for parameter in reference.trainable_parameters():
        assert not parameter.requires_grad
        assert parameter.grad is None


def test_precomputed_reference_logps_match_the_live_reference(world: PreferenceWorld) -> None:
    rows = _strict_rows(world)
    policy = _adapter(world)
    reference = policy.clone_frozen()
    batches = build_pair_batches(rows, _media_provider, batch_size=4)
    table = precompute_reference_logps(reference, batches)
    live = PreferenceTrainer(
        policy.clone_trainable(),
        DPOObjective(beta=0.1),
        TrainerConfig(learning_rate=0.01, total_steps=1, max_grad_norm=1.0, seed=1),
        reference=reference,
    )
    precomputed = PreferenceTrainer(
        policy.clone_trainable(),
        DPOObjective(beta=0.1),
        TrainerConfig(learning_rate=0.01, total_steps=1, max_grad_norm=1.0, seed=1),
        reference_logps=table,
    )
    live_batch = live.preference_batch(batches[0])
    precomputed_batch = precomputed.preference_batch(batches[0])
    assert torch.allclose(live_batch.ref_chosen_logps, precomputed_batch.ref_chosen_logps, atol=1e-5)
    assert torch.allclose(live_batch.ref_rejected_logps, precomputed_batch.ref_rejected_logps, atol=1e-5)


def test_initial_margins_are_zero_when_policy_equals_reference(world: PreferenceWorld) -> None:
    rows = _strict_rows(world)
    policy = _adapter(world)
    reference = policy.clone_frozen()
    batches = build_pair_batches(rows, _media_provider, batch_size=4)
    trainer = PreferenceTrainer(
        policy.clone_trainable(),
        DPOObjective(beta=0.1),
        TrainerConfig(learning_rate=0.01, total_steps=1, max_grad_norm=1.0, seed=1),
        reference=reference,
    )
    batch = trainer.preference_batch(batches[0])
    margin = (batch.policy_chosen_logps - batch.ref_chosen_logps) - (
        batch.policy_rejected_logps - batch.ref_rejected_logps
    )
    assert torch.allclose(margin, torch.zeros_like(margin), atol=1e-4)


def test_sample_weights_flow_from_rows_to_the_objective(world: PreferenceWorld) -> None:
    rows = _strict_rows(world)
    batches = build_pair_batches(rows, _media_provider, batch_size=len(rows))
    weights = batches[0].sample_weights
    assert weights is not None
    assert weights.tolist() == [row.weight for row in rows]


def test_step_diagnostics_carry_the_full_shared_schema(world: PreferenceWorld) -> None:
    rows = _strict_rows(world)
    policy = _adapter(world)
    trainer = PreferenceTrainer(
        policy.clone_trainable(),
        DPOObjective(beta=0.1),
        TrainerConfig(learning_rate=0.01, total_steps=1, max_grad_norm=1.0, seed=1),
        reference=policy.clone_frozen(),
    )
    batches = build_pair_batches(rows, _media_provider, batch_size=4)
    trainer.step(batches[0], 0)
    assert set(trainer.log.steps[0]) >= REQUIRED_PREFERENCE_KEYS


def test_trainer_requires_exactly_one_reference_source(world: PreferenceWorld) -> None:
    policy = _adapter(world)
    with pytest.raises(TrainerError, match="exactly one"):
        PreferenceTrainer(
            policy.clone_trainable(),
            DPOObjective(beta=0.1),
            TrainerConfig(learning_rate=0.01, total_steps=1, max_grad_norm=1.0, seed=1),
        )
    with pytest.raises(TrainerError, match="frozen copy"):
        live = policy.clone_trainable()
        PreferenceTrainer(
            live,
            DPOObjective(beta=0.1),
            TrainerConfig(learning_rate=0.01, total_steps=1, max_grad_norm=1.0, seed=1),
            reference=live,
        )
