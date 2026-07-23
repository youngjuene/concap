"""Objective unit tests (PRD section 32.1)."""

from __future__ import annotations

import math

import pytest
import torch

from dpo.objectives import OBJECTIVES, build_objective
from dpo.objectives.base import ObjectiveError, PreferenceBatch, preference_margin
from dpo.objectives.cdpo import CDPOObjective
from dpo.objectives.dpo import DPOObjective
from dpo.objectives.drdpo import DrDPOObjective
from dpo.objectives.ipo import IPOObjective
from dpo.objectives.rdpo import RDPOObjective
from dpo.objectives.sft import SFTObjective
from dpo.objectives.wdpo import WDPOObjective


def make_batch(n: int = 8, *, seed: int = 0, weights: torch.Tensor | None = None) -> PreferenceBatch:
    generator = torch.Generator().manual_seed(seed)
    return PreferenceBatch(
        policy_chosen_logps=torch.randn(n, generator=generator, requires_grad=True),
        policy_rejected_logps=torch.randn(n, generator=generator, requires_grad=True),
        ref_chosen_logps=torch.randn(n, generator=generator),
        ref_rejected_logps=torch.randn(n, generator=generator),
        sample_weights=weights,
    )


def swapped(batch: PreferenceBatch) -> PreferenceBatch:
    return PreferenceBatch(
        policy_chosen_logps=batch.policy_rejected_logps,
        policy_rejected_logps=batch.policy_chosen_logps,
        ref_chosen_logps=batch.ref_rejected_logps,
        ref_rejected_logps=batch.ref_chosen_logps,
        sample_weights=batch.sample_weights,
    )


class TestDPO:
    def test_swap_reverses_preference_direction(self) -> None:
        batch = make_batch()
        out = DPOObjective(beta=0.2)(batch, 0, 1)
        out_swapped = DPOObjective(beta=0.2)(swapped(batch), 0, 1)
        assert torch.allclose(out_swapped.margin, -out.margin)
        accuracy = float(str(out.diagnostics["preference_accuracy"]))
        swapped_accuracy = float(str(out_swapped.diagnostics["preference_accuracy"]))
        assert accuracy + swapped_accuracy == pytest.approx(1.0)

    def test_identical_policy_and_reference_gives_zero_margin_log2_loss(self) -> None:
        chosen = torch.randn(6)
        rejected = torch.randn(6)
        batch = PreferenceBatch(
            policy_chosen_logps=chosen.clone().requires_grad_(),
            policy_rejected_logps=rejected.clone().requires_grad_(),
            ref_chosen_logps=chosen,
            ref_rejected_logps=rejected,
        )
        out = DPOObjective(beta=0.1)(batch, 0, 1)
        assert torch.allclose(out.margin, torch.zeros(6))
        assert float(out.loss.detach()) == pytest.approx(math.log(2.0), rel=1e-6)

    def test_sample_weighting_is_a_weighted_mean(self) -> None:
        weights = torch.tensor([2.0, 0.0, 1.0, 1.0])
        batch = make_batch(4, weights=weights)
        out = DPOObjective(beta=0.1)(batch, 0, 1)
        expected = float((out.per_example_loss.detach() * weights).sum() / weights.sum())
        assert float(out.loss.detach()) == pytest.approx(expected, rel=1e-6)

    def test_mean_objectives_reduce_consistently_across_sub_batches(self) -> None:
        # The distributed-reduction contract for mean-type objectives: the
        # full-batch loss equals the size-weighted combination of sub-batch
        # losses, so gradient-accumulated or multi-worker averaging is exact.
        batch = make_batch(8)
        objective = DPOObjective(beta=0.1)
        full = objective(batch, 0, 1)
        halves = []
        for a, b in ((0, 4), (4, 8)):
            halves.append(
                objective(
                    PreferenceBatch(
                        policy_chosen_logps=batch.policy_chosen_logps[a:b],
                        policy_rejected_logps=batch.policy_rejected_logps[a:b],
                        ref_chosen_logps=batch.ref_chosen_logps[a:b],
                        ref_rejected_logps=batch.ref_rejected_logps[a:b],
                    ),
                    0,
                    1,
                )
            )
        combined = 0.5 * float(halves[0].loss.detach()) + 0.5 * float(halves[1].loss.detach())
        assert float(full.loss.detach()) == pytest.approx(combined, rel=1e-6)


class TestIPO:
    def test_gradient_vanishes_at_target_margin(self) -> None:
        beta = 0.25
        target = 1.0 / (2.0 * beta)
        chosen = torch.full((1,), target, requires_grad=True)
        batch = PreferenceBatch(
            policy_chosen_logps=chosen,
            policy_rejected_logps=torch.zeros(1),
            ref_chosen_logps=torch.zeros(1),
            ref_rejected_logps=torch.zeros(1),
        )
        out = IPOObjective(beta=beta)(batch, 0, 1)
        out.loss.backward()
        assert chosen.grad is not None
        assert abs(float(chosen.grad[0])) < 1e-8

    def test_gradient_magnitude_decreases_toward_target(self) -> None:
        beta = 0.25
        target = 1.0 / (2.0 * beta)

        def gradient_at(margin: float) -> float:
            chosen = torch.full((1,), margin, requires_grad=True)
            batch = PreferenceBatch(
                policy_chosen_logps=chosen,
                policy_rejected_logps=torch.zeros(1),
                ref_chosen_logps=torch.zeros(1),
                ref_rejected_logps=torch.zeros(1),
            )
            IPOObjective(beta=beta)(batch, 0, 1).loss.backward()
            assert chosen.grad is not None
            return abs(float(chosen.grad[0]))

        assert gradient_at(target * 0.9) < gradient_at(target * 0.5) < gradient_at(0.0)


class TestCDPO:
    def test_epsilon_zero_equals_dpo(self) -> None:
        batch = make_batch()
        assert torch.allclose(
            CDPOObjective(beta=0.1, epsilon=0.0)(batch, 0, 1).loss,
            DPOObjective(beta=0.1)(batch, 0, 1).loss,
        )

    def test_epsilon_bound(self) -> None:
        with pytest.raises(ObjectiveError):
            CDPOObjective(beta=0.1, epsilon=0.5)

    def test_smoothing_mixes_forward_and_reverse(self) -> None:
        batch = make_batch()
        out = CDPOObjective(beta=0.1, epsilon=0.2)(batch, 0, 1)
        forward = DPOObjective(beta=0.1)(batch, 0, 1).per_example_loss
        reverse = DPOObjective(beta=0.1)(swapped(batch), 0, 1).per_example_loss
        assert torch.allclose(out.per_example_loss, 0.8 * forward + 0.2 * reverse, atol=1e-6)


class TestRDPO:
    def test_epsilon_zero_equals_dpo(self) -> None:
        batch = make_batch()
        assert torch.allclose(
            RDPOObjective(beta=0.1, epsilon=0.0)(batch, 0, 1).loss,
            DPOObjective(beta=0.1)(batch, 0, 1).loss,
        )

    def test_epsilon_upper_bound_guard(self) -> None:
        with pytest.raises(ObjectiveError):
            RDPOObjective(beta=0.1, epsilon=0.5)
        with pytest.raises(ObjectiveError):
            RDPOObjective(beta=0.1, epsilon=0.7)

    def test_rdpo_is_not_label_smoothing(self) -> None:
        # The debiased loss subtracts the reverse term; conservative smoothing
        # adds it. At the same epsilon they must disagree.
        batch = make_batch()
        rdpo = RDPOObjective(beta=0.1, epsilon=0.2)(batch, 0, 1)
        cdpo = CDPOObjective(beta=0.1, epsilon=0.2)(batch, 0, 1)
        assert not torch.allclose(rdpo.loss, cdpo.loss.detach())

    def test_unbiasedness_identity_on_flipped_mixture(self) -> None:
        # E_flip[rdpo(observed)] = dpo(clean): (1-e)*L(z) - e*L(-z) applied to
        # a clean z and its flip recovers the clean logistic loss exactly.
        batch = make_batch()
        epsilon = 0.3
        clean = DPOObjective(beta=0.1)(batch, 0, 1).per_example_loss
        observed = RDPOObjective(beta=0.1, epsilon=epsilon)(batch, 0, 1).per_example_loss
        flipped = RDPOObjective(beta=0.1, epsilon=epsilon)(swapped(batch), 0, 1).per_example_loss
        mixture = (1.0 - epsilon) * observed + epsilon * flipped
        assert torch.allclose(mixture, clean, atol=1e-5)


class TestDrDPO:
    def test_reference_equivalence_on_hand_computed_example(self) -> None:
        # Uniform batch of two losses l1, l2: the aggregate must equal
        # -b' * log(0.5 * (exp(-l1/b') + exp(-l2/b'))).
        beta_prime = 0.5
        chosen = torch.tensor([1.0, -1.0], requires_grad=True)
        batch = PreferenceBatch(
            policy_chosen_logps=chosen,
            policy_rejected_logps=torch.zeros(2),
            ref_chosen_logps=torch.zeros(2),
            ref_rejected_logps=torch.zeros(2),
        )
        out = DrDPOObjective(beta=1.0, beta_prime=beta_prime)(batch, 0, 1)
        losses = [-math.log(1.0 / (1.0 + math.exp(-z))) for z in (1.0, -1.0)]
        expected = -beta_prime * math.log(0.5 * sum(math.exp(-loss / beta_prime) for loss in losses))
        assert float(out.loss.detach()) == pytest.approx(expected, rel=1e-6)

    def test_logsumexp_stability_on_extreme_losses(self) -> None:
        chosen = torch.tensor([2000.0, -2000.0], requires_grad=True)
        batch = PreferenceBatch(
            policy_chosen_logps=chosen,
            policy_rejected_logps=torch.zeros(2),
            ref_chosen_logps=torch.zeros(2),
            ref_rejected_logps=torch.zeros(2),
        )
        out = DrDPOObjective(beta=1.0, beta_prime=0.5)(batch, 0, 1)
        assert bool(torch.isfinite(out.loss.detach()))
        out.loss.backward()
        assert chosen.grad is not None
        assert bool(torch.isfinite(chosen.grad).all())

    def test_batch_composition_is_load_bearing(self) -> None:
        # Dr.DPO aggregates at the batch level; splitting a batch changes the
        # objective. This pins the batch-level semantics (and is why batch
        # composition must be deterministic and world_size is 1).
        batch = make_batch(8)
        objective = DrDPOObjective(beta=0.1, beta_prime=0.5)
        full = float(objective(batch, 0, 1).loss.detach())
        halves = []
        for a, b in ((0, 4), (4, 8)):
            halves.append(
                float(
                    objective(
                        PreferenceBatch(
                            policy_chosen_logps=batch.policy_chosen_logps[a:b],
                            policy_rejected_logps=batch.policy_rejected_logps[a:b],
                            ref_chosen_logps=batch.ref_chosen_logps[a:b],
                            ref_rejected_logps=batch.ref_rejected_logps[a:b],
                        ),
                        0,
                        1,
                    ).loss.detach()
                )
            )
        assert full != pytest.approx(0.5 * halves[0] + 0.5 * halves[1], rel=1e-9)

    def test_explicit_sample_weights_reweight_the_aggregate(self) -> None:
        batch_uniform = make_batch(4, seed=3)
        weights = torch.tensor([1.0, 1.0, 1.0, 3.0])
        batch_weighted = PreferenceBatch(
            policy_chosen_logps=batch_uniform.policy_chosen_logps,
            policy_rejected_logps=batch_uniform.policy_rejected_logps,
            ref_chosen_logps=batch_uniform.ref_chosen_logps,
            ref_rejected_logps=batch_uniform.ref_rejected_logps,
            sample_weights=weights,
        )
        objective = DrDPOObjective(beta=0.1, beta_prime=0.5)
        assert float(objective(batch_weighted, 0, 1).loss.detach()) != pytest.approx(
            float(objective(batch_uniform, 0, 1).loss.detach()), rel=1e-9
        )


class TestWDPO:
    def test_both_stages_disabled_equals_dpo(self) -> None:
        batch = make_batch(16)
        wdpo = WDPOObjective(beta=0.1, enable_correction=False, enable_winsorization=False)
        assert torch.allclose(wdpo(batch, 50, 100).loss, DPOObjective(beta=0.1)(batch, 0, 1).loss.detach())

    def test_warmup_disables_correction(self) -> None:
        batch = make_batch(16)
        objective = WDPOObjective(beta=0.1, flip_warmup_ratio=0.5, flip_budget=0.5)
        during_warmup = objective(batch, 10, 100)
        assert during_warmup.diagnostics["wdpo_warmup_active"] is True
        assert float(str(during_warmup.diagnostics["wdpo_corrected_pair_ratio"])) == 0.0
        after_warmup = objective(batch, 60, 100)
        assert after_warmup.diagnostics["wdpo_warmup_active"] is False

    def test_correction_budget_bounds_corrected_pairs(self) -> None:
        batch = make_batch(20, seed=5)
        objective = WDPOObjective(beta=1.0, flip_warmup_ratio=0.0, flip_budget=0.1)
        out = objective(batch, 10, 100)
        ratio = float(str(out.diagnostics["wdpo_corrected_pair_ratio"]))
        assert ratio <= 0.1 + 1e-9

    def test_correction_only_and_winsorization_only_ablations_differ(self) -> None:
        batch = make_batch(32, seed=9)
        base = DPOObjective(beta=1.0)(batch, 0, 1)
        correction_only = WDPOObjective(
            beta=1.0, flip_warmup_ratio=0.0, flip_budget=0.5, enable_winsorization=False
        )(batch, 10, 100)
        winsor_only = WDPOObjective(
            beta=1.0, enable_correction=False, tail_quantile=0.5, max_tail_cap=0.5, cap_floor=0.2
        )(batch, 10, 100)
        assert float(correction_only.loss.detach()) < float(base.loss.detach())
        assert float(winsor_only.loss.detach()) < float(base.loss.detach())
        assert float(correction_only.loss.detach()) != pytest.approx(
            float(winsor_only.loss.detach()), rel=1e-9
        )

    def test_labels_are_never_fully_flipped(self) -> None:
        # The correction weight is capped at 0.5: the strongest correction is
        # an equal mix of observed and swapped losses, never a full label flip.
        batch = make_batch(16, seed=11)
        out = WDPOObjective(beta=1.0, flip_warmup_ratio=0.0, flip_budget=1.0, enable_winsorization=False)(
            batch, 10, 100
        )
        mixed = out.per_example_loss.detach()
        observed = DPOObjective(beta=1.0)(batch, 0, 1).per_example_loss.detach()
        reverse = DPOObjective(beta=1.0)(swapped(batch), 0, 1).per_example_loss.detach()
        equal_mix = 0.5 * observed + 0.5 * reverse
        # With lambda in [0, 0.5], every mixed loss lies between the observed
        # loss and the equal mix; it can never cross to the fully-swapped side.
        lower = torch.minimum(observed, equal_mix)
        upper = torch.maximum(observed, equal_mix)
        assert bool(((mixed >= lower - 1e-6) & (mixed <= upper + 1e-6)).all())

    def test_tail_statistics_are_detached(self) -> None:
        chosen = torch.randn(16, requires_grad=True)
        batch = PreferenceBatch(
            policy_chosen_logps=chosen,
            policy_rejected_logps=torch.zeros(16),
            ref_chosen_logps=torch.zeros(16),
            ref_rejected_logps=torch.zeros(16),
        )
        out = WDPOObjective(beta=1.0, enable_correction=False)(batch, 10, 100)
        out.loss.backward()
        assert chosen.grad is not None
        assert bool(torch.isfinite(chosen.grad).all())


class TestSharedContracts:
    def test_reference_logps_must_be_detached(self) -> None:
        with pytest.raises(ObjectiveError, match="reference"):
            PreferenceBatch(
                policy_chosen_logps=torch.randn(4, requires_grad=True),
                policy_rejected_logps=torch.randn(4, requires_grad=True),
                ref_chosen_logps=torch.randn(4, requires_grad=True),
                ref_rejected_logps=torch.randn(4),
            )

    def test_non_finite_inputs_are_rejected(self) -> None:
        with pytest.raises(ObjectiveError, match="non-finite"):
            PreferenceBatch(
                policy_chosen_logps=torch.tensor([float("nan")]),
                policy_rejected_logps=torch.zeros(1),
                ref_chosen_logps=torch.zeros(1),
                ref_rejected_logps=torch.zeros(1),
            )

    def test_margin_definition(self) -> None:
        batch = make_batch()
        margin = preference_margin(batch)
        manual = (batch.policy_chosen_logps - batch.ref_chosen_logps) - (
            batch.policy_rejected_logps - batch.ref_rejected_logps
        )
        assert torch.allclose(margin, manual)

    def test_registry_and_builder_cover_every_preference_objective(self) -> None:
        assert set(OBJECTIVES) == {"dpo", "ipo", "cdpo", "rdpo", "drdpo", "wdpo"}
        tables: dict[str, dict[str, object]] = {
            "dpo": {"objective": "dpo", "beta": 0.1},
            "ipo": {"objective": "ipo", "beta": 0.1},
            "cdpo": {"objective": "cdpo", "beta": 0.1, "epsilon": 0.1},
            "rdpo": {
                "objective": "rdpo",
                "beta": 0.1,
                "epsilon": 0.1,
                "epsilon_mode": "estimated",
            },
            "drdpo": {"objective": "drdpo", "beta": 0.1, "beta_prime": 0.5},
            "wdpo": {
                "objective": "wdpo",
                "beta": 0.1,
                "flip_warmup_ratio": 0.1,
                "flip_budget": 0.05,
                "tail_quantile": 0.8,
                "max_tail_cap": 0.5,
                "cap_floor": 0.0,
                "paper_revision": "arxiv_2603.07211v1",
                "code_commit": "",
            },
        }
        batch = make_batch()
        for name, table in tables.items():
            objective = build_objective(table)
            out = objective(batch, 1, 10)
            assert out.diagnostics["objective"] == name


class TestSFT:
    def test_sft_loss_is_weighted_negative_logp(self) -> None:
        logps = torch.tensor([-1.0, -2.0, -3.0], requires_grad=True)
        weights = torch.tensor([1.0, 1.0, 2.0])
        out = SFTObjective().loss(logps, weights)
        assert float(out.loss.detach()) == pytest.approx((1.0 + 2.0 + 2 * 3.0) / 4.0)

    def test_sft_rejects_zero_weight_sum(self) -> None:
        with pytest.raises(ObjectiveError):
            SFTObjective().loss(torch.tensor([-1.0]), torch.tensor([0.0]))
