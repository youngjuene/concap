"""Provably robust DPO under symmetric label flips. PRD section 17.

Specification provenance: Chowdhury, Kini & Natarajan, "Provably Robust DPO:
Aligning Language Models with Noisy Feedback" (ICML 2024, arXiv:2403.00409).
The unbiased loss under a symmetric flip rate eps is

    [(1 - eps) * l(z) - eps * l(-z)] / (1 - 2*eps),   l(z) = -log sigmoid(z).

This is NOT ordinary label smoothing: the reverse term is subtracted, and the
normalization debiases the estimator. At eps=0 it reduces exactly to DPO. The
divergent eps -> 0.5 limit is excluded by construction; per-example losses can
legitimately be negative.
"""

from __future__ import annotations

from dataclasses import dataclass

from dpo.objectives.base import (
    LossOutput,
    ObjectiveError,
    PreferenceBatch,
    batch_weights,
    logistic_loss,
    preference_margin,
    require_positive,
    shared_diagnostics,
    weighted_mean,
)

RDPO_SPECIFICATION = "chowdhury-2024-provably-robust-dpo/arXiv-2403.00409"
RDPO_EPSILON_MODES = ("estimated", "validation_tuned", "known_synthetic")


@dataclass(frozen=True)
class RDPOObjective:
    beta: float
    epsilon: float
    epsilon_mode: str = "estimated"

    def __post_init__(self) -> None:
        require_positive(self.beta, "rdpo beta")
        if not 0.0 <= self.epsilon < 0.5:
            raise ObjectiveError("rdpo epsilon must satisfy 0 <= epsilon < 0.5")
        if self.epsilon_mode not in RDPO_EPSILON_MODES:
            raise ObjectiveError(f"rdpo epsilon_mode must be one of {sorted(RDPO_EPSILON_MODES)}")

    def __call__(self, batch: PreferenceBatch, global_step: int, total_steps: int) -> LossOutput:
        del global_step, total_steps
        margin = preference_margin(batch)
        z = self.beta * margin
        forward = logistic_loss(z)
        reverse = logistic_loss(-z)
        per_example = ((1.0 - self.epsilon) * forward - self.epsilon * reverse) / (1.0 - 2.0 * self.epsilon)
        weights = batch_weights(batch)
        loss = weighted_mean(per_example, weights)
        diagnostics = shared_diagnostics(batch, z, per_example)
        diagnostics["objective"] = "rdpo"
        diagnostics["beta"] = self.beta
        diagnostics["epsilon"] = self.epsilon
        diagnostics["epsilon_mode"] = self.epsilon_mode
        diagnostics["specification"] = RDPO_SPECIFICATION
        diagnostics["forward_loss_mean"] = float(forward.detach().mean())
        diagnostics["reverse_loss_mean"] = float(reverse.detach().mean())
        return LossOutput(loss=loss, per_example_loss=per_example, margin=margin, diagnostics=diagnostics)
