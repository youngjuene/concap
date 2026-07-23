"""Conservative DPO: label-smoothed pairwise preference. PRD section 16.

Specification provenance: Eric Mitchell, "A note on DPO with noisy preferences
and relationship to IPO", technical note v1.1 (2023) — the ``conservative DPO``
loss (1-eps) * -log sigmoid(z) + eps * -log sigmoid(-z) with 0 <= eps < 0.5.
At eps=0 this is exactly standard DPO, which the unit test asserts.
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

CDPO_SPECIFICATION = "mitchell-2023-noisy-preferences-note/v1.1"


@dataclass(frozen=True)
class CDPOObjective:
    beta: float
    epsilon: float

    def __post_init__(self) -> None:
        require_positive(self.beta, "cdpo beta")
        if not 0.0 <= self.epsilon < 0.5:
            raise ObjectiveError("cdpo epsilon must satisfy 0 <= epsilon < 0.5")

    def __call__(self, batch: PreferenceBatch, global_step: int, total_steps: int) -> LossOutput:
        del global_step, total_steps
        margin = preference_margin(batch)
        z = self.beta * margin
        forward = logistic_loss(z)
        reverse = logistic_loss(-z)
        per_example = (1.0 - self.epsilon) * forward + self.epsilon * reverse
        weights = batch_weights(batch)
        loss = weighted_mean(per_example, weights)
        diagnostics = shared_diagnostics(batch, z, per_example)
        diagnostics["objective"] = "cdpo"
        diagnostics["beta"] = self.beta
        diagnostics["epsilon"] = self.epsilon
        diagnostics["specification"] = CDPO_SPECIFICATION
        diagnostics["forward_loss_mean"] = float(forward.detach().mean())
        diagnostics["reverse_loss_mean"] = float(reverse.detach().mean())
        return LossOutput(loss=loss, per_example_loss=per_example, margin=margin, diagnostics=diagnostics)
