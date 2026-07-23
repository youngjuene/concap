"""Direct Preference Optimization (Rafailov et al., 2023). PRD section 14."""

from __future__ import annotations

from dataclasses import dataclass

from dpo.objectives.base import (
    LossOutput,
    PreferenceBatch,
    batch_weights,
    logistic_loss,
    preference_margin,
    require_positive,
    shared_diagnostics,
    weighted_mean,
)


@dataclass(frozen=True)
class DPOObjective:
    beta: float

    def __post_init__(self) -> None:
        require_positive(self.beta, "dpo beta")

    def __call__(self, batch: PreferenceBatch, global_step: int, total_steps: int) -> LossOutput:
        del global_step, total_steps
        margin = preference_margin(batch)
        z = self.beta * margin
        per_example = logistic_loss(z)
        weights = batch_weights(batch)
        loss = weighted_mean(per_example, weights)
        diagnostics = shared_diagnostics(batch, z, per_example)
        diagnostics["objective"] = "dpo"
        diagnostics["beta"] = self.beta
        return LossOutput(loss=loss, per_example_loss=per_example, margin=margin, diagnostics=diagnostics)
