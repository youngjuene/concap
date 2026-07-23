"""Identity Preference Optimization (Azar et al., 2023). PRD section 15.

Paper notation: the IPO loss is (h_theta - 1/(2*tau))^2 where tau is the
regularization temperature. This module keeps the PRD/DPO parameter name
``beta`` for that temperature, so ``target_margin = 1 / (2 * beta)``. The
gradient with respect to h vanishes at the target margin, which the unit test
asserts directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from dpo.objectives.base import (
    LossOutput,
    PreferenceBatch,
    batch_weights,
    preference_margin,
    require_positive,
    shared_diagnostics,
    weighted_mean,
)


@dataclass(frozen=True)
class IPOObjective:
    beta: float

    def __post_init__(self) -> None:
        require_positive(self.beta, "ipo beta")

    @property
    def target_margin(self) -> float:
        return 1.0 / (2.0 * self.beta)

    def __call__(self, batch: PreferenceBatch, global_step: int, total_steps: int) -> LossOutput:
        del global_step, total_steps
        margin = preference_margin(batch)
        per_example = (margin - self.target_margin) ** 2
        weights = batch_weights(batch)
        loss = weighted_mean(per_example, weights)
        diagnostics = shared_diagnostics(batch, self.beta * margin, per_example)
        diagnostics["objective"] = "ipo"
        diagnostics["beta"] = self.beta
        diagnostics["target_margin"] = self.target_margin
        return LossOutput(loss=loss, per_example_loss=per_example, margin=margin, diagnostics=diagnostics)
