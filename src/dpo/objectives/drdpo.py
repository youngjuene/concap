"""Distributionally robust DPO (Dr.DPO). PRD section 18.

Specification provenance: Wu et al., "Towards Robust Alignment of Language
Models: Distributionally Robustifying Direct Preference Optimization" (ICLR
2025, arXiv:2407.07880). Per-pair DPO losses are aggregated with the robust
temperature beta_prime:

    L = -beta_prime * log( (1/B) * sum_i exp(-l_i / beta_prime) )

computed via logsumexp for numerical stability. With explicit sample weights
w_i the uniform 1/B becomes normalized weights p_i = w_i / sum(w), so weighted
and unweighted batches share one code path. Aggregation is over the local
optimizer batch; the canonical pipeline runs world_size=1, so local and global
aggregation coincide (a distributed run must gather losses first, and the test
suite pins this batch-level semantics).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dpo.objectives.base import (
    LossOutput,
    PreferenceBatch,
    batch_weights,
    logistic_loss,
    preference_margin,
    require_positive,
    shared_diagnostics,
)

DRDPO_SPECIFICATION = "wu-2025-drdpo/arXiv-2407.07880"


@dataclass(frozen=True)
class DrDPOObjective:
    beta: float
    beta_prime: float

    def __post_init__(self) -> None:
        require_positive(self.beta, "drdpo beta")
        require_positive(self.beta_prime, "drdpo beta_prime")

    def __call__(self, batch: PreferenceBatch, global_step: int, total_steps: int) -> LossOutput:
        del global_step, total_steps
        margin = preference_margin(batch)
        z = self.beta * margin
        per_example = logistic_loss(z)
        weights = batch_weights(batch)
        log_probabilities = torch.log(weights / weights.sum())
        aggregate = -self.beta_prime * torch.logsumexp(
            log_probabilities - per_example / self.beta_prime, dim=0
        )
        diagnostics = shared_diagnostics(batch, z, per_example)
        effective = torch.softmax(log_probabilities - per_example.detach() / self.beta_prime, dim=0)
        diagnostics["objective"] = "drdpo"
        diagnostics["beta"] = self.beta
        diagnostics["beta_prime"] = self.beta_prime
        diagnostics["specification"] = DRDPO_SPECIFICATION
        diagnostics["robust_aggregate"] = float(aggregate.detach())
        diagnostics["mean_aggregate"] = float(per_example.detach().mean())
        diagnostics["effective_weight_max"] = float(effective.max())
        diagnostics["effective_weight_min"] = float(effective.min())
        return LossOutput(
            loss=aggregate, per_example_loss=per_example, margin=margin, diagnostics=diagnostics
        )
