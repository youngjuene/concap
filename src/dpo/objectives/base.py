"""The shared training interface every preference objective implements.

`PreferenceBatch` carries completion-level sequence log-probabilities from one
shared implementation (`dpo.models.logprob`); objectives are pure functions
of those tensors plus the step counters, so DPO, IPO, cDPO, rDPO, Dr.DPO, and
wDPO are directly comparable and unit-testable without a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


class ObjectiveError(ValueError):
    """Raised when a preference batch or objective configuration is invalid."""


@dataclass
class PreferenceBatch:
    policy_chosen_logps: Tensor
    policy_rejected_logps: Tensor
    ref_chosen_logps: Tensor
    ref_rejected_logps: Tensor
    sample_weights: Tensor | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tensors = {
            "policy_chosen_logps": self.policy_chosen_logps,
            "policy_rejected_logps": self.policy_rejected_logps,
            "ref_chosen_logps": self.ref_chosen_logps,
            "ref_rejected_logps": self.ref_rejected_logps,
        }
        shape = self.policy_chosen_logps.shape
        if len(shape) != 1 or shape[0] == 0:
            raise ObjectiveError("preference batch tensors must be non-empty 1-D [batch]")
        for name, tensor in tensors.items():
            if tensor.shape != shape:
                raise ObjectiveError(f"{name} shape {tuple(tensor.shape)} != {tuple(shape)}")
            if not bool(torch.isfinite(tensor.detach()).all()):
                raise ObjectiveError(f"{name} contains non-finite values")
        for name in ("ref_chosen_logps", "ref_rejected_logps"):
            if tensors[name].requires_grad:
                raise ObjectiveError(f"{name} must be detached; the reference receives no gradients")
        if self.sample_weights is not None:
            if self.sample_weights.shape != shape:
                raise ObjectiveError("sample_weights shape does not match the batch")
            if not bool(torch.isfinite(self.sample_weights.detach()).all()):
                raise ObjectiveError("sample_weights contain non-finite values")
            if bool((self.sample_weights.detach() < 0).any()):
                raise ObjectiveError("sample_weights must be non-negative")
            if float(self.sample_weights.detach().sum()) <= 0.0:
                raise ObjectiveError("sample_weights must not sum to zero")

    @property
    def batch_size(self) -> int:
        return int(self.policy_chosen_logps.shape[0])


@dataclass
class LossOutput:
    loss: Tensor
    per_example_loss: Tensor
    margin: Tensor
    diagnostics: dict[str, object]


@runtime_checkable
class PreferenceObjective(Protocol):
    def __call__(
        self,
        batch: PreferenceBatch,
        global_step: int,
        total_steps: int,
    ) -> LossOutput: ...


def preference_margin(batch: PreferenceBatch) -> Tensor:
    """h_theta = (log pi(y+|x) - log ref(y+|x)) - (log pi(y-|x) - log ref(y-|x))."""
    return (batch.policy_chosen_logps - batch.ref_chosen_logps) - (
        batch.policy_rejected_logps - batch.ref_rejected_logps
    )


def logistic_loss(z: Tensor) -> Tensor:
    """-log sigmoid(z), computed stably as softplus(-z)."""
    return torch.nn.functional.softplus(-z)


def batch_weights(batch: PreferenceBatch) -> Tensor:
    if batch.sample_weights is None:
        return torch.ones_like(batch.policy_chosen_logps)
    return batch.sample_weights


def weighted_mean(per_example: Tensor, weights: Tensor) -> Tensor:
    return (per_example * weights).sum() / weights.sum()


def require_positive(value: float, name: str) -> float:
    if not value > 0:
        raise ObjectiveError(f"{name} must be positive")
    return float(value)


def shared_diagnostics(batch: PreferenceBatch, z: Tensor, per_example: Tensor) -> dict[str, object]:
    """The diagnostics every preference trainer logs, per PRD section 11.4."""
    margin = preference_margin(batch)
    weights = batch_weights(batch)
    detached = per_example.detach()
    return {
        "policy_chosen_logp_mean": float(batch.policy_chosen_logps.detach().mean()),
        "policy_rejected_logp_mean": float(batch.policy_rejected_logps.detach().mean()),
        "ref_chosen_logp_mean": float(batch.ref_chosen_logps.detach().mean()),
        "ref_rejected_logp_mean": float(batch.ref_rejected_logps.detach().mean()),
        "margin_mean": float(margin.detach().mean()),
        "z_mean": float(z.detach().mean()),
        "preference_accuracy": float((margin.detach() > 0).float().mean()),
        "per_example_loss_mean": float(detached.mean()),
        "per_example_loss_max": float(detached.max()),
        "kl_proxy_chosen": float((batch.policy_chosen_logps - batch.ref_chosen_logps).detach().mean()),
        "kl_proxy_rejected": float((batch.policy_rejected_logps - batch.ref_rejected_logps).detach().mean()),
        "sample_weight_sum": float(weights.detach().sum()),
        "nan_count": int(torch.isnan(detached).sum()),
        "inf_count": int(torch.isinf(detached).sum()),
        "batch_size": batch.batch_size,
    }
