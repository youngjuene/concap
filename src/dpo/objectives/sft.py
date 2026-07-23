"""Chosen-only supervised fine-tuning loss. PRD section 13.

SFT is not a preference objective: it reads only the policy's completion
log-probabilities of positively endorsed targets. It shares the
PreferenceBatch carrier (chosen slots) so SFT and SFT_DPO reuse the same
trainer plumbing and the same shared log-probability implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from dpo.objectives.base import LossOutput, ObjectiveError


@dataclass(frozen=True)
class SFTObjective:
    def loss(self, target_logps: Tensor, weights: Tensor | None = None) -> LossOutput:
        if target_logps.dim() != 1 or target_logps.shape[0] == 0:
            raise ObjectiveError("sft target log-probabilities must be non-empty 1-D [batch]")
        if not bool(torch.isfinite(target_logps.detach()).all()):
            raise ObjectiveError("sft target log-probabilities contain non-finite values")
        if weights is None:
            weights = torch.ones_like(target_logps)
        if weights.shape != target_logps.shape:
            raise ObjectiveError("sft weights shape does not match the batch")
        if float(weights.detach().sum()) <= 0.0:
            raise ObjectiveError("sft weights must not sum to zero")
        per_example = -target_logps
        loss = (per_example * weights).sum() / weights.sum()
        diagnostics: dict[str, object] = {
            "objective": "sft",
            "target_logp_mean": float(target_logps.detach().mean()),
            "per_example_loss_mean": float(per_example.detach().mean()),
            "sample_weight_sum": float(weights.detach().sum()),
            "nan_count": int(torch.isnan(per_example.detach()).sum()),
            "inf_count": int(torch.isinf(per_example.detach()).sum()),
            "batch_size": int(target_logps.shape[0]),
        }
        return LossOutput(
            loss=loss,
            per_example_loss=per_example,
            margin=torch.zeros_like(per_example),
            diagnostics=diagnostics,
        )
