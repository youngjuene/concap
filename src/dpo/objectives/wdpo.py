"""Winsorized DPO (wDPO). PRD section 19. EXPERIMENTAL.

Specification provenance: Liu et al., "wDPO: Winsorized Direct Preference
Optimization for Robust LLM Alignment", arXiv:2603.07211v1 (2026), preprint.
The pinned revision and code commit are carried in the experiment contract
(``paper_revision`` / ``code_commit``). Where the preprint leaves scalar maps
unspecified, the formulas below are explicit, deterministic placeholders that
must be re-verified against the pinned official code before any confirmatory
run; the golden tests snapshot them so a re-pin is a visible diff, and the
required ablation identity (both stages disabled == DPO) is enforced by test.

Stage I - margin-aware soft label correction:
  observed_i = -log sigmoid(z_i); swapped_i = -log sigmoid(-z_i);
  improvement_i = observed_i - swapped_i (detached). During warm-up the
  correction is disabled. Otherwise at most floor(flip_budget * B) pairs with
  the largest positive improvement receive a sparse partial correction weight
  lambda_i = 0.5 * improvement_i / max(selected improvement), capped at 0.5 so
  a label is never fully flipped, and the losses are mixed:
  mixed_i = (1 - lambda_i) * observed_i + lambda_i * swapped_i.

Stage II - gradient-oriented winsorization:
  tau = quantile(mixed, tail_quantile) (detached); excess_i = relu(mixed_i -
  tau); tail_energy_share = sum(excess) / sum(mixed); adaptive cap strength
  c = clamp(cap_floor + (max_tail_cap - cap_floor) * min(1, 2 *
  tail_energy_share), cap_floor, max_tail_cap), all statistics detached from
  backpropagation; tail losses are soft-capped, not discarded:
  final_i = tau + (1 - c) * (mixed_i - tau) for tail samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

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

WDPO_DEFAULT_PAPER_REVISION = "arxiv_2603.07211v1"
_HISTOGRAM_EDGES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0)


def _histogram(values: Tensor) -> dict[str, int]:
    counts: dict[str, int] = {}
    detached = values.detach()
    for low, high in zip(_HISTOGRAM_EDGES[:-1], _HISTOGRAM_EDGES[1:], strict=True):
        key = f"[{low:.1f},{high:.1f})"
        counts[key] = int(((detached >= low) & (detached < high)).sum())
    return counts


@dataclass(frozen=True)
class WDPOObjective:
    beta: float
    flip_warmup_ratio: float = 0.1
    flip_budget: float = 0.05
    tail_quantile: float = 0.8
    max_tail_cap: float = 0.5
    cap_floor: float = 0.0
    enable_correction: bool = True
    enable_winsorization: bool = True
    paper_revision: str = WDPO_DEFAULT_PAPER_REVISION
    code_commit: str = ""

    def __post_init__(self) -> None:
        require_positive(self.beta, "wdpo beta")
        if not 0.0 <= self.flip_warmup_ratio <= 1.0:
            raise ObjectiveError("wdpo flip_warmup_ratio must be in [0, 1]")
        if not 0.0 <= self.flip_budget <= 1.0:
            raise ObjectiveError("wdpo flip_budget must be in [0, 1]")
        if not 0.0 < self.tail_quantile <= 1.0:
            raise ObjectiveError("wdpo tail_quantile must be in (0, 1]")
        if not 0.0 <= self.cap_floor <= self.max_tail_cap <= 1.0:
            raise ObjectiveError("wdpo caps must satisfy 0 <= cap_floor <= max_tail_cap <= 1")

    def _correction_weights(
        self, observed: Tensor, swapped: Tensor, global_step: int, total_steps: int
    ) -> tuple[Tensor, dict[str, object]]:
        batch_size = observed.shape[0]
        improvement = (observed - swapped).detach()
        warmup_steps = int(self.flip_warmup_ratio * total_steps)
        warmup_active = global_step < warmup_steps
        budget = int(self.flip_budget * batch_size)
        weights = torch.zeros_like(improvement)
        corrected = 0
        if self.enable_correction and not warmup_active and budget > 0:
            positive = improvement > 0
            if bool(positive.any()):
                eligible = torch.where(positive, improvement, torch.zeros_like(improvement))
                selected = torch.topk(eligible, k=min(budget, int(positive.sum()))).indices
                peak = float(eligible[selected].max())
                if peak > 0:
                    weights[selected] = torch.clamp(0.5 * eligible[selected] / peak, max=0.5)
                    corrected = int((weights > 0).sum())
        return weights, {
            "wdpo_corrected_pair_ratio": corrected / batch_size,
            "wdpo_correction_weight_histogram": _histogram(weights),
            "wdpo_correction_weight_mean": float(weights.mean()),
            "wdpo_warmup_active": warmup_active,
            "wdpo_warmup_steps": warmup_steps,
            "wdpo_budget_pairs": budget,
            "wdpo_budget_utilization": corrected / budget if budget else 0.0,
        }

    def _winsorize(self, mixed: Tensor) -> tuple[Tensor, dict[str, object]]:
        detached = mixed.detach()
        tau = torch.quantile(detached, self.tail_quantile)
        tail_mask = detached > tau
        excess = torch.clamp(detached - tau, min=0.0)
        total = float(detached.sum())
        tail_energy_share = float(excess.sum()) / total if total > 0 else 0.0
        if self.enable_winsorization:
            strength = min(1.0, 2.0 * tail_energy_share)
            cap = min(
                self.max_tail_cap,
                max(self.cap_floor, self.cap_floor + (self.max_tail_cap - self.cap_floor) * strength),
            )
        else:
            cap = 0.0
        capped = torch.where(tail_mask, tau + (1.0 - cap) * (mixed - tau), mixed)
        cap_weights = torch.where(tail_mask, torch.full_like(detached, cap), torch.zeros_like(detached))
        share = detached / total if total > 0 else torch.zeros_like(detached)
        return capped, {
            "wdpo_tail_threshold": float(tau),
            "wdpo_tail_ratio": float(tail_mask.float().mean()),
            "wdpo_cap_strength": cap,
            "wdpo_cap_weight_histogram": _histogram(cap_weights),
            "wdpo_tail_energy_share": tail_energy_share,
            "wdpo_gradient_energy_hhi": float((share**2).sum()),
            "wdpo_top_group_share": float(share.max()),
        }

    def __call__(self, batch: PreferenceBatch, global_step: int, total_steps: int) -> LossOutput:
        if total_steps <= 0:
            raise ObjectiveError("wdpo requires a positive total_steps schedule")
        margin = preference_margin(batch)
        z = self.beta * margin
        observed = logistic_loss(z)
        swapped = logistic_loss(-z)
        correction_weights, correction_diagnostics = self._correction_weights(
            observed, swapped, global_step, total_steps
        )
        mixed = (1.0 - correction_weights) * observed + correction_weights * swapped
        corrected_margin = (1.0 - 2.0 * correction_weights) * margin.detach()
        final, tail_diagnostics = self._winsorize(mixed)
        weights = batch_weights(batch)
        loss = weighted_mean(final, weights)
        diagnostics = shared_diagnostics(batch, z, final)
        diagnostics["objective"] = "wdpo"
        diagnostics["beta"] = self.beta
        diagnostics["paper_revision"] = self.paper_revision
        diagnostics["code_commit"] = self.code_commit
        diagnostics["wdpo_margin_before_correction_mean"] = float(margin.detach().mean())
        diagnostics["wdpo_margin_after_correction_mean"] = float(corrected_margin.mean())
        diagnostics.update(correction_diagnostics)
        diagnostics.update(tail_diagnostics)
        return LossOutput(loss=loss, per_example_loss=final, margin=margin, diagnostics=diagnostics)
