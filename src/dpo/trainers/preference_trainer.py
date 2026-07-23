"""The one preference trainer behind DPO, IPO, CDPO, RDPO, DRDPO, WDPO, and SFT_DPO.

The reference is a frozen adapter (or precomputed log-probabilities); the
trainer re-freezes it structurally and the test suite asserts no gradient can
reach it. Chosen and rejected completions score against the identical media
batch by construction of PreferencePairBatch.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import torch

from dpo.contracts.captions import word_count
from dpo.data.derive_pairs import StrictPair
from dpo.models.base import CompletionBatch, MediaBatch, ModelAdapter, PreferencePairBatch
from dpo.objectives.base import LossOutput, ObjectiveError, PreferenceBatch, PreferenceObjective
from dpo.trainers.callbacks import REQUIRED_PREFERENCE_KEYS, DiagnosticsLog

MediaProvider = Callable[[str, Sequence[str]], MediaBatch]


class TrainerError(RuntimeError):
    """Raised when a training invariant would be violated."""


@dataclass(frozen=True)
class TrainerConfig:
    learning_rate: float
    total_steps: int
    max_grad_norm: float
    seed: int
    gradient_accumulation_steps: int = 1

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.total_steps <= 0 or self.max_grad_norm <= 0:
            raise TrainerError("trainer learning_rate, total_steps, and max_grad_norm must be positive")
        if self.gradient_accumulation_steps < 1:
            raise TrainerError("gradient_accumulation_steps must be >= 1")


def _agreement_bucket(agreement: float) -> str:
    if agreement >= 0.9:
        return "high"
    if agreement >= 0.7:
        return "medium"
    return "low"


def build_pair_batches(
    rows: Sequence[StrictPair],
    media_provider: MediaProvider,
    *,
    batch_size: int,
) -> list[PreferencePairBatch]:
    """Deterministic batches in row order; the caller owns shuffling by seed."""
    if batch_size < 1:
        raise TrainerError("batch_size must be positive")
    batches = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        tracks = {row.track for row in chunk}
        if len(tracks) != 1:
            raise TrainerError("mixed-track batches are rejected")
        track = next(iter(tracks))
        media = media_provider(track, [row.clip_id for row in chunk])
        batches.append(
            PreferencePairBatch(
                track=track,
                pair_ids=tuple(row.pair_id for row in chunk),
                media=media,
                chosen=CompletionBatch(texts=tuple(row.chosen_text for row in chunk)),
                rejected=CompletionBatch(texts=tuple(row.rejected_text for row in chunk)),
                sample_weights=torch.tensor([row.weight for row in chunk], dtype=torch.float32),
                metadata={
                    "difficulty": [row.difficulty for row in chunk],
                    "agreement": [row.agreement for row in chunk],
                    "category": [row.category for row in chunk],
                },
            )
        )
    return batches


def precompute_reference_logps(
    reference: ModelAdapter, batches: Iterable[PreferencePairBatch]
) -> dict[str, tuple[float, float]]:
    """Reference log-probabilities per pair id, computed once, without gradients."""
    table: dict[str, tuple[float, float]] = {}
    with torch.no_grad():
        for batch in batches:
            chosen = reference.completion_logps(batch.media, batch.chosen)
            rejected = reference.completion_logps(batch.media, batch.rejected)
            for index, pair_id in enumerate(batch.pair_ids):
                table[pair_id] = (float(chosen[index]), float(rejected[index]))
    return table


class PreferenceTrainer:
    def __init__(
        self,
        policy: ModelAdapter,
        objective: PreferenceObjective,
        config: TrainerConfig,
        *,
        reference: ModelAdapter | None = None,
        reference_logps: Mapping[str, tuple[float, float]] | None = None,
        log: DiagnosticsLog | None = None,
    ) -> None:
        if (reference is None) == (reference_logps is None):
            raise TrainerError(
                "provide exactly one of a frozen reference adapter or precomputed reference logps"
            )
        if reference is not None:
            if reference is policy:
                raise TrainerError("the reference must be a frozen copy, not the live policy")
            for parameter in reference.trainable_parameters():
                parameter.requires_grad_(False)
        self.policy = policy
        self.reference = reference
        self.reference_logps = dict(reference_logps) if reference_logps is not None else None
        self.objective = objective
        self.config = config
        self.log = log if log is not None else DiagnosticsLog(required=REQUIRED_PREFERENCE_KEYS)
        torch.manual_seed(config.seed)
        parameters = [parameter for parameter in policy.trainable_parameters() if parameter.requires_grad]
        if not parameters:
            raise TrainerError("the policy has no trainable parameters")
        self.optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)

    def _reference_pair_logps(self, batch: PreferencePairBatch) -> tuple[torch.Tensor, torch.Tensor]:
        if self.reference is not None:
            with torch.no_grad():
                chosen = self.reference.completion_logps(batch.media, batch.chosen)
                rejected = self.reference.completion_logps(batch.media, batch.rejected)
            return chosen.detach(), rejected.detach()
        assert self.reference_logps is not None
        try:
            pairs = [self.reference_logps[pair_id] for pair_id in batch.pair_ids]
        except KeyError as exc:
            raise TrainerError(f"pair {exc.args[0]!r} has no precomputed reference logps") from exc
        chosen = torch.tensor([pair[0] for pair in pairs], dtype=torch.float32)
        rejected = torch.tensor([pair[1] for pair in pairs], dtype=torch.float32)
        return chosen, rejected

    def preference_batch(self, batch: PreferencePairBatch) -> PreferenceBatch:
        policy_chosen = self.policy.completion_logps(batch.media, batch.chosen)
        policy_rejected = self.policy.completion_logps(batch.media, batch.rejected)
        ref_chosen, ref_rejected = self._reference_pair_logps(batch)
        return PreferenceBatch(
            policy_chosen_logps=policy_chosen,
            policy_rejected_logps=policy_rejected,
            ref_chosen_logps=ref_chosen,
            ref_rejected_logps=ref_rejected,
            sample_weights=batch.sample_weights,
            metadata=dict(batch.metadata),
        )

    def step(self, batch: PreferencePairBatch, global_step: int) -> LossOutput:
        preference = self.preference_batch(batch)
        try:
            output = self.objective(preference, global_step, self.config.total_steps)
        except ObjectiveError as exc:
            raise TrainerError(f"objective rejected the batch at step {global_step}: {exc}") from exc
        if not bool(torch.isfinite(output.loss.detach())):
            raise TrainerError(f"non-finite loss at step {global_step}")
        accumulation = self.config.gradient_accumulation_steps
        (output.loss / accumulation).backward()  # type: ignore[no-untyped-call]
        parameters = [
            parameter for parameter in self.policy.trainable_parameters() if parameter.requires_grad
        ]
        boundary = (global_step + 1) % accumulation == 0 or global_step + 1 == self.config.total_steps
        if boundary:
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, self.config.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        else:
            # Report the running accumulated norm without clipping mid-accumulation.
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float("inf"))
        difficulties = batch.metadata.get("difficulty")
        agreements = batch.metadata.get("agreement")
        difficulty_mean = (
            sum(float(str(value)) for value in difficulties) / len(difficulties)
            if isinstance(difficulties, list) and difficulties
            else 0.0
        )
        agreement_mean = (
            sum(float(str(value)) for value in agreements) / len(agreements)
            if isinstance(agreements, list) and agreements
            else 1.0
        )
        step_record: dict[str, object] = dict(output.diagnostics)
        step_record.update(
            {
                "loss": float(output.loss.detach()),
                "grad_norm": float(grad_norm),
                "caption_length_chosen_mean": sum(word_count(text) for text in batch.chosen.texts)
                / len(batch.chosen.texts),
                "caption_length_rejected_mean": sum(word_count(text) for text in batch.rejected.texts)
                / len(batch.rejected.texts),
                "pair_difficulty_mean": difficulty_mean,
                "agreement_bucket": _agreement_bucket(agreement_mean),
                "track": batch.track,
                "global_step": global_step,
            }
        )
        self.log.record(step_record)
        return output

    def train(self, batches: Sequence[PreferencePairBatch]) -> dict[str, object]:
        if not batches:
            raise TrainerError("training requires at least one batch")
        global_step = 0
        while global_step < self.config.total_steps:
            batch = batches[global_step % len(batches)]
            self.step(batch, global_step)
            global_step += 1
        summary = self.log.summary()
        means = summary.get("means")
        if isinstance(means, dict) and not math.isfinite(
            float(str(means.get("loss", 0.0)))
        ):  # pragma: no cover
            raise TrainerError("training summary loss is non-finite")
        return summary
