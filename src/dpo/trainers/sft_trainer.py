"""Chosen-only SFT trainer for SFT (and the SFT_DPO warm start's first stage)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from dpo.data.derive_sft import SftExample
from dpo.models.base import CompletionBatch, MediaBatch, ModelAdapter
from dpo.objectives.sft import SFTObjective
from dpo.trainers.callbacks import DiagnosticsLog
from dpo.trainers.preference_trainer import MediaProvider, TrainerConfig, TrainerError


@dataclass(frozen=True)
class SftBatch:
    track: str
    example_ids: tuple[str, ...]
    media: MediaBatch
    completions: CompletionBatch
    weights: torch.Tensor


def build_sft_batches(
    rows: Sequence[SftExample],
    media_provider: MediaProvider,
    *,
    batch_size: int,
) -> list[SftBatch]:
    if batch_size < 1:
        raise TrainerError("batch_size must be positive")
    seen: set[str] = set()
    for row in rows:
        if row.candidate_id in seen:
            raise TrainerError(
                f"candidate {row.candidate_id!r} appears twice; SFT targets must be deduplicated"
            )
        seen.add(row.candidate_id)
    batches = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        tracks = {row.track for row in chunk}
        if len(tracks) != 1:
            raise TrainerError("mixed-track batches are rejected")
        track = next(iter(tracks))
        batches.append(
            SftBatch(
                track=track,
                example_ids=tuple(row.example_id for row in chunk),
                media=media_provider(track, [row.clip_id for row in chunk]),
                completions=CompletionBatch(texts=tuple(row.completion for row in chunk)),
                weights=torch.tensor([row.consensus_weight for row in chunk], dtype=torch.float32),
            )
        )
    return batches


class SftTrainer:
    def __init__(
        self,
        policy: ModelAdapter,
        config: TrainerConfig,
        *,
        log: DiagnosticsLog | None = None,
    ) -> None:
        self.policy = policy
        self.config = config
        self.objective = SFTObjective()
        self.log = log if log is not None else DiagnosticsLog()
        torch.manual_seed(config.seed)
        parameters = [parameter for parameter in policy.trainable_parameters() if parameter.requires_grad]
        if not parameters:
            raise TrainerError("the policy has no trainable parameters")
        self.optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)

    def step(self, batch: SftBatch, global_step: int) -> float:
        logps = self.policy.completion_logps(batch.media, batch.completions)
        output = self.objective.loss(logps, batch.weights)
        if not bool(torch.isfinite(output.loss.detach())):
            raise TrainerError(f"non-finite SFT loss at step {global_step}")
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
        record: dict[str, object] = dict(output.diagnostics)
        record.update(
            {
                "loss": float(output.loss.detach()),
                "grad_norm": float(grad_norm),
                "track": batch.track,
                "global_step": global_step,
            }
        )
        self.log.record(record)
        return float(output.loss.detach())

    def train(self, batches: Sequence[SftBatch]) -> dict[str, object]:
        if not batches:
            raise TrainerError("training requires at least one batch")
        global_step = 0
        while global_step < self.config.total_steps:
            self.step(batches[global_step % len(batches)], global_step)
            global_step += 1
        return self.log.summary()
