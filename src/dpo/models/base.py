"""Modality-isolated media/batch contracts and the model adapter protocol.

A visual media batch may only ever carry visual feature keys and an audio
media batch only audio keys; mixed-track batches are rejected at construction.
A preference pair batch holds exactly one media batch, so chosen and rejected
completions physically cannot receive different media inputs.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch import Tensor

from dpo.contracts.study_contract import TRACKS

VISUAL_FEATURE_KEYS = frozenset({"frames", "frame_mask"})
AUDIO_FEATURE_KEYS = frozenset({"waveform", "waveform_mask"})
TRACK_FEATURE_KEYS = {"visual": VISUAL_FEATURE_KEYS, "audio": AUDIO_FEATURE_KEYS}


class ModalityIsolationError(ValueError):
    """Raised when a batch would mix tracks or carry cross-modal tensors."""


@dataclass(frozen=True)
class MediaBatch:
    """Batched media features of exactly one track."""

    track: str
    clip_ids: tuple[str, ...]
    features: dict[str, Tensor]

    def __post_init__(self) -> None:
        if self.track not in TRACKS:
            raise ModalityIsolationError(f"media track must be one of {sorted(TRACKS)}")
        if not self.clip_ids:
            raise ModalityIsolationError("media batch must reference at least one clip")
        if not self.features:
            raise ModalityIsolationError("media batch must carry at least one feature tensor")
        allowed = TRACK_FEATURE_KEYS[self.track]
        forbidden = TRACK_FEATURE_KEYS["audio" if self.track == "visual" else "visual"]
        for key, tensor in self.features.items():
            if key in forbidden:
                raise ModalityIsolationError(f"{self.track} media batch carries cross-modal tensor {key!r}")
            if key not in allowed:
                raise ModalityIsolationError(
                    f"{self.track} media batch carries unknown tensor {key!r};"
                    f" allowed keys are {sorted(allowed)}"
                )
            if tensor.shape[0] != len(self.clip_ids):
                raise ModalityIsolationError(
                    f"media tensor {key!r} batch dimension does not match clip count"
                )
            if not bool(torch.isfinite(tensor.detach()).all()):
                raise ModalityIsolationError(f"media tensor {key!r} contains non-finite values")

    @property
    def batch_size(self) -> int:
        return len(self.clip_ids)


@dataclass(frozen=True)
class CompletionBatch:
    """Completion texts to score against one media batch."""

    texts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.texts or any(not text.strip() for text in self.texts):
            raise ModalityIsolationError("completion batch texts must be non-empty")


@dataclass
class PreferencePairBatch:
    """One media batch, two completion batches. Media is structurally shared."""

    track: str
    pair_ids: tuple[str, ...]
    media: MediaBatch
    chosen: CompletionBatch
    rejected: CompletionBatch
    sample_weights: Tensor | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.track != self.media.track:
            raise ModalityIsolationError("pair batch track and media track disagree")
        count = len(self.pair_ids)
        if not (
            count
            and count == self.media.batch_size
            and count == len(self.chosen.texts)
            and count == len(self.rejected.texts)
        ):
            raise ModalityIsolationError("pair batch components disagree on batch size")
        if self.sample_weights is not None and self.sample_weights.shape != (count,):
            raise ModalityIsolationError("pair batch sample_weights shape is invalid")


class ModelAdapter(Protocol):
    """One track-bound policy: scoring, generation, and persistence.

    ``completion_logps`` must be differentiable and must route through
    ``dpo.models.logprob.completion_logprobs`` so that every experiment
    shares one log-probability implementation.
    """

    @property
    def track(self) -> str: ...

    def completion_logps(self, media: MediaBatch, completions: CompletionBatch) -> Tensor: ...

    def generate(
        self, media: MediaBatch, *, temperature: float, top_p: float, max_new_tokens: int, seed: int
    ) -> list[str]: ...

    def trainable_parameters(self) -> Iterator[torch.nn.Parameter]: ...

    def state_signature(self) -> str: ...


def ensure_single_track(batches: Sequence[MediaBatch]) -> str:
    tracks = {batch.track for batch in batches}
    if len(tracks) != 1:
        raise ModalityIsolationError(f"mixed-track batches are rejected: {sorted(tracks)}")
    return next(iter(tracks))
