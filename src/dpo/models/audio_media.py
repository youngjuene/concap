"""Audio-track processor boundary: waveforms in, never frames."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from dpo.models.base import MediaBatch, ModalityIsolationError


def build_audio_media(clip_ids: Sequence[str], waveform: Tensor) -> MediaBatch:
    """Waveforms are [B, S] mono samples or [B, T, D] spectrogram features."""
    if waveform.dim() not in (2, 3):
        raise ModalityIsolationError(
            "audio input must be [B, S] mono samples or [B, T, D] spectrogram features"
        )
    if waveform.shape[0] != len(clip_ids):
        raise ModalityIsolationError("audio batch dimension does not match clip count")
    return MediaBatch(
        track="audio",
        clip_ids=tuple(clip_ids),
        features={"waveform": waveform.to(torch.float32)},
    )
