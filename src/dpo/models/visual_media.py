"""Visual-track processor boundary: frames in, never audio.

The processor accepts only the muted-video derivative of a clip. Any attempt
to attach an audio tensor fails at MediaBatch construction; this module adds
the input-side checks (shape discipline and per-clip provenance).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from dpo.models.base import MediaBatch, ModalityIsolationError


def build_visual_media(clip_ids: Sequence[str], frames: Tensor) -> MediaBatch:
    """Frames are [B, T, D] pooled features or [B, T, C, H, W] raw frames."""
    if frames.dim() not in (3, 5):
        raise ModalityIsolationError("visual frames must be [B, T, D] features or [B, T, C, H, W] raw frames")
    if frames.shape[0] != len(clip_ids):
        raise ModalityIsolationError("visual frames batch dimension does not match clip count")
    if frames.dim() == 5 and frames.shape[2] not in (1, 3):
        raise ModalityIsolationError("raw visual frames must have 1 or 3 channels")
    if frames.dim() == 3 and 1 in frames.shape[1:]:
        # A [B, T, 1] tensor is indistinguishable from a mono waveform that was
        # reshaped to sneak past the track check; refuse the ambiguity.
        raise ModalityIsolationError("degenerate visual feature shape; provide real frame features")
    return MediaBatch(
        track="visual",
        clip_ids=tuple(clip_ids),
        features={"frames": frames.to(torch.float32)},
    )
