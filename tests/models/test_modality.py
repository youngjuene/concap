"""Modality-isolation tests (PRD section 32.3)."""

from __future__ import annotations

import pytest
import torch

from dpo.models.audio_media import build_audio_media
from dpo.models.base import (
    CompletionBatch,
    MediaBatch,
    ModalityIsolationError,
    PreferencePairBatch,
    ensure_single_track,
)
from dpo.models.tiny import TinyAdapter, synthetic_media
from dpo.models.visual_media import build_visual_media


def test_visual_batches_contain_no_audio_tensor() -> None:
    with pytest.raises(ModalityIsolationError, match="cross-modal"):
        MediaBatch(
            track="visual",
            clip_ids=("clip-a",),
            features={"frames": torch.zeros(1, 4, 8), "waveform": torch.zeros(1, 16)},
        )


def test_audio_batches_contain_no_video_tensor() -> None:
    with pytest.raises(ModalityIsolationError, match="cross-modal"):
        MediaBatch(
            track="audio",
            clip_ids=("clip-a",),
            features={"waveform": torch.zeros(1, 16), "frames": torch.zeros(1, 4, 8)},
        )


def test_mixed_track_batches_fail() -> None:
    visual = build_visual_media(["clip-a"], torch.zeros(1, 4, 8))
    audio = build_audio_media(["clip-a"], torch.zeros(1, 16))
    with pytest.raises(ModalityIsolationError, match="mixed-track"):
        ensure_single_track([visual, audio])


def test_adapters_reject_wrong_track_media() -> None:
    adapter = TinyAdapter(track="visual", prompt="Describe.", media_dim=8)
    audio = build_audio_media(["clip-a"], torch.zeros(1, 6, 8))
    with pytest.raises(ModalityIsolationError):
        adapter.completion_logps(audio, CompletionBatch(texts=("a caption",)))


def test_chosen_and_rejected_share_one_media_input() -> None:
    media = synthetic_media("visual", ["clip-a", "clip-b"], media_dim=8)
    batch = PreferencePairBatch(
        track="visual",
        pair_ids=("pair-1", "pair-2"),
        media=media,
        chosen=CompletionBatch(texts=("chosen one", "chosen two")),
        rejected=CompletionBatch(texts=("rejected one", "rejected two")),
    )
    # Structurally a single media object: the batch has one media field and it
    # is the same object for both scoring passes.
    assert batch.media is media


def test_processor_output_shapes_are_validated() -> None:
    with pytest.raises(ModalityIsolationError):
        build_visual_media(["clip-a"], torch.zeros(1, 16))  # 2-D is not a frame tensor
    with pytest.raises(ModalityIsolationError):
        build_audio_media(["clip-a", "clip-b"], torch.zeros(1, 16))  # batch mismatch
    with pytest.raises(ModalityIsolationError):
        build_visual_media(["clip-a"], torch.zeros(1, 4, 1))  # degenerate feature dim


def test_media_mask_and_track_keys_are_closed_vocabulary() -> None:
    with pytest.raises(ModalityIsolationError, match="unknown tensor"):
        MediaBatch(
            track="visual",
            clip_ids=("clip-a",),
            features={"frames": torch.zeros(1, 4, 8), "extra": torch.zeros(1, 2)},
        )
