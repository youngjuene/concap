"""Live Gemma C0 generation over real media (requires CUDA + cached model).

Run with: DPO_RUN_LIVE=1 uv run pytest -m live tests/candidates/test_live_generation.py
"""

from __future__ import annotations

import math
import os
import random
import struct
import wave
from pathlib import Path

import pytest
import torch

from dpo.candidates.generation import generate_c0_candidates_gemma
from tests.candidates.test_live_wiring import _gemma_contract

pytestmark = pytest.mark.live

_LIVE = os.environ.get("DPO_RUN_LIVE") == "1" and torch.cuda.is_available()


def _write_sine(path: Path, *, frequency: float, seconds: float = 2.0, rate: int = 16000) -> None:
    frames = b"".join(
        struct.pack("<h", int(0.4 * 32767 * math.sin(2 * math.pi * frequency * index / rate)))
        for index in range(int(seconds * rate))
    )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames)


def _write_texture(path: Path, kind: str, *, seconds: float = 2.0, rate: int = 16000) -> None:
    """One acoustically distinct clip per kind.

    Pure tones are captioned identically by the model, and a controlled error
    is by construction another clip's greedy caption — so a tone-only fixture
    makes every controlled error collide and generation correctly refuses.
    Real collections vary; these textures reproduce that.
    """
    rng = random.Random(hash(kind) & 0xFFFF)
    values = []
    for index in range(int(seconds * rate)):
        t = index / rate
        if kind == "noise":
            value = rng.uniform(-0.5, 0.5)
        elif kind == "chirp":
            value = 0.4 * math.sin(2 * math.pi * (200 + 600 * t) * t)
        elif kind == "clicks":
            value = 0.8 if (index % (rate // 5)) < 40 else 0.0
        else:  # amplitude-modulated tone
            value = 0.4 * math.sin(2 * math.pi * 440 * t) * (0.5 + 0.5 * math.sin(2 * math.pi * 3 * t))
        values.append(max(-1.0, min(1.0, value)))
    frames = b"".join(struct.pack("<h", int(value * 32767)) for value in values)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames)


@pytest.mark.skipif(not _LIVE, reason="set DPO_RUN_LIVE=1 on a CUDA machine with the cached model")
def test_gemma_generates_real_audio_captions(tmp_path: Path) -> None:
    clip_ids = ["live-clip-a", "live-clip-b", "live-clip-c", "live-clip-d"]
    for clip_id, kind in zip(clip_ids, ("noise", "chirp", "clicks", "am"), strict=True):
        _write_texture(tmp_path / f"{clip_id}.wav", kind)
    records = generate_c0_candidates_gemma(
        _gemma_contract(),
        track="audio",
        clip_ids=clip_ids,
        backend_config_path=Path("configs/gemma4/e4b.toml"),
        media_dir=tmp_path,
    )
    per_clip = int(str(_gemma_contract().candidates["per_clip"]))
    assert len(records) == per_clip * len(clip_ids)
    texts = {record.text for record in records}
    assert len(texts) > 1
    assert all(record.source_kind in {"greedy", "sample", "controlled_error"} for record in records)


@pytest.mark.skipif(not _LIVE, reason="set DPO_RUN_LIVE=1 on a CUDA machine with the cached model")
def test_gemma_completion_scoring_is_media_conditioned(tmp_path: Path) -> None:
    """The adapter must pass the full media encoding to the forward pass: the
    same completion over different audio must score differently."""
    from dpo.contracts.study_contract import load_contract
    from dpo.models.audio_media import build_audio_media
    from dpo.models.base import CompletionBatch
    from dpo.models.gemma4.adapter import GemmaCaptionAdapter
    from dpo.models.gemma4.backend_config import load_config

    clip_ids = ["cond-clip-a", "cond-clip-b"]
    for index, clip_id in enumerate(clip_ids):
        _write_sine(tmp_path / f"{clip_id}.wav", frequency=262.0 * (index + 1))
    files = {clip_id: tmp_path / f"{clip_id}.wav" for clip_id in clip_ids}
    adapter = GemmaCaptionAdapter(
        config=load_config("configs/gemma4/e4b.toml"),
        contract=load_contract("configs/study/canary.toml").tracks["audio"],
        media_resolver=lambda clip_id: files[clip_id],
    )
    completion = CompletionBatch(texts=("A steady tone plays throughout the recording.",))
    scores = [
        float(adapter.completion_logps(build_audio_media([clip_id], torch.zeros(1, 1600)), completion)[0])
        for clip_id in clip_ids
    ]
    assert abs(scores[0] - scores[1]) > 1e-4
