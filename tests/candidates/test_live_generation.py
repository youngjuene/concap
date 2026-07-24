"""Live Gemma C0 generation over real media (requires CUDA + cached model).

Run with: DPO_RUN_LIVE=1 uv run pytest -m live tests/candidates/test_live_generation.py
"""

from __future__ import annotations

import math
import os
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


@pytest.mark.skipif(not _LIVE, reason="set DPO_RUN_LIVE=1 on a CUDA machine with the cached model")
def test_gemma_generates_real_audio_captions(tmp_path: Path) -> None:
    clip_ids = ["live-clip-a", "live-clip-b"]
    for index, clip_id in enumerate(clip_ids):
        _write_sine(tmp_path / f"{clip_id}.wav", frequency=330.0 + 110.0 * index)
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
