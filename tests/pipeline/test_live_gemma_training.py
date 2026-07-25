"""A real Gemma QLoRA training cell (requires CUDA + the cached checkpoint).

Run with: DPO_RUN_LIVE=1 uv run pytest -m live tests/pipeline/test_live_gemma_training.py
"""

from __future__ import annotations

import copy
import math
import os
import struct
import tomllib
import wave
from pathlib import Path
from typing import Any

import pytest
import torch

from dpo.contracts.study_contract import validate_contract
from dpo.data.derive_sft import SftExample
from dpo.models.gemma4.backend_config import load_config
from dpo.models.gemma4.training_backend import (
    LANGUAGE_MODEL_TARGETS,
    GemmaBackend,
    resolve_lora_targets,
)
from dpo.pipeline.live_runner import LiveMatrixRunner
from tests.candidates.test_live_wiring import _gemma_contract_document

pytestmark = pytest.mark.live

_LIVE = os.environ.get("DPO_RUN_LIVE") == "1" and torch.cuda.is_available()
_CLIPS = ("train-clip-a", "train-clip-b")


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


def _live_contract() -> Any:
    """A Gemma contract that trains exactly one step with a language-model LoRA."""
    document = copy.deepcopy(_gemma_contract_document())
    training = document["training"]
    assert isinstance(training, dict)
    training["precision"] = "bf16"
    training["epochs"] = 0.5  # one batch of two rows -> ceil(0.5 * 1) == 1 step
    training["batch_size"] = 4
    training["lora"] = dict(training["lora"], targets=[LANGUAGE_MODEL_TARGETS])
    return validate_contract(document)


def test_the_live_contract_resolves_a_language_model_scope() -> None:
    # Cheap enough to run without a GPU: proves the fixture contract is the one
    # the guard would accept, so a skipped live test still pins its shape.
    assert resolve_lora_targets(_live_contract()) == LANGUAGE_MODEL_TARGETS
    with Path("configs/gemma4/e4b.toml").open("rb") as handle:
        assert tomllib.load(handle)["model"]["media_inputs"] == "audio"


@pytest.mark.skipif(not _LIVE, reason="set DPO_RUN_LIVE=1 on a CUDA machine with the cached model")
def test_gemma_sft_cell_trains_one_step_and_saves_a_distinct_checkpoint(tmp_path: Path) -> None:
    for index, clip_id in enumerate(_CLIPS):
        _write_sine(tmp_path / f"{clip_id}.wav", frequency=330.0 + 110.0 * index)
    contract = _live_contract()
    backend = GemmaBackend(
        contract=contract,
        configs={"audio": load_config("configs/gemma4/e4b.toml")},
        media_dir=tmp_path,
    )
    rows = tuple(
        SftExample(
            example_id=f"sft-{index}",
            clip_id=clip_id,
            track="audio",
            candidate_id=f"cand-{index}",
            completion="A steady tone plays throughout the short recording.",
            consensus_weight=1.0,
        )
        for index, clip_id in enumerate(_CLIPS)
    )
    runner = LiveMatrixRunner(
        contract=contract,
        backend=backend,
        checkpoint_dir=tmp_path / "checkpoints",
        strict_pairs={"audio": ()},
        metadata_pairs={"audio": ()},
        sft_rows={"audio": rows},
    )
    seed_signature = backend.seed_adapter("audio").state_signature()
    cell = runner.run_cell("SFT", track="audio", seed=1)
    assert cell.trained
    assert cell.steps == 1
    assert cell.checkpoint_signature != seed_signature
    directory = runner.cell_directory("SFT", cell.variant_id, "audio", 1)
    assert (directory / "adapter_model.safetensors").is_file()
    assert (directory / "adapter_config.json").is_file()
    assert (directory / "cell.json").is_file()
    # The saved checkpoint reloads to the same weights.
    reloaded = backend.load("audio", directory)
    assert reloaded.state_signature() == cell.checkpoint_signature
    backend.release(reloaded)
    # A later process resumes from disk instead of retraining. A fresh runner
    # over the same checkpoint directory is exactly that situation; reusing the
    # first runner would report "trained", which is the truthful answer to
    # "did THIS run have to train the cell?".
    resumed_runner = LiveMatrixRunner(
        contract=contract,
        backend=backend,
        checkpoint_dir=tmp_path / "checkpoints",
        strict_pairs={"audio": ()},
        metadata_pairs={"audio": ()},
        sft_rows={"audio": rows},
    )
    again = resumed_runner.run_cell("SFT", track="audio", seed=1)
    assert again.document() == cell.document()
    assert resumed_runner.resumed_count == 1
    assert resumed_runner.trained_count == 0
    backend.close()
    peak = cell.diagnostics_summary.get("peak_vram_bytes")
    assert isinstance(peak, int) and peak > 0
