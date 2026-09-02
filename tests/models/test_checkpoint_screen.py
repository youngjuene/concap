"""The load seam must refuse a hostile checkpoint directory before any GPU work."""

from __future__ import annotations

from pathlib import Path

import pytest

from dpo.contracts.study_contract import load_contract
from dpo.core.safety import CheckpointSafetyError, screen_checkpoint_directory
from dpo.models.gemma4.adapter import GemmaCaptionAdapter
from dpo.models.gemma4.backend_config import load_config
from dpo.models.gemma4.training_backend import GemmaBackend, _Attachment


def _adapter_dir(tmp_path: Path) -> Path:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"safetensors-fixture")
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    return adapter


def test_screen_accepts_a_clean_adapter(tmp_path: Path) -> None:
    assert screen_checkpoint_directory(_adapter_dir(tmp_path)) == tmp_path / "adapter"


@pytest.mark.parametrize(
    ("plant", "match"),
    [
        (lambda root: (root / "training_args.bin").write_bytes(b"pickle"), "legacy or unsafe"),
        (lambda root: (root / "extra.pkl").write_bytes(b"pickle"), "legacy or unsafe"),
        (lambda root: (root / "adapter_model.safetensors").unlink(), "safetensors is required"),
    ],
)
def test_screen_rejects_hostile_contents(tmp_path: Path, plant, match: str) -> None:
    adapter = _adapter_dir(tmp_path)
    plant(adapter)
    with pytest.raises(CheckpointSafetyError, match=match):
        screen_checkpoint_directory(adapter)


def test_screen_rejects_symlinked_weights(tmp_path: Path) -> None:
    adapter = _adapter_dir(tmp_path)
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"elsewhere")
    (adapter / "adapter_model.safetensors").unlink()
    (adapter / "adapter_model.safetensors").symlink_to(outside)
    with pytest.raises(CheckpointSafetyError, match="symlink"):
        screen_checkpoint_directory(adapter)


def test_the_caption_adapter_screens_its_checkpoint_at_construction(tmp_path: Path) -> None:
    """The seam both instruments load through: screened before the CUDA gate, not after."""
    contract = load_contract("configs/study/street-audio.toml")
    config = load_config("configs/gemma4/e4b-audio.toml")
    adapter = _adapter_dir(tmp_path)
    (adapter / "extra.pkl").write_bytes(b"pickle")
    with pytest.raises(CheckpointSafetyError, match="legacy or unsafe"):
        GemmaCaptionAdapter(
            config=config,
            contract=contract.tracks["audio"],
            media_resolver=lambda reference: reference,
            adapter_dir=str(adapter),
        )


def test_the_caption_adapter_accepts_a_clean_checkpoint(tmp_path: Path) -> None:
    """The screen must not stand in the way of the directory the training backend writes."""
    contract = load_contract("configs/study/street-audio.toml")
    config = load_config("configs/gemma4/e4b-audio.toml")
    built = GemmaCaptionAdapter(
        config=config,
        contract=contract.tracks["audio"],
        media_resolver=lambda reference: reference,
        adapter_dir=str(_adapter_dir(tmp_path)),
    )
    assert built.adapter_dir == str(tmp_path / "adapter")


def test_backend_attach_screens_before_any_cuda_work(tmp_path: Path) -> None:
    """The screen must beat the CUDA gate: a CPU box still refuses a hostile dir."""
    contract = load_contract("configs/study/street-audio.toml")
    config = load_config("configs/gemma4/e4b-audio.toml")
    backend = GemmaBackend(contract=contract, configs={"audio": config}, media_dir=tmp_path)
    adapter = _adapter_dir(tmp_path)
    (adapter / "training_args.bin").write_bytes(b"pickle")
    with pytest.raises(CheckpointSafetyError, match="legacy or unsafe"):
        backend._attach("audio", _Attachment(name="probe", track="audio", directory=adapter))
