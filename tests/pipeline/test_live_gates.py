"""Gates before GPUs for the training commands, and the frozen-towers guard.

Everything a live run can get wrong about its contract or its hardware must be
refused before the workspace exists, and every LoRA attach must be provably
inside the text language model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dpo.candidates.generation import GEMMA_IMPLEMENTATION
from dpo.cli import main
from dpo.contracts.study_contract import ContractError, load_contract
from dpo.models.gemma4.modeling import (
    ModelSetupError,
    assert_text_only_lora_scope,
    trainable_parameter_names,
)
from dpo.models.gemma4.training_backend import LANGUAGE_MODEL_TARGETS, resolve_lora_targets
from tests.candidates.test_live_wiring import E4B_REVISION
from tests.conftest import CANARY_CONTRACT

_VALID_NAMES = [
    "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight",
    "base_model.model.model.language_model.layers.0.mlp.gate_proj.lora_B.default.weight",
]


def _gemma_contract_text(tmp_path: Path, *, language_model_targets: bool) -> str:
    """The canary contract with a Gemma seed model, written back out as TOML."""
    source = CANARY_CONTRACT.read_text(encoding="utf-8")
    seed_block = (
        "[models.seed]\n"
        'model_id = "google/gemma-4-E4B-it"\n'
        f'revision = "{E4B_REVISION}"\n'
        f'implementation = "{GEMMA_IMPLEMENTATION}"\n'
        'lock_hash = "sha256:' + "2c" * 32 + '"\n'
        "init_seed = 0\n"
    )
    start = source.index("[models.seed]")
    end = source.index("[candidates]")
    text = source[:start] + seed_block + "\n" + source[end:]
    if language_model_targets:
        text = text.replace('targets = ["mixer", "head"]', f"targets = ['{LANGUAGE_MODEL_TARGETS}']")
    path = tmp_path / f"gemma-{'lm' if language_model_targets else 'tiny'}.toml"
    path.write_text(text, encoding="utf-8")
    return str(path)


class _Parameter:
    def __init__(self, requires_grad: bool) -> None:
        self.requires_grad = requires_grad


class _Model:
    def __init__(self, names: list[str], frozen: list[str] | None = None) -> None:
        self._named = [(name, _Parameter(True)) for name in names]
        self._named.extend((name, _Parameter(False)) for name in frozen or [])

    def named_parameters(self) -> list[tuple[str, _Parameter]]:
        return list(self._named)


def test_trainable_parameter_names_reports_only_trainable_parameters() -> None:
    model = _Model(_VALID_NAMES, frozen=["base_model.model.vision_tower.blocks.0.weight"])
    assert trainable_parameter_names(model) == _VALID_NAMES


def test_text_only_lora_scope_accepts_language_model_parameters() -> None:
    assert assert_text_only_lora_scope(_Model(_VALID_NAMES)) == _VALID_NAMES


def test_text_only_lora_scope_rejects_an_empty_trainable_set() -> None:
    with pytest.raises(ModelSetupError, match="no trainable LoRA parameters"):
        assert_text_only_lora_scope(_Model([]))
    with pytest.raises(ModelSetupError, match="no trainable LoRA parameters"):
        assert_text_only_lora_scope(_Model(_VALID_NAMES), names=[])


def test_text_only_lora_scope_rejects_parameters_outside_the_language_model() -> None:
    tower = "base_model.model.model.vision_tower.encoder.layers.0.self_attn.q_proj.lora_A.weight"
    with pytest.raises(ModelSetupError, match="escaped the text language model"):
        assert_text_only_lora_scope(_Model([*_VALID_NAMES, tower]))


def test_text_only_lora_scope_rejects_vision_and_audio_parameters() -> None:
    for modality in ("vision", "audio"):
        name = f"base_model.model.model.language_model.{modality}_projector.lora_A.default.weight"
        with pytest.raises(ModelSetupError, match="vision/audio parameters became trainable"):
            assert_text_only_lora_scope(_Model([name]))


def test_lora_targets_must_name_language_model_modules(tmp_path: Path) -> None:
    tiny = load_contract(_gemma_contract_text(tmp_path, language_model_targets=False))
    with pytest.raises(ContractError) as excinfo:
        resolve_lora_targets(tiny)
    message = str(excinfo.value)
    assert "tiny model's modules" in message
    assert LANGUAGE_MODEL_TARGETS in message  # the known-good scope is in the refusal
    live = load_contract(_gemma_contract_text(tmp_path, language_model_targets=True))
    assert resolve_lora_targets(live) == LANGUAGE_MODEL_TARGETS


def test_train_run_refuses_a_tiny_lora_scope_before_any_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "store"
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                *("train", "run", "--workspace", str(workspace)),
                *("--contract", _gemma_contract_text(tmp_path, language_model_targets=False)),
                *("--artifact-id", "sha256:" + "ab" * 32),
                *("--checkpoint-dir", str(tmp_path / "checkpoints")),
                *("--backend-config", "configs/gemma4/e4b.toml"),
                *("--media-dir", str(tmp_path)),
            ]
        )
    assert excinfo.value.code == 2
    assert not workspace.exists()


@pytest.mark.parametrize("command", ["train", "select"])
def test_live_training_gates_on_cuda_without_side_effects(
    command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    workspace = tmp_path / f"store-{command}"
    code = main(
        [
            *(command, "run", "--workspace", str(workspace)),
            *("--contract", _gemma_contract_text(tmp_path, language_model_targets=True)),
            *("--artifact-id", "sha256:" + "ab" * 32),
            *("--checkpoint-dir", str(tmp_path / "checkpoints")),
            *("--backend-config", "configs/gemma4/e4b.toml"),
            *("--media-dir", str(tmp_path)),
        ]
    )
    assert code == 3
    document = json.loads(capsys.readouterr().out)
    assert document["status"] == "blocked_pending_external_operation"
    assert document["command"] == f"{command} run"
    assert document["side_effects"] is False
    assert not workspace.exists()


@pytest.mark.parametrize("command", ["train", "select"])
def test_live_training_requires_both_live_arguments(command: str, tmp_path: Path) -> None:
    workspace = tmp_path / f"store-{command}"
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                *(command, "run", "--workspace", str(workspace)),
                *("--contract", _gemma_contract_text(tmp_path, language_model_targets=True)),
                *("--artifact-id", "sha256:" + "ab" * 32),
                *("--checkpoint-dir", str(tmp_path / "checkpoints")),
            ]
        )
    assert excinfo.value.code == 2
    assert not workspace.exists()


@pytest.mark.parametrize("command", ["train", "select"])
def test_the_tiny_backend_refuses_live_arguments(command: str, tmp_path: Path) -> None:
    workspace = tmp_path / f"store-{command}"
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                *(command, "run", "--workspace", str(workspace)),
                *("--contract", str(CANARY_CONTRACT)),
                *("--artifact-id", "sha256:" + "ab" * 32),
                *("--checkpoint-dir", str(tmp_path / "checkpoints")),
                *("--media-dir", str(tmp_path)),
            ]
        )
    assert excinfo.value.code == 2
    assert not workspace.exists()
