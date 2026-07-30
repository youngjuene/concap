"""Offline tests for the live (Gemma) C0 generation wiring: gates before GPUs."""

from __future__ import annotations

import copy
import json
import tomllib
from pathlib import Path

import pytest

from dpo.candidates.candidate_records import CandidateError
from dpo.candidates.generation import (
    GEMMA_IMPLEMENTATION,
    generate_c0_candidates_gemma,
    resolve_media_files,
    verify_backend_pin,
)
from dpo.cli import main
from dpo.contracts.study_contract import StudyContract, validate_contract
from dpo.core.identity import sha256_file
from tests.conftest import CANARY_CONTRACT

E4B_REVISION = "a4c2d58be94dda072b918d9db64ee85c8ed34e3f"


def _gemma_contract_document() -> dict[str, object]:
    with CANARY_CONTRACT.open("rb") as handle:
        document = tomllib.load(handle)
    mutated = copy.deepcopy(document)
    models = mutated["models"]
    assert isinstance(models, dict)
    models["seed"] = {
        "model_id": "google/gemma-4-E4B-it",
        "revision": E4B_REVISION,
        "implementation": GEMMA_IMPLEMENTATION,
        "lock_hash": "sha256:" + "2c" * 32,
        "init_seed": 0,
    }
    return mutated


def _gemma_contract(**extra: object) -> StudyContract:
    document = _gemma_contract_document()
    document.update(extra)
    return validate_contract(document)


def test_media_resolution_finds_by_suffix_and_fails_on_gaps(tmp_path: Path) -> None:
    (tmp_path / "clip-a.wav").write_bytes(b"RIFF")
    (tmp_path / "clip-b.mp3").write_bytes(b"ID3")
    resolved = resolve_media_files(tmp_path, ["clip-a", "clip-b"], track="audio")
    assert resolved["clip-a"].suffix == ".wav"
    assert resolved["clip-b"].suffix == ".mp3"
    with pytest.raises(CandidateError, match="clip-c"):
        resolve_media_files(tmp_path, ["clip-a", "clip-c"], track="audio")
    # A video file does not satisfy the audio track.
    (tmp_path / "clip-d.mp4").write_bytes(b"\x00")
    with pytest.raises(CandidateError, match="clip-d"):
        resolve_media_files(tmp_path, ["clip-d"], track="audio")


def test_backend_pin_enforced_when_present(tmp_path: Path) -> None:
    backend = tmp_path / "e4b.toml"
    backend.write_text("[model]\n", encoding="utf-8")
    pinned = _gemma_contract(backends={"audio": {"config_hash": sha256_file(backend)}})
    verify_backend_pin(pinned, track="audio", backend_config_path=backend)
    backend.write_text("[model]\n# drifted\n", encoding="utf-8")
    with pytest.raises(CandidateError, match="pins backends.audio"):
        verify_backend_pin(pinned, track="audio", backend_config_path=backend)
    # No pin registered: any file passes.
    verify_backend_pin(_gemma_contract(), track="audio", backend_config_path=backend)


def test_gemma_generation_rejects_wrong_implementation_and_track(tmp_path: Path) -> None:
    with CANARY_CONTRACT.open("rb") as handle:
        tiny_contract = validate_contract(tomllib.load(handle))
    with pytest.raises(CandidateError, match="Gemma seed-model implementation"):
        generate_c0_candidates_gemma(
            tiny_contract,
            track="audio",
            clip_ids=["clip-a"],
            backend_config_path=tmp_path / "missing.toml",
            media_dir=tmp_path,
        )
    # Track/backend mismatch fails before any media or model work.
    contract = _gemma_contract()
    backend = Path("configs/gemma4/e4b-audio.toml")
    with pytest.raises(CandidateError, match="media_inputs"):
        generate_c0_candidates_gemma(
            contract,
            track="visual",
            clip_ids=["clip-a"],
            backend_config_path=backend,
            media_dir=tmp_path,
        )


def test_cli_requires_live_arguments_and_gates_on_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    contract_path = tmp_path / "gemma.toml"
    # Writing TOML back out from a mutated document is brittle; patch the
    # canary contract text, replacing only the models.seed table.
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
    contract_path.write_text(source[:start] + seed_block + "\n" + source[end:], encoding="utf-8")

    workspace = tmp_path / "workspace"
    base = [
        "candidates",
        "generate",
        "--workspace",
        str(workspace),
        "--contract",
        str(contract_path),
        "--artifact-id",
        "sha256:" + "ab" * 32,
        "--track",
        "audio",
        "--split",
        "train",
        "--dataset-version",
        "live/v1",
    ]
    # Missing live arguments is a domain error, before any store side effects.
    with pytest.raises(SystemExit) as excinfo:
        main(base)
    assert excinfo.value.code == 2
    assert not workspace.exists()

    # With the arguments present but no CUDA, the gate refuses with exit 3.
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    code = main(base + ["--backend-config", "configs/gemma4/e4b-audio.toml", "--media-dir", str(tmp_path)])
    assert code == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked_pending_external_operation"
    assert payload["side_effects"] is False
    assert not workspace.exists()
