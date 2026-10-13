"""Pinned profiles, and verification that refuses any file that does not match."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from diff.__main__ import verification_record_path
from diff.model_profile import (
    PROFILES,
    QWEN_7B,
    TRIDA_7B,
    ModelProfile,
    ProfileMismatch,
    assert_weights_unchanged,
    file_sha256,
    verify_loaded_code,
    verify_snapshot,
)


def test_the_pins_and_hash_shapes() -> None:
    assert QWEN_7B.revision == "a09a35458c702b33eeacc393d103063234e8bc28"
    assert QWEN_7B.token_ids == {"endoftext": 151643, "im_end": 151645, "config_bos": 151643}
    assert all(
        len(value) == 64 for p in PROFILES.values() for value in [*p.files.values(), *p.weights.values()]
    )
    assert set(PROFILES) == {TRIDA_7B.profile_id, QWEN_7B.profile_id}  # Dream was removed on 2026-10-12


def test_the_trida_pin_and_token_ids() -> None:
    # The researcher's model since 2026-10-12; text hashes as in plan §2.3.
    assert TRIDA_7B.revision == "c37755e81222de0aff69ee481f5299a99928b105"
    assert set(TRIDA_7B.remote_code) == {"configuration_trida.py", "modeling_trida.py"}
    assert TRIDA_7B.files["modeling_trida.py"].startswith("100558ff8e0e")
    assert TRIDA_7B.token_ids == {
        "mask": 128012,
        "im_start": 128000,
        "im_end": 128001,
        "pad": 128004,
        "bos": 0,
    }
    assert TRIDA_7B.context_limit == 4096 and len(TRIDA_7B.weights) == 4


def synthetic_profile(directory: Path) -> ModelProfile:
    (directory / "config.json").write_text("{}", encoding="utf-8")
    (directory / "remote_thing.py").write_text("class Thing:\n    pass\n", encoding="utf-8")
    (directory / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    return ModelProfile(
        profile_id="synthetic@0",
        runner="trida",
        repo_id="synthetic/model",
        revision="0" * 40,
        files={name: file_sha256(directory / name) for name in ("config.json", "remote_thing.py")},
        weights={
            "model-00001-of-00001.safetensors": file_sha256(directory / "model-00001-of-00001.safetensors")
        },
        remote_code=("remote_thing.py",),
        tokenizer_files=(),
        transformers_version="4.46.2",
        token_ids={},
        context_limit=2048,
    )


def test_verify_snapshot_accepts_matching_files(tmp_path: Path) -> None:
    profile = synthetic_profile(tmp_path)
    record = verify_snapshot(tmp_path, profile)
    assert record["weights_hashed"] and set(record["files"]) == {"config.json", "remote_thing.py"}
    assert_weights_unchanged(tmp_path, record)


def test_a_tampered_remote_code_file_is_refused_before_loading(tmp_path: Path) -> None:
    profile = synthetic_profile(tmp_path)
    (tmp_path / "remote_thing.py").write_text("class Thing:\n    evil = True\n", encoding="utf-8")
    with pytest.raises(ProfileMismatch, match="remote_thing.py: sha256"):
        verify_snapshot(tmp_path, profile)


def test_missing_files_and_moved_weights_are_refused(tmp_path: Path) -> None:
    profile = synthetic_profile(tmp_path)
    record = verify_snapshot(tmp_path, profile)
    weight = tmp_path / "model-00001-of-00001.safetensors"
    os.utime(weight, ns=(1, 1))
    with pytest.raises(ProfileMismatch, match="changed after verification"):
        assert_weights_unchanged(tmp_path, record)
    (tmp_path / "config.json").unlink()
    with pytest.raises(ProfileMismatch, match="config.json: missing"):
        verify_snapshot(tmp_path, profile)
    with pytest.raises(ProfileMismatch, match="never hashed"):
        assert_weights_unchanged(tmp_path, verify_snapshot_without_weights(tmp_path, profile))


def verify_snapshot_without_weights(directory: Path, profile: ModelProfile) -> dict[str, object]:
    (directory / "config.json").write_text("{}", encoding="utf-8")
    return verify_snapshot(directory, profile, weights=False)


def test_loaded_code_must_be_the_verified_file(tmp_path: Path) -> None:
    profile = synthetic_profile(tmp_path)
    spec = importlib.util.spec_from_file_location("remote_thing", tmp_path / "remote_thing.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["remote_thing"] = module
    spec.loader.exec_module(module)
    try:
        assert verify_loaded_code([module.Thing()], profile) == {
            "remote_thing.py": profile.files["remote_thing.py"]
        }
        with pytest.raises(ProfileMismatch, match="no loaded class"):
            verify_loaded_code([object()], profile)
        (tmp_path / "remote_thing.py").write_text("class Thing:\n    changed = 1\n", encoding="utf-8")
        with pytest.raises(ProfileMismatch, match="imported copy"):
            verify_loaded_code([module.Thing()], profile)
    finally:
        del sys.modules["remote_thing"]


LOCAL_TRIDA = Path(__file__).resolve().parents[2] / "artifacts/diff/models/trida-7b-c37755e8"


@pytest.mark.skipif(
    not verification_record_path(LOCAL_TRIDA).is_file(), reason="no verified local Trida snapshot"
)
def test_the_local_trida_snapshot_matches_its_pins() -> None:
    record = verify_snapshot(LOCAL_TRIDA, TRIDA_7B, weights=False)
    assert set(record["files"]) == set(TRIDA_7B.files)


def test_the_record_path_survives_dots_in_the_name() -> None:
    assert verification_record_path(Path("models/qwen2.5-7b-instruct-a09a3545")).name == (
        "qwen2.5-7b-instruct-a09a3545.verification.json"
    )
