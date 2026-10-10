"""Pinned profiles, and verification that refuses any file that does not match."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from diff.__main__ import verification_record_path
from diff.model_profile import (
    DREAM_7B,
    PROFILES,
    QWEN_7B,
    ModelProfile,
    ProfileMismatch,
    assert_weights_unchanged,
    file_sha256,
    verify_loaded_code,
    verify_snapshot,
)

DREAM_FILES = {
    "configuration_dream.py",
    "modeling_dream.py",
    "generation_utils.py",
    "tokenization_dream.py",
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
}


def test_the_pins_and_token_ids() -> None:
    assert DREAM_7B.revision == "05334cb9faaf763692dcf9d8737c642be2b2a6ae"
    assert QWEN_7B.revision == "a09a35458c702b33eeacc393d103063234e8bc28"
    assert set(DREAM_7B.files) >= DREAM_FILES  # every file the follow-up names is pinned
    assert set(DREAM_7B.remote_code) == {f for f in DREAM_FILES if f.endswith(".py")}
    assert DREAM_7B.token_ids == {
        "mask": 151666,
        "endoftext": 151643,
        "im_end": 151645,
        "config_bos": 151643,
        "tokenizer_bos": 151665,
    }
    assert DREAM_7B.context_limit == 2048 and len(DREAM_7B.weights) == 4
    assert all(
        len(value) == 64 for p in PROFILES.values() for value in [*p.files.values(), *p.weights.values()]
    )
    assert set(PROFILES) == {DREAM_7B.profile_id, QWEN_7B.profile_id}


def synthetic_profile(directory: Path) -> ModelProfile:
    (directory / "config.json").write_text("{}", encoding="utf-8")
    (directory / "remote_thing.py").write_text("class Thing:\n    pass\n", encoding="utf-8")
    (directory / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    return ModelProfile(
        profile_id="synthetic@0",
        runner="dream",
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


LOCAL_DREAM = Path(__file__).resolve().parents[2] / "artifacts/diff/models/dream-v0-instruct-7b-05334cb9"


# `model fetch` writes the record only after a complete, verified download.
@pytest.mark.skipif(
    not verification_record_path(LOCAL_DREAM).is_file(), reason="no verified local Dream snapshot"
)
def test_the_local_dream_snapshot_matches_its_pins() -> None:
    record = verify_snapshot(LOCAL_DREAM, DREAM_7B, weights=False)
    assert set(record["files"]) == set(DREAM_7B.files)


def test_the_record_path_survives_dots_in_the_name() -> None:
    assert verification_record_path(Path("models/qwen2.5-7b-instruct-a09a3545")).name == (
        "qwen2.5-7b-instruct-a09a3545.verification.json"
    )
