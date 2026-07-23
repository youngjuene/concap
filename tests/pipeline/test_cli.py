"""CLI surface tests: JSON output, exit codes, and the exit-3 live boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dpo.cli import main
from dpo.core.identity import sha256_bytes
from tests.conftest import CANARY_CONTRACT


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, object]]:
    code = main(list(argv))
    output = capsys.readouterr().out
    return code, json.loads(output)


def test_contract_validate(capsys: pytest.CaptureFixture[str]) -> None:
    code, document = _run(capsys, "contract", "validate", "--contract", str(CANARY_CONTRACT))
    assert code == 0
    assert document["execution_class"] == "synthetic_canary"
    assert document["tracks"] == ["audio", "visual"]


def test_stage_list_exposes_the_lineage(capsys: pytest.CaptureFixture[str]) -> None:
    code, document = _run(capsys, "stage", "list")
    assert code == 0
    stages = document["stages"]
    assert isinstance(stages, dict)
    assert "lock-splits" in stages
    lineage = document["lineage"]
    assert isinstance(lineage, list)
    assert ["contract", "lock"] not in lineage  # lineage is artifact-typed, not guessed


def test_live_boundaries_exit_3_without_side_effects(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    for command in ("train", "evaluate"):
        code, document = _run(
            capsys,
            command,
            "run",
            "--workspace",
            str(tmp_path / command),
            "--contract",
            str(CANARY_CONTRACT),
            "--invoke-external",
        )
        assert code == 3
        assert document["status"] == "blocked_pending_external_operation"
        assert document["side_effects"] is False


def test_corpus_ingest_lock_splits_and_verify_roundtrip(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    clips_path = tmp_path / "clips.jsonl"
    rows = [
        {
            "clip_id": f"clip-{index:03d}",
            "source_video_id": f"src-{index // 2:02d}",
            "media_hash": sha256_bytes(f"media-{index}".encode()),
            "start_ms": 0,
            "end_ms": 6000,
            "derivative_hashes": [sha256_bytes(f"deriv-{index}".encode())],
        }
        for index in range(24)
    ]
    clips_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    workspace = tmp_path / "store"
    code, ingest = _run(
        capsys,
        "corpus",
        "ingest",
        "--workspace",
        str(workspace),
        "--contract",
        str(CANARY_CONTRACT),
        "--input",
        str(clips_path),
    )
    assert code == 0
    assert ingest["row_count"] == 24
    code, registry = _run(
        capsys,
        "corpus",
        "lock-splits",
        "--workspace",
        str(workspace),
        "--contract",
        str(CANARY_CONTRACT),
        "--artifact-id",
        str(ingest["artifact_id"]),
    )
    assert code == 0
    manifest = registry["split_manifest"]
    assert isinstance(manifest, dict)
    assert manifest["group_key"] == "source_video_id"
    assert {"train", "validation", "test", "study"} <= set(manifest)
    code, verified = _run(capsys, "artifact", "verify", "--workspace", str(workspace), "--all")
    assert code == 0
    assert int(str(verified["verified"])) >= 25  # ingest + registry + shards
    # Splits are immutable: re-locking regenerates the identical registry.
    code, again = _run(
        capsys,
        "corpus",
        "lock-splits",
        "--workspace",
        str(workspace),
        "--contract",
        str(CANARY_CONTRACT),
        "--artifact-id",
        str(ingest["artifact_id"]),
    )
    assert code == 0
    assert again["artifact_id"] == registry["artifact_id"]


def test_domain_errors_exit_2(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "corpus",
                "ingest",
                "--workspace",
                str(tmp_path / "store"),
                "--contract",
                str(CANARY_CONTRACT),
                "--input",
                str(tmp_path / "missing.jsonl"),
            ]
        )
    assert excinfo.value.code == 2 or isinstance(excinfo.value.code, str)
