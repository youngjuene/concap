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


def test_report_show_lists_the_published_reports(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    from dpo.pipeline.canary import run_canary

    run_canary(tmp_path / "store", CANARY_CONTRACT)
    code, document = _run(capsys, "report", "show", "--workspace", str(tmp_path / "store"))
    assert code == 0
    assert document["status"] == "ok"
    reports = document["reports"]
    assert isinstance(reports, dict)
    assert len(reports["selection"]) == 1
    ranking = reports["selection"][0]["ranking"]
    assert set(ranking) == {"visual", "audio"}
    assert len(ranking["visual"]) == 9  # the nine-condition matrix, ranked
    assert reports["selection"][0]["selected_variants"]["visual"].keys() == set(ranking["visual"])
    assert reports["selection"][0]["selected_hyperparameters"]["visual"]["DPO"]
    assert len(reports["validation"]) == 1
    assert set(reports["validation"][0]["accuracy"]) == {"visual", "audio"}
    assert len(reports["locks"]) == 1
    assert str(reports["locks"][0]["lock_id"]).startswith("sha256:")


def _locked_corpus(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> tuple[str, dict[str, object]]:
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
    workspace = str(tmp_path / "store")
    code, ingest = _run(
        capsys,
        "corpus",
        "ingest",
        "--workspace",
        workspace,
        "--contract",
        str(CANARY_CONTRACT),
        "--input",
        str(clips_path),
    )
    assert code == 0
    code, registry = _run(
        capsys,
        "corpus",
        "lock-splits",
        "--workspace",
        workspace,
        "--contract",
        str(CANARY_CONTRACT),
        "--artifact-id",
        str(ingest["artifact_id"]),
    )
    assert code == 0
    return workspace, registry


def test_candidates_generate_publishes_a_pool_that_exports_tasks(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    workspace, registry = _locked_corpus(capsys, tmp_path)
    code, generated = _run(
        capsys,
        "candidates",
        "generate",
        "--workspace",
        workspace,
        "--contract",
        str(CANARY_CONTRACT),
        "--artifact-id",
        str(registry["artifact_id"]),
        "--track",
        "visual",
        "--split",
        "train",
        "--dataset-version",
        "cli/v1",
    )
    assert code == 0
    assert generated["status"] == "published"
    assert generated["operation"] == "candidates-generate"
    clips = int(str(generated["clips"]))
    assert clips > 0
    assert int(str(generated["candidates"])) == clips * 4  # canary candidates.per_clip
    assert int(str(generated["pairs"])) >= clips * 3  # canary pairs.per_clip_min
    out_tasks = tmp_path / "tasks.json"
    out_answers = tmp_path / "answers.json"
    code, exported = _run(
        capsys,
        "annotation",
        "export-tasks",
        "--workspace",
        workspace,
        "--contract",
        str(CANARY_CONTRACT),
        "--artifact-id",
        str(generated["artifact_id"]),
        "--artifact-id",
        str(registry["artifact_id"]),
        "--out-tasks",
        str(out_tasks),
        "--out-answers",
        str(out_answers),
    )
    assert code == 0
    assert int(str(exported["tasks"])) > 0
    tasks_document = json.loads(out_tasks.read_text(encoding="utf-8"))
    assert tasks_document["schema"] == "dpo.collection-tasks/v1"
    assert len(tasks_document["tasks"]) > 0
    assert out_answers.is_file()


def test_candidates_generate_blocks_on_a_live_seed_model(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    live_contract = tmp_path / "live-seed.toml"
    live_contract.write_text(
        CANARY_CONTRACT.read_text(encoding="utf-8").replace(
            'implementation = "dpo.models.tiny.TinyAdapter"',
            'implementation = "vendor.models.RealSeedAdapter"',
        ),
        encoding="utf-8",
    )
    workspace = tmp_path / "store"
    code, document = _run(
        capsys,
        "candidates",
        "generate",
        "--workspace",
        str(workspace),
        "--contract",
        str(live_contract),
        "--artifact-id",
        "sha256:" + "0" * 64,
        "--track",
        "visual",
        "--split",
        "train",
        "--dataset-version",
        "cli/v1",
    )
    assert code == 3
    assert document["status"] == "blocked_pending_external_operation"
    assert document["command"] == "candidates generate"
    assert document["side_effects"] is False
    assert not workspace.exists()  # blocked before any workspace side effect


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
