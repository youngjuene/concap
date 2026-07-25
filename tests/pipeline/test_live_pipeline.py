"""The whole pipeline through the CLI, on the offline backend, with no GPU.

Corpus ingest, split lock, candidate generation, task export, a simulated
collection session, annotation ingest, view derivation, the nine-condition
matrix, and selection — every stage as its real command, over real artifacts,
ending in a published lock manifest. This is the executable proof that the live
commands compose; the canary proves the same stages compose in process.

The fixture contract differs from the canary contract in three places, because
the offline seed model's captions are hex transcriptions of raw byte output
rather than English sentences (see ``dpo.candidates.generation``):

* the caption length bounds and the completion budget are widened to admit a
  transcription, which no other fixture has to survive;
* the decoding mixture samples only. The tiny model's temperature-0 generation
  collapses to one repeated byte for many clips, so every controlled error —
  which is by construction another clip's greedy caption — would be a genuine
  near-duplicate of some other split's greedy caption, and the cross-split
  leakage audit would (correctly) refuse to publish the views.

Every gate still runs at full strength against this contract; none is relaxed.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Any

import pytest

from dpo.cli import main
from dpo.core.identity import sha256_bytes
from tests.annotation.test_collection import _simulated_responses
from tests.conftest import CANARY_CONTRACT

TRACKS = ("visual", "audio")
SPLITS = ("train", "validation")
ANNOTATORS = ("Annotator One", "Annotator Two", "Annotator Three")

_SAMPLING_ONLY_CANDIDATES = """[candidates]
source_policy = "C0"
per_clip = 4
challenge_fraction_min = 0.0
challenge_fraction_max = 0.0
controlled_error_rate = 0.0
max_new_tokens = 48
generation_seed = 1

[[candidates.decoding]]
name = "sample"
temperature = 0.7
top_p = 0.9
count = 2

[[candidates.decoding]]
name = "alt_decoding"
temperature = 1.0
top_p = 0.95
count = 2

"""


def invoke(*argv: str) -> dict[str, Any]:
    """One command, its single JSON document, and a required exit code of 0."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = main(list(argv))
    document = json.loads(buffer.getvalue())
    assert code == 0, document
    assert isinstance(document, dict)
    return document


def _fixture_contract(tmp_path: Path) -> str:
    source = CANARY_CONTRACT.read_text(encoding="utf-8")
    start = source.index("[candidates]")
    end = source.index("[pairs]")
    patched = source[:start] + _SAMPLING_ONLY_CANDIDATES + source[end:]
    patched = patched.replace("max_words = 30", "max_words = 160")
    patched = patched.replace("max_completion_tokens = 128", "max_completion_tokens = 512")
    path = tmp_path / "fixture-contract.toml"
    path.write_text(patched, encoding="utf-8")
    return str(path)


def _corpus_file(tmp_path: Path) -> Path:
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
    path = tmp_path / "clips.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _collect(
    tmp_path: Path, *, workspace: str, contract: str, registry_id: str, track: str, split: str
) -> dict[str, str]:
    """Generate candidates, export tasks, simulate a session, and ingest it."""
    generated = invoke(
        *("candidates", "generate", "--workspace", workspace, "--contract", contract),
        *("--artifact-id", registry_id, "--track", track, "--split", split),
        *("--dataset-version", "live-cli/v1"),
    )
    pool_id = str(generated["artifact_id"])
    tasks_path = tmp_path / f"tasks-{track}-{split}.json"
    answers_path = tmp_path / f"answers-{track}-{split}.json"
    invoke(
        *("annotation", "export-tasks", "--workspace", workspace, "--contract", contract),
        *("--artifact-id", pool_id, "--artifact-id", registry_id),
        *("--out-tasks", str(tasks_path), "--out-answers", str(answers_path)),
    )
    tasks_document = json.loads(tasks_path.read_text(encoding="utf-8"))
    answers_document = json.loads(answers_path.read_text(encoding="utf-8"))
    response_paths = []
    for index, annotator in enumerate(ANNOTATORS):
        responses = _simulated_responses(tasks_document, answers_document, annotator=annotator)
        path = tmp_path / f"responses-{track}-{split}-{index}.json"
        path.write_text(json.dumps(responses, ensure_ascii=False), encoding="utf-8")
        response_paths.append(str(path))
    ingested = invoke(
        *("annotation", "ingest", "--workspace", workspace, "--contract", contract),
        *("--artifact-id", pool_id, "--split", split),
        *("--tasks", str(tasks_path), "--answers", str(answers_path)),
        *[argument for path in response_paths for argument in ("--responses", path)],
    )
    assert int(str(ingested["retained"])) > 0
    artifacts = ingested["artifacts"]
    assert isinstance(artifacts, dict)
    return {"pool": pool_id, "annotations": str(artifacts["raw_annotations"])}


def _train(workspace: str, contract: str, view_ids: list[str], checkpoints: Path) -> dict[str, Any]:
    return invoke(
        *("train", "run", "--workspace", workspace, "--contract", contract),
        *[argument for artifact_id in view_ids for argument in ("--artifact-id", artifact_id)],
        *("--checkpoint-dir", str(checkpoints)),
    )


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One full offline run of the live commands, shared by every assertion."""
    tmp_path = tmp_path_factory.mktemp("live-pipeline")
    workspace = str(tmp_path / "store")
    contract = _fixture_contract(tmp_path)
    ingest = invoke(
        *("corpus", "ingest", "--workspace", workspace, "--contract", contract),
        *("--input", str(_corpus_file(tmp_path))),
    )
    registry = invoke(
        *("corpus", "lock-splits", "--workspace", workspace, "--contract", contract),
        *("--artifact-id", str(ingest["artifact_id"])),
    )
    registry_id = str(registry["artifact_id"])
    views_by_track: dict[str, dict[str, str]] = {}
    for track in TRACKS:
        collected = {
            split: _collect(
                tmp_path,
                workspace=workspace,
                contract=contract,
                registry_id=registry_id,
                track=track,
                split=split,
            )
            for split in SPLITS
        }
        derived = invoke(
            *("views", "derive", "--workspace", workspace, "--contract", contract),
            *("--artifact-id", registry_id),
            *("--artifact-id", collected["train"]["pool"]),
            *("--artifact-id", collected["validation"]["pool"]),
            *("--artifact-id", collected["train"]["annotations"]),
            *("--artifact-id", collected["validation"]["annotations"]),
            *("--track", track),
        )
        artifacts = derived["artifacts"]
        assert isinstance(artifacts, dict)
        views_by_track[track] = {key: str(value) for key, value in artifacts.items()}
        assert int(str(derived["rows"]["sft"])) > 0
        assert int(str(derived["rows"]["pair_strict"])) > 0
        assert int(str(derived["rows"]["validation_pairs"])) > 0
    checkpoints = tmp_path / "checkpoints"
    training_inputs = [
        views_by_track[track][key] for track in TRACKS for key in ("sft", "pair_strict", "pair_all")
    ]
    trained = _train(workspace, contract, training_inputs, checkpoints)
    cell_ids = [str(row["artifact_id"]) for row in trained["matrix"]]
    selection_inputs = [
        views_by_track[track][key] for track in TRACKS for key in ("pair_strict", "validation_pairs")
    ]
    selected = invoke(
        *("select", "run", "--workspace", workspace, "--contract", contract),
        *[
            argument
            for artifact_id in (*selection_inputs, *cell_ids)
            for argument in ("--artifact-id", artifact_id)
        ],
        *("--checkpoint-dir", str(checkpoints)),
    )
    return {
        "tmp_path": tmp_path,
        "workspace": workspace,
        "contract": contract,
        "views": views_by_track,
        "training_inputs": training_inputs,
        "trained": trained,
        "selected": selected,
        "cell_ids": cell_ids,
        "checkpoints": checkpoints,
    }


def test_views_derive_publishes_every_view_of_both_tracks(pipeline: dict[str, Any]) -> None:
    views = pipeline["views"]
    for track in TRACKS:
        assert set(views[track]) == {"sft", "pair_strict", "pair_all", "validation_pairs"}
    # The two tracks never share a view: same clips, separate everything else.
    everything = [artifact_id for track in TRACKS for artifact_id in views[track].values()]
    assert len(set(everything)) == len(everything)


def test_train_run_publishes_the_whole_matrix(pipeline: dict[str, Any]) -> None:
    trained = pipeline["trained"]
    assert trained["status"] == "published"
    assert trained["backend"] == "tiny"
    assert int(str(trained["cells"])) == 18  # nine experiments, two tracks
    assert int(str(trained["cells_trained"])) == 18
    assert int(str(trained["cells_resumed"])) == 0
    matrix = trained["matrix"]
    assert {str(row["experiment_id"]) for row in matrix} == {
        "SEED",
        "SFT",
        "DPO",
        "IPO",
        "CDPO",
        "RDPO",
        "DRDPO",
        "WDPO",
        "SFT_DPO",
    }
    seed_cells = [row for row in matrix if row["experiment_id"] == "SEED"]
    assert [int(str(row["steps"])) for row in seed_cells] == [0, 0]
    assert all(int(str(row["steps"])) > 0 for row in matrix if row["experiment_id"] != "SEED")


def test_checkpoints_carry_every_trained_policy(pipeline: dict[str, Any]) -> None:
    from dpo.pipeline.live_runner import scan_checkpoints

    cells = scan_checkpoints(pipeline["checkpoints"])
    assert len(cells) == 18
    for cell, directory in cells:
        assert (directory / "cell.json").is_file()
        assert (directory / "adapter_model.safetensors").is_file() == cell.trained


def test_select_run_ranks_every_experiment_and_publishes_the_lock(pipeline: dict[str, Any]) -> None:
    selected = pipeline["selected"]
    ranking = selected["ranking"]
    assert isinstance(ranking, dict)
    assert set(ranking) == set(TRACKS)
    for track in TRACKS:
        assert len(ranking[track]) == 9
        assert set(ranking[track]) == set(selected["selected_variants"][track])
        assert set(selected["validation_accuracy"][track]) == set(ranking[track])
    assert str(selected["lock_id"]).startswith("sha256:")
    artifacts = selected["artifacts"]
    assert isinstance(artifacts, dict)
    assert set(artifacts) == {"validation_report", "selection_report", "lock_manifest"}


def test_a_second_train_run_resumes_every_cell(pipeline: dict[str, Any]) -> None:
    again = _train(
        pipeline["workspace"],
        pipeline["contract"],
        pipeline["training_inputs"],
        pipeline["checkpoints"],
    )
    assert int(str(again["cells_resumed"])) == 18
    assert int(str(again["cells_trained"])) == 0
    assert all(bool(row["resumed"]) for row in again["matrix"])
    # Resumption is not a new result: the published cells are the same artifacts.
    assert [str(row["artifact_id"]) for row in again["matrix"]] == pipeline["cell_ids"]


def test_published_reports_verify_in_the_store(pipeline: dict[str, Any]) -> None:
    workspace = str(pipeline["workspace"])
    verified = invoke("artifact", "verify", "--workspace", workspace, "--all")
    assert int(str(verified["verified"])) > 40
    reports = invoke("report", "show", "--workspace", workspace)["reports"]
    assert isinstance(reports, dict)
    assert len(reports["selection"]) == 1
    assert len(reports["locks"]) == 1
    assert reports["selection"][0]["ranking"] == pipeline["selected"]["ranking"]
