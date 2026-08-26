"""The study's response reader, its analysis, and the two artifacts they publish."""

from __future__ import annotations

import copy
import json
import tomllib
from pathlib import Path
from typing import Any

import pytest

from dpo.analysis.study import summarize_study
from dpo.cli import main
from dpo.contracts.study_contract import validate_contract
from dpo.core.artifacts import ArtifactStore
from dpo.evaluation.congruency import ClipLadder, ScoredCaption
from dpo.pipeline.study_results_stage import STUDY_RESPONSES_TYPE, STUDY_RESULTS_TYPE, publish_study_results
from dpo.pipeline.study_stage import StudyError, publish_study_export
from dpo.userstudy.responses import (
    RESPONSES_SCHEMA,
    StudyResponseError,
    load_study_responses,
    parse_study_responses,
    participant_hash,
)
from tests.conftest import CANARY_CONTRACT, PreferenceWorld, study_publisher

_CONGRUENCY = (-0.1, 0.05, 0.2, 0.4, 0.6)


def _ladder(clip_id: str) -> ClipLadder:
    """A five-rung ladder, the canary contract's ``study.rungs``."""
    return ClipLadder(
        clip_id=clip_id,
        rungs=tuple(
            ScoredCaption(
                text=f"Rung {index} of {clip_id}, an unseen study caption.",
                congruency=value,
                conditioning="audio" if value < 0.1 else "audio+video",
            )
            for index, value in enumerate(_CONGRUENCY)
        ),
    )


def _export(tmp_path: Path, world: PreferenceWorld) -> tuple[Path, dict[str, Any], str]:
    """A published study export over every study-role clip of the fixture world."""
    publisher, shard_ids = study_publisher(tmp_path, world)
    clip_ids = sorted(shard_ids)
    document, artifact_id = publish_study_export(
        publisher,
        world.contract,
        track=world.pool.track,
        experiment_id="DPO",
        variant_id="base",
        validation_accuracy=0.75,
        ladders=[_ladder(clip_id) for clip_id in clip_ids],
        training_pool=world.pool,
        lock_artifact_id=shard_ids[clip_ids[0]],
        shard_artifact_ids=shard_ids,
        decoding={"temperature": 0.0, "top_p": 1.0},
    )
    return tmp_path / "store", document, artifact_id


def _responses(participant: str, export: dict[str, Any], *, choose: int, rating: int) -> dict[str, Any]:
    rows = []
    for order, clip in enumerate(export["clips"]):
        rung = clip["levels"][choose]
        rows.append(
            {
                "clip_id": clip["clip_id"],
                "presentation_index": order,
                "congruency_position": rung["position"],
                "congruency_index": choose,
                "caption_shown": rung["text"],
                "match_rating": rating,
                "heard_freetext": "a tram and some chatter",
                "slider_moves": 3,
                "time_to_first_move_ms": 1200,
                "response_time_ms": 8000 + order,
                "replay_count": 1,
            }
        )
    return {"schema": RESPONSES_SCHEMA, "participant": participant, "responses": rows}


def test_parse_binds_each_response_to_its_rung_and_hashes_the_participant(
    tmp_path: Path, world: PreferenceWorld
) -> None:
    _, export, _ = _export(tmp_path, world)
    rows = parse_study_responses(_responses("P01", export, choose=3, rating=4), export)
    assert len(rows) == len(export["clips"])
    assert {row.participant_hash for row in rows} == {participant_hash("P01")}
    assert "P01" not in json.dumps([row.document() for row in rows])
    # The rung's measured congruency travels with the response: that is the
    # placement on the axis the study observes.
    assert {row.congruency for row in rows} == {_CONGRUENCY[3]}
    assert [row.presentation_index for row in rows] == list(range(len(rows)))


def test_parse_refuses_what_the_instrument_could_not_have_written(
    tmp_path: Path, world: PreferenceWorld
) -> None:
    _, export, _ = _export(tmp_path, world)
    good = _responses("P02", export, choose=1, rating=3)

    def mutated(**changes: object) -> dict[str, Any]:
        document = copy.deepcopy(good)
        document["responses"][0].update(changes)
        return document

    with pytest.raises(StudyResponseError, match="does not carry"):
        parse_study_responses(mutated(clip_id="clip-elsewhere"), export)
    with pytest.raises(StudyResponseError, match="not rung 1"):
        parse_study_responses(mutated(caption_shown="A caption the export never held."), export)
    with pytest.raises(StudyResponseError, match="places rung 1"):
        parse_study_responses(mutated(congruency_position=0.99), export)
    with pytest.raises(StudyResponseError, match="match_rating"):
        parse_study_responses(mutated(match_rating=6), export)
    with pytest.raises(StudyResponseError, match="unexpected field"):
        parse_study_responses(mutated(extra="field"), export)
    twice = copy.deepcopy(good)
    twice["responses"].append(dict(twice["responses"][0], presentation_index=99))
    with pytest.raises(StudyResponseError, match="twice"):
        parse_study_responses(twice, export)
    with pytest.raises(StudyResponseError, match="schema"):
        parse_study_responses({**good, "schema": "dpo.annotation-responses/v1"}, export)


def test_load_refuses_the_same_participant_in_two_files(tmp_path: Path, world: PreferenceWorld) -> None:
    _, export, _ = _export(tmp_path, world)
    first = tmp_path / "responses-a.json"
    second = tmp_path / "responses-b.json"
    first.write_text(json.dumps(_responses("Same Person", export, choose=0, rating=2)), encoding="utf-8")
    second.write_text(json.dumps(_responses("Same Person", export, choose=4, rating=5)), encoding="utf-8")
    with pytest.raises(StudyResponseError, match="same participant"):
        load_study_responses([first, second], export)
    assert len(load_study_responses([first], export)) == len(export["clips"])


def test_summary_is_deterministic_and_its_intervals_cover_the_point(
    tmp_path: Path, world: PreferenceWorld
) -> None:
    _, export, _ = _export(tmp_path, world)
    responses = (
        *parse_study_responses(_responses("P01", export, choose=4, rating=5), export),
        *parse_study_responses(_responses("P02", export, choose=2, rating=3), export),
        *parse_study_responses(_responses("P03", export, choose=0, rating=1), export),
    )
    first = summarize_study(responses, rungs=5, bootstrap_samples=64)
    second = summarize_study(responses, rungs=5, bootstrap_samples=64)
    assert first == second
    assert first["participants"] == 3 and first["rungs"] == 5
    placement = first["placement"]
    assert isinstance(placement, dict)
    assert placement["rung_histogram"] == [
        len(export["clips"]),
        0,
        len(export["clips"]),
        0,
        len(export["clips"]),
    ]
    for interval in placement["position_ci95"].values():
        assert interval[0] <= placement["mean_position"] <= interval[1]
    rating = first["match_rating"]
    assert isinstance(rating, dict)
    assert rating["mean"] == pytest.approx(3.0)
    assert rating["by_rung"]["4"]["mean_match_rating"] == 5.0
    # Higher-congruency rungs got higher ratings by construction (monotone,
    # not linear: the fixture's rungs are unevenly spaced on the axis).
    correlation = first["congruency_rating_correlation"]
    assert isinstance(correlation, dict)
    assert correlation["pearson_r"] > 0.99
    for interval in correlation["ci95"].values():
        assert interval[0] <= correlation["pearson_r"] <= interval[1]


def test_study_ingest_publishes_the_record_and_the_analysis(tmp_path: Path, world: PreferenceWorld) -> None:
    workspace, export, export_id = _export(tmp_path, world)
    files = []
    for name, choose, rating in (("P01", 3, 4), ("P02", 1, 2)):
        path = tmp_path / f"responses-{name}.json"
        path.write_text(json.dumps(_responses(name, export, choose=choose, rating=rating)), encoding="utf-8")
        files.append(str(path))
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = main(
            [
                *("study", "ingest", "--workspace", str(workspace), "--contract", str(CANARY_CONTRACT)),
                *("--artifact-id", export_id),
                *[argument for path in files for argument in ("--responses", path)],
            ]
        )
    document = json.loads(buffer.getvalue())
    assert code == 0, document
    assert document["status"] == "published"
    assert document["participants"] == 2
    artifacts = document["artifacts"]
    store = ArtifactStore.open(workspace)
    responses_id, results_id = str(artifacts["study_responses"]), str(artifacts["study_results"])
    assert store.verify(responses_id).artifact_type == STUDY_RESPONSES_TYPE
    results_manifest = store.verify(results_id)
    assert results_manifest.artifact_type == STUDY_RESULTS_TYPE
    # The analysis descends from the record, which descends from the export.
    assert [parent.artifact_id for parent in results_manifest.parents] == [responses_id]
    assert [parent.artifact_id for parent in store.verify(responses_id).parents] == [export_id]
    # Both inherit study exposure and both open without a capability: they are
    # what participants chose, not the sealed clip rows.
    assert "study" in results_manifest.role_exposure
    results = json.loads(store.read_payload(results_id))
    assert results["schema"] == STUDY_RESULTS_TYPE and results["responses"] == 2 * len(export["clips"])
    rows = json.loads(store.read_payload(responses_id))["rows"]
    assert {row["participant_hash"] for row in rows} == {participant_hash("P01"), participant_hash("P02")}
    # The export's own view of the ranking stays out of the results payload.
    assert "validation_accuracy" not in results


def test_study_results_refuse_an_export_of_another_rung_count(tmp_path: Path, world: PreferenceWorld) -> None:
    workspace, export, export_id = _export(tmp_path, world)
    with CANARY_CONTRACT.open("rb") as handle:
        document = tomllib.load(handle)
    narrower = copy.deepcopy(document)
    narrower["study"] = {"rungs": 3}
    contract = validate_contract(narrower)
    responses = parse_study_responses(_responses("P01", export, choose=2, rating=3), export)
    publisher, _ = study_publisher(tmp_path / "again", world)
    with pytest.raises(StudyError, match="study.rungs = 3"):
        publish_study_results(
            publisher, contract, export=export, export_artifact_id=export_id, responses=responses
        )
