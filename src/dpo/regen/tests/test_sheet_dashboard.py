"""Spreadsheet responses stay inspectable across the shared-store handoff."""

from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from fastapi.routing import APIRoute

from dpo.regen.dashboard import build_dashboard_app
from dpo.regen.dashboard_data import DashboardData
from dpo.regen.sheet_instrument import page_spec, snapshot
from dpo.regen.study_store import StudyStore

PROTOCOL = "dpo.sheet-questionnaire/v1"
POLICY = "interaction-only-sheet-v1"
OPTIONS = {"A1s": [{"value": "wind", "label": "바람"}], "R2": [{"value": "wind", "label": "바람"}]}
STAMP = 1791068400
PRIVATE = "private-field-must-not-export"


def _submission(page: str, answers: dict[str, Any]) -> dict[str, Any]:
    items = page_spec(page, session_key="sheet-person", options=OPTIONS)["items"]
    original = page in {"P2", "P3", "P4"}
    return {
        "page": page,
        "clip_index": 0,
        "clip_id": "short-1",
        "view_id": f"sheet-person-0-{'P1' if original else 'P6'}",
        "condition": "prepared" if original else "regenerated",
        "answers": answers,
        "items": items,
        "presented_order": [item["id"] for item in reversed(items)],
        "instrument_hash": snapshot()["hash"],
        "submitted_at": STAMP,
        "client_submitted_at": "2026-10-04T00:19:59Z",
        "analysis_excluded": False,
    }


def _fixture(tmp_path: Path) -> tuple[DashboardData, StudyStore, str, str]:
    legacy, viewing = tmp_path / "legacy", tmp_path / "viewing"
    legacy.mkdir()
    store = StudyStore(viewing / "study.sqlite3")
    token = store.create("calibration", "ko")
    rating = page_spec("P4")["items"][0]["id"]
    submissions = [
        _submission("P2", {"V1": [{"frame_id": "0", "x": 0, "y": 0.25, "token": PRIVATE}]}),
        _submission("P3", {"A1s": "wind"}),
        _submission("P4", {rating: 0}),
        _submission("P11", {"R2": ["memory_unknown"]}),
    ]
    excluded = deepcopy(submissions[2])
    excluded.update(page="practice", analysis_excluded=True)

    def initialize(state: dict[str, Any]) -> None:
        state.update(
            session_id="sheet-person",
            stage="sheet-questionnaire",
            sheet_protocol=PROTOCOL,
            sheet_page="P11",
            sheet_instrument=snapshot(),
            sheet_responses=submissions,
            sheet_viewings=[],
            sheet_practice=[excluded],
            sheet_clip_order=["clip1"],
            survey_policy=POLICY,
        )

    store.mutate(token, "init", 0, json.dumps({"action": "sheet-enrollment"}), initialize)
    return DashboardData(legacy, viewing), store, token, rating


def test_sheet_answers_preserve_zero_choice_multi_visual_and_provenance(tmp_path: Path) -> None:
    data, _, token, rating = _fixture(tmp_path)
    overview = data.overview()
    participant = overview["participants"][0]
    assert participant["label"] == "sheet-person"
    assert participant["phase1_status"] == "in_progress"
    assert participant["phase2_started"] is False
    assert overview["metrics"]["survey_submissions"] == 4
    assert overview["metrics"]["response_values"] == 4
    rows = {row["item_id"]: row for row in overview["responses"]}
    assert rows[rating]["value"] == 0
    assert (rows[rating]["min"], rows[rating]["max"], rows[rating]["points"]) == (0, 6, 7)
    assert len(rows[rating]["labels"]) == 7
    assert rows[rating]["source_row"].startswith("문항표!")
    assert rows[rating]["source_row_hash"]
    assert rows[rating]["presentation_index"] == rows[rating]["presented_order"].index(rating)
    assert rows[rating]["instrument"] == snapshot()["hash"]
    assert rows[rating]["flow_version"] == PROTOCOL
    assert rows[rating]["condition"] == "prepared"
    assert rows[rating]["submitted_at"] == "2026-10-03T23:00:00.000Z"
    assert rows[rating]["client_submitted_at"] == "2026-10-04T00:19:59.000Z"
    assert rows["A1s"]["value"] == "wind"
    assert rows["A1s"]["options"] == ["wind"]
    assert rows["A1s"]["option_labels"] == ["바람"]
    assert rows["R2"]["value"] == ["memory_unknown"]
    assert rows["V1"]["value"] == [{"frame_id": "0", "x": 0, "y": 0.25}]
    detail = data.participant(participant["id"])
    assert not any("no readable answers" in warning for warning in detail["warnings"])
    assert token not in json.dumps(detail)
    assert PRIVATE not in json.dumps(detail)


def test_handoff_mirror_deduplicates_and_excludes_practice(tmp_path: Path) -> None:
    data, store, token, _ = _fixture(tmp_path)
    before = data.overview()

    def handoff(state: dict[str, Any]) -> None:
        state.update(stage="ready", sheet_page="handoff")
        state["calibration_source"] = {
            "schema": PROTOCOL,
            "participant": "sheet-person",
            "responses": deepcopy(state["sheet_responses"] + state["sheet_practice"]),
            "viewings": [],
        }

    store.mutate(token, "handoff", 1, json.dumps({"action": "handoff"}), handoff)
    after = data.overview()
    assert after["participants"][0]["id"] == before["participants"][0]["id"]
    assert after["participants"][0]["phase1_status"] == "done"
    assert after["metrics"]["response_values"] == 4
    assert after["metrics"]["survey_submissions"] == 4
    assert after["responses"] == before["responses"]


def test_sheet_viewings_events_and_condition_stay_phase_scoped(tmp_path: Path) -> None:
    data, store, token, rating = _fixture(tmp_path)

    def record(state: dict[str, Any]) -> None:
        state["sheet_viewings"] = [
            {
                "view_id": f"sheet-person-0-{page}",
                "clip_index": 0,
                "clip_id": "short-1",
                "page": page,
                "condition": condition,
                "started_at": STAMP,
                "ended_at": STAMP + 10,
                "stimulus_assignment": "variant-b",
                "captions": [{"text": "저장된 자막", "start_ms": 0, "end_ms": 1000}],
            }
            for page, condition in (("P1", "prepared"), ("P6", "regenerated"))
        ]
        state["sheet_regenerations"] = {"clip1": {"fallback": True}}
        state["calibration_source"] = {
            "participant": "sheet-person",
            "viewings": deepcopy(state["sheet_viewings"]),
        }

    store.mutate(token, "view", 1, json.dumps({"action": "view-ended", "data": {"page": "P6"}}), record)
    store.mutate(
        token,
        "practice",
        2,
        json.dumps({"action": "practice-complete", "data": {"page": "practice"}}),
        lambda _: None,
    )
    store.mutate(
        token,
        "control",
        3,
        json.dumps(
            {
                "action": "settings",
                "data": {
                    "video_id": "long-1",
                    "texture": 0,
                    "context": 1,
                    "position_hint_ms": 0,
                },
            }
        ),
        lambda _: None,
    )
    participant = data.overview()["participants"][0]
    detail = data.participant(participant["id"])
    assert len(detail["captions"]) == 2
    assert {row["kind"] for row in detail["captions"]} == {"prepared", "fallback"}
    assert all(row["created_at"] == "2026-10-03T23:00:00.000Z" for row in detail["captions"])
    assert (
        next(row for row in detail["responses"] if row["item_id"] == rating)["stimulus_assignment"]
        == "variant-b"
    )
    event_phases = {event["kind"]: event["phase"] for event in detail["events"]}
    assert event_phases["view-ended"] == 1
    assert event_phases["settings"] == 2
    assert "practice-complete" not in event_phases
    assert detail["interactions"][0]["texture"] == 0


@pytest.mark.parametrize("stage", ["poststudy", "debrief", "done"])
def test_interaction_only_endings_never_count_phase2_surveys(tmp_path: Path, stage: str) -> None:
    data, store, token, _ = _fixture(tmp_path)

    def complete(state: dict[str, Any]) -> None:
        state.update(
            stage=stage,
            sheet_page="handoff",
            poststudy={"acknowledged": True},
            debrief={"acknowledged": True},
            final_survey={"answers": {"obsolete": 5}},
            video_surveys=[{"video_id": "long-1", "answers": {"obsolete": 4}}],
        )

    store.mutate(token, "finish", 1, json.dumps({"action": "debrief-complete"}), complete)
    overview = data.overview()
    person = overview["participants"][0]
    assert person["phase1_status"] == "done"
    assert person["phase2_started"] is True
    assert person["phase2_status"] == stage
    assert person["phase2_surveys"] == 0
    assert all(row["phase"] == 1 for row in overview["responses"])
    assert overview["metrics"]["survey_submissions"] == 4


def test_factorial_assignment_and_strategy_survive_json_and_csv(tmp_path: Path) -> None:
    data, store, token, _ = _fixture(tmp_path)
    assignment = {"audiovisual_congruence": "incongruent", "description_depth": "deep"}

    def conditions(state: dict[str, Any]) -> None:
        for response in state["sheet_responses"]:
            response.update(
                condition="second",
                caption_strategy="opposite",
                stimulus_assignment={**assignment, "token": PRIVATE},
            )

    store.mutate(token, "conditions", 1, "conditions", conditions)
    overview = data.overview()
    assert all(row["stimulus_assignment"] == assignment for row in overview["responses"])
    assert all(row["caption_strategy"] == "opposite" for row in overview["responses"])
    app = build_dashboard_app(data.legacy_dir, data.viewing_dir)
    endpoint = next(
        route.endpoint
        for route in app.routes
        if isinstance(route, APIRoute) and route.path == "/api/export/responses.csv"
    )
    payload = bytes(endpoint(include_qa=True).body).decode("utf-8-sig")
    csv_rows = list(csv.DictReader(io.StringIO(payload)))
    assert len(csv_rows) == 4
    assert all(json.loads(row["stimulus_assignment"]) == assignment for row in csv_rows)
    assert all(row["caption_strategy"] == "opposite" for row in csv_rows)
    assert PRIVATE not in payload


@pytest.mark.parametrize("layout", ["standalone", "same_directory", "same_inode", "stale_copy"])
def test_sqlite_sources_are_read_once_and_latest_same_session_is_used(tmp_path: Path, layout: str) -> None:
    data, store, token, _ = _fixture(tmp_path)
    source = store.path
    alternate = data.legacy_dir / "questionnaire.sqlite3"
    if layout == "same_directory":
        data = DashboardData(data.viewing_dir, data.viewing_dir)
        alternate = data.viewing_dir / "questionnaire.sqlite3"
    if layout == "same_inode":
        os.link(source, alternate)
    else:
        with sqlite3.connect(source) as origin, sqlite3.connect(alternate) as copied:
            origin.backup(copied)
    if layout == "standalone":
        source.unlink()
    if layout == "stale_copy":
        store.mutate(
            token,
            "newer",
            1,
            json.dumps(
                {
                    "action": "settings",
                    "data": {
                        "phase": 2,
                        "video_id": "long-1",
                        "texture": 0,
                        "context": 1,
                    },
                }
            ),
            lambda state: state.update(stage="watch"),
        )
    before = {path: path.read_bytes() for path in (source, alternate) if path.is_file()}
    overview = data.overview()
    assert overview["metrics"]["participants"] == 1
    assert overview["metrics"]["survey_submissions"] == 4
    assert overview["metrics"]["response_values"] == 4
    person = overview["participants"][0]
    assert person["control_changes"] == (1 if layout == "stale_copy" else 0)
    detail = data.participant(person["id"])
    assert sum(event["kind"] == "sheet-enrollment" for event in detail["events"]) == 1
    assert not any("database is unavailable" in warning for warning in overview["warnings"])
    assert {path: path.read_bytes() for path in before} == before


def test_explicit_server_event_scope_is_preserved_without_inventing_historical_scope(tmp_path: Path) -> None:
    data, store, token, _ = _fixture(tmp_path)
    scope = {
        "phase": 1,
        "page": "P9",
        "stage": "P9",
        "clip_index": 1,
        "clip_id": "short-2",
        "view_id": "sheet-person-1-P6",
        "condition": "second",
        "protocol": PROTOCOL,
        "session_id": "sheet-person",
        "position_ms": 0,
    }
    store.mutate(token, "scoped", 1, json.dumps({"action": "submit", "data": scope}), lambda _: None)
    store.mutate(
        token, "historic", 2, json.dumps({"action": "playback", "data": {"page": "P1"}}), lambda _: None
    )
    person = data.overview()["participants"][0]
    detail = data.participant(person["id"])
    event = next(event for event in detail["events"] if event["kind"] == "submit")
    assert event["phase"] == 1
    for key in ("stage", "clip_index", "clip_id", "view_id", "condition", "session_id", "protocol"):
        assert event[key] == scope[key]
    assert event["video_id"] == "short-2"
    assert event["flow_version"] == PROTOCOL
    assert event["position_ms"] == 0
    historical = next(event for event in detail["events"] if event["kind"] == "playback")
    assert historical["clip_id"] is None and historical["view_id"] is None
    assert historical["clip_index"] is None and historical["video_id"] is None


def test_distinct_standalone_and_linked_sessions_are_both_collected(tmp_path: Path) -> None:
    data, store, token, _ = _fixture(tmp_path)
    original = store.state(token)
    standalone = StudyStore(data.legacy_dir / "questionnaire.sqlite3")
    other = standalone.create("calibration", "ko")

    def initialize(state: dict[str, Any]) -> None:
        state.update(deepcopy(original))
        state.update(session_id="another-person", revision=0)
        for row in state["sheet_responses"]:
            row["view_id"] = row["view_id"].replace("sheet-person", "another-person")

    standalone.mutate(other, "init", 0, json.dumps({"action": "sheet-enrollment"}), initialize)
    overview = data.overview()
    assert {person["label"] for person in overview["participants"]} == {"sheet-person", "another-person"}
    assert overview["metrics"]["survey_submissions"] == 8
    assert overview["metrics"]["response_values"] == 8
