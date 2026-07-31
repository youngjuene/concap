"""The user-study instrument: what reaches a participant, and what must not."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dpo.userstudy.app import RESPONSES_SCHEMA, build_app

_EXPORT = {
    "schema": "dpo.study-export/v1",
    "track": "audio",
    "experiment_id": "DPO",
    "variant_id": "base",
    "validation_accuracy": 0.83,
    "training_candidate_reuse_rate": 0.0,
    "congruency_ladder": [
        {"level": 0.0, "conditioning": "audio", "instruction": "heard only"},
        {"level": 0.5, "conditioning": "audio+video", "instruction": "name the source"},
        {"level": 1.0, "conditioning": "audio+video", "instruction": "bind sound to sight"},
    ],
    "clips": [
        {
            "clip_id": "clip-a",
            "levels": [
                {"level": 0.0, "text": "A low rumble and voices."},
                {"level": 0.5, "text": "A tram rumbles while people talk."},
                {"level": 1.0, "text": "The tram crossing the square rumbles as people talk beside it."},
            ],
        }
    ],
}


def _client(tmp_path: Path) -> TestClient:
    media = tmp_path / "media"
    (media / "unmuted_video").mkdir(parents=True)
    (media / "unmuted_video" / "clip-a.mp4").write_bytes(b"video-with-sound")
    return TestClient(build_app(_EXPORT, media, tmp_path / "responses"))


def test_participant_sees_the_ladder_and_nothing_that_unblinds_it(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert client.get("/").status_code == 200
    study = client.get("/api/study").json()
    assert study["congruency_levels"] == [0.0, 0.5, 1.0]
    assert [entry["text"] for entry in study["clips"][0]["levels"]][0].startswith("A low rumble")
    # Which arm produced the caption, and how well it scored, must not reach a
    # participant — knowing it would unblind the judgment being measured.
    serialized = json.dumps(study)
    for leaked in ("DPO", "validation_accuracy", "training_candidate_reuse_rate", "instruction"):
        assert leaked not in serialized


def test_media_route_serves_the_clip_with_its_soundtrack(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert client.get("/media/clip-a").status_code == 200
    assert client.get("/media/missing").status_code == 404
    assert client.get("/media/..%2Fetc").status_code in {400, 404}


def test_responses_round_trip_and_reject_foreign_clips(tmp_path: Path) -> None:
    client = _client(tmp_path)
    body = {
        "schema": RESPONSES_SCHEMA,
        "participant": "P01",
        "responses": [
            {
                "clip_id": "clip-a",
                "congruency_level": 0.5,
                "congruency_index": 1,
                "caption_shown": "A tram rumbles while people talk.",
                "match_rating": 4,
                "heard_freetext": "a tram and chatter",
                "slider_moves": 3,
                "time_to_first_move_ms": 1800,
                "response_time_ms": 9100,
                "replay_count": 1,
            }
        ],
    }
    saved = client.post("/api/responses", json=body)
    assert saved.status_code == 200 and saved.json()["responses"] == 1
    written = json.loads((tmp_path / "responses" / "responses-P01.json").read_text())
    assert written["responses"][0]["congruency_level"] == 0.5

    stray = {**body, "responses": [{**body["responses"][0], "clip_id": "clip-elsewhere"}]}
    assert client.post("/api/responses", json=stray).status_code == 400
    assert client.post("/api/responses", json={**body, "schema": "wrong"}).status_code == 400
    assert client.post("/api/responses", json={**body, "participant": " "}).status_code == 400


def test_a_foreign_export_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="schema must be"):
        build_app({"schema": "dpo.selection-report/v1"}, tmp_path, tmp_path / "out")
