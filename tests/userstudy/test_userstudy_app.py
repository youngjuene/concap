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
    "congruency_measure": "nats/token: logP(caption|audio,video) - logP(caption|audio)",
    "clips": [
        {
            "clip_id": "clip-a",
            "congruency_span": 0.62,
            "levels": [
                {
                    "position": 0.0,
                    "congruency": -0.02,
                    "conditioning": "audio",
                    "text": "A low rumble and voices.",
                },
                {
                    "position": 0.35,
                    "congruency": 0.20,
                    "conditioning": "audio+video",
                    "text": "A tram rumbles while people talk.",
                },
                {
                    "position": 1.0,
                    "congruency": 0.60,
                    "conditioning": "audio+video",
                    "text": "The tram crossing the square rumbles as people talk beside it.",
                },
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
    levels = study["clips"][0]["levels"]
    # Stops sit where the measure put them — 0.35, not an even third.
    assert [entry["position"] for entry in levels] == [0.0, 0.35, 1.0]
    assert levels[0]["text"].startswith("A low rumble")
    # Which arm produced the caption, how well it scored, and which stop the
    # congruency measure ranked highest must all stay out of the browser:
    # any of them unblinds the judgment being measured.
    serialized = json.dumps(study)
    for leaked in (
        "DPO",
        "validation_accuracy",
        "training_candidate_reuse_rate",
        "congruency",
        "conditioning",
    ):
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
                "congruency_position": 0.35,
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
    assert written["responses"][0]["congruency_position"] == 0.35

    stray = {**body, "responses": [{**body["responses"][0], "clip_id": "clip-elsewhere"}]}
    assert client.post("/api/responses", json=stray).status_code == 400
    assert client.post("/api/responses", json={**body, "schema": "wrong"}).status_code == 400
    assert client.post("/api/responses", json={**body, "participant": " "}).status_code == 400


def test_a_foreign_export_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="schema must be"):
        build_app({"schema": "dpo.selection-report/v1"}, tmp_path, tmp_path / "out")
