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


def _page() -> str:
    from importlib.resources import files

    return str(files("dpo.userstudy").joinpath("page.html").read_text(encoding="utf-8"))


class TestInstrumentInvariants:
    """Assertions against the page source, because the defects these guard were
    invisible to every test that only exercised the server: the slider snapped
    to measured positions while its tick marks were spread evenly, and the
    free-text question claimed the caption had not been seen while showing it.
    """

    def test_marks_are_placed_by_measurement_not_distributed(self) -> None:
        page = _page()
        ticks_rule = page.split("#ticks {", 1)[1].split("}", 1)[0]
        # Flexbox distribution puts marks at even intervals whatever the measure
        # said; absolute placement is the only way a mark can sit on its stop.
        assert "position: relative" in ticks_rule
        assert "space-between" not in ticks_rule
        assert "margin-left" not in page.split("#ticks span {", 1)[1].split("}", 1)[0]
        # And the placement has to read the stop's own position.
        assert 'tick.style.left = "calc(var(--thumb) / 2 + (100% - var(--thumb)) * " + level.position' in page

    def test_what_you_heard_is_asked_before_the_caption_exists(self) -> None:
        page = _page()
        assert page.index('id="heard-text"') < page.index('id="caption"')
        assert page.index('id="hear-step"') < page.index('id="judge-step"')
        # The judging half starts hidden, so the caption is genuinely not on
        # screen while the participant describes what they heard.
        judge = page.split('<div id="judge-step"', 1)[1].split(">", 1)[0]
        assert "hidden" in judge
        assert "before you saw any sentence" not in page

    def test_clip_order_is_per_participant_and_recorded(self) -> None:
        page = _page()
        assert "clips = ordered(studyClips, seedFrom(participant));" in page
        # An order that is not recorded cannot be modelled, which is the only
        # reason to vary it.
        assert "presentation_index: index," in page

    def test_the_clock_starts_when_the_clip_is_watchable(self) -> None:
        page = _page()
        assert 'for (const event of ["loadeddata", "canplaythrough", "playing"])' in page
        # The slider's own latency is measured from when the slider appeared,
        # not from when the clip did.
        assert "Math.round(firstMoveAt - judgeShownAt)" in page
