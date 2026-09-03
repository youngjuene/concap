"""The HTTP surface: the six steps in order, and what each one refuses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from dpo.regen.app import build_app
from dpo.regen.items import load_items
from dpo.regen.regeneration import RegenTemplateWriter

POINTS = [{"x": 0.20, "y": 0.20}, {"x": 0.40, "y": 0.80}]


@pytest.fixture
def client(document: dict[str, Any], media_dir: Path, tmp_path: Path) -> TestClient:
    app = build_app(document, media_dir, tmp_path / "out", RegenTemplateWriter())
    return TestClient(app)


def enrol(client: TestClient) -> str:
    body = client.post("/api/session", json={}).json()
    return str(body["participant"])


def submit(client: TestClient, participant: str, page: str) -> Any:
    """Post a complete submission without reading the page first.

    The order tests need the *submit* refused, and ``/api/step`` is gated too,
    so going through it would test the gate on the wrong route.
    """
    responses = {item: 4 for item in load_items().item_ids(page)}
    return client.post(
        "/api/survey",
        json={
            "participant": participant,
            "page": page,
            "responses": responses,
            "entered_at": "t0",
            "submitted_at": "t1",
        },
    )


def answer(client: TestClient, participant: str, page: str) -> Any:
    """Read the page, then answer every item it asks — the participant's path."""
    blocks = client.get(f"/api/step/{page}?participant={participant}").json()["blocks"]
    responses = {item["id"]: 4 for block in blocks for item in block["items"]}
    return client.post(
        "/api/survey",
        json={
            "participant": participant,
            "page": page,
            "responses": responses,
            "entered_at": "t0",
            "submitted_at": "t1",
        },
    )


def walk_to_regenerated(client: TestClient) -> str:
    """Every step up to and including the regeneration."""
    participant = enrol(client)
    client.post(
        "/api/viewing",
        json={"participant": participant, "step": "view_prepared", "started_at": "t0", "ended_at": "t1"},
    )
    answer(client, participant, "art")
    client.post("/api/visual", json={"participant": participant, "points": POINTS})
    client.post("/api/auditory", json={"participant": participant, "selected": ["traffic"], "lanes": {}})
    client.post("/api/regenerate", json={"participant": participant})
    return participant


class TestEntry:
    def test_a_session_issues_an_identifier_and_fixes_the_assignment(self, client: TestClient) -> None:
        body = client.post("/api/session", json={}).json()
        assert body["participant"]
        assert body["assignment"]["prepared_segment"] in ("A", "B")
        assert body["step"] == "view_prepared"

    def test_consecutive_participants_alternate_which_segment_is_prepared(self, client: TestClient) -> None:
        first = client.post("/api/session", json={}).json()["assignment"]
        second = client.post("/api/session", json={}).json()["assignment"]
        assert first["prepared_segment"] != second["prepared_segment"]

    def test_re_entry_returns_the_same_assignment(self, client: TestClient) -> None:
        # A reload must not re-roll the condition, or the two viewings stop
        # being the pair the design calls for.
        first = client.post("/api/session", json={}).json()
        again = client.post("/api/session", json={"participant": first["participant"]}).json()
        assert again["assignment"] == first["assignment"]
        assert again["participant"] == first["participant"]

    def test_the_operator_is_told_which_item_set_is_loaded(self, client: TestClient) -> None:
        assert client.post("/api/session", json={}).json()["items_provenance"] == "placeholder"


class TestOrder:
    def test_the_whole_session_walks_forward(self, client: TestClient) -> None:
        participant = walk_to_regenerated(client)
        assert client.get(f"/api/state?participant={participant}").json()["step"] == "view_regenerated"
        client.post(
            "/api/viewing",
            json={
                "participant": participant,
                "step": "view_regenerated",
                "started_at": "t2",
                "ended_at": "t3",
            },
        )
        assert answer(client, participant, "survey").json()["step"] == "done"

    def test_a_skipped_step_is_refused(self, client: TestClient) -> None:
        participant = enrol(client)
        response = submit(client, participant, "art")
        assert response.status_code == 409
        assert "run in order" in response.json()["error"]

    def test_a_completed_step_cannot_be_re_entered(self, client: TestClient) -> None:
        participant = walk_to_regenerated(client)
        response = submit(client, participant, "art")
        assert response.status_code == 409
        assert "already complete" in response.json()["error"]

    def test_a_refusal_says_where_the_session_actually_is(self, client: TestClient) -> None:
        participant = enrol(client)
        assert submit(client, participant, "art").json()["step"] == "view_prepared"


class TestViewings:
    def test_the_first_viewing_plays_the_prepared_track(self, client: TestClient) -> None:
        participant = enrol(client)
        detail = client.get(f"/api/step/view_prepared?participant={participant}").json()
        assert detail["condition"] == "prepared"
        assert all(cue["text"].startswith("Prepared") for cue in detail["captions"])

    def test_the_second_viewing_plays_what_was_just_written(self, client: TestClient) -> None:
        participant = walk_to_regenerated(client)
        detail = client.get(f"/api/step/view_regenerated?participant={participant}").json()
        assert detail["condition"] == "regenerated"
        assert not any(cue["text"].startswith("Prepared") for cue in detail["captions"])

    def test_the_two_viewings_are_different_segments(self, client: TestClient) -> None:
        participant = walk_to_regenerated(client)
        client.post(
            "/api/viewing",
            json={
                "participant": participant,
                "step": "view_regenerated",
                "started_at": "t2",
                "ended_at": "t3",
            },
        )
        rows = client.get(f"/api/log?participant={participant}").json()["viewings"]
        assert [row["condition"] for row in rows] == ["prepared", "regenerated"]
        assert rows[0]["segment"] != rows[1]["segment"]
        assert rows[0]["clip_id"] != rows[1]["clip_id"]

    def test_the_viewing_row_carries_the_full_caption_text(self, client: TestClient) -> None:
        participant = enrol(client)
        client.post(
            "/api/viewing",
            json={"participant": participant, "step": "view_prepared", "started_at": "t0", "ended_at": "t1"},
        )
        row = client.get(f"/api/log?participant={participant}").json()["viewings"][0]
        assert len(row["captions"]) == 4
        assert row["playback_started_at"] == "t0"
        assert row["playback_ended_at"] == "t1"


class TestSurvey:
    def test_an_incomplete_submission_is_refused(self, client: TestClient) -> None:
        participant = enrol(client)
        client.post(
            "/api/viewing",
            json={"participant": participant, "step": "view_prepared", "started_at": "t0", "ended_at": "t1"},
        )
        response = client.post(
            "/api/survey",
            json={
                "participant": participant,
                "page": "art",
                "responses": {},
                "entered_at": "t0",
                "submitted_at": "t1",
            },
        )
        assert response.status_code == 400
        assert response.json()["missing"]

    def test_an_answer_off_the_scale_is_refused(self, client: TestClient) -> None:
        participant = enrol(client)
        client.post(
            "/api/viewing",
            json={"participant": participant, "step": "view_prepared", "started_at": "t0", "ended_at": "t1"},
        )
        items = load_items().item_ids("art")
        response = client.post(
            "/api/survey",
            json={
                "participant": participant,
                "page": "art",
                "responses": {item: 99 for item in items},
                "entered_at": "t0",
                "submitted_at": "t1",
            },
        )
        assert response.status_code == 400
        assert response.json()["invalid"]

    def test_both_pages_ask_the_identical_art_items(self, client: TestClient) -> None:
        participant = walk_to_regenerated(client)
        client.post(
            "/api/viewing",
            json={
                "participant": participant,
                "step": "view_regenerated",
                "started_at": "t2",
                "ended_at": "t3",
            },
        )
        second = client.get(f"/api/step/survey?participant={participant}").json()["blocks"]
        art = next(block for block in second if block["id"] == "art")
        assert art["items"] == [
            item for block in load_items().page_blocks("art") for item in [i.record() for i in block.items]
        ]

    def test_responses_join_to_the_right_viewing(self, client: TestClient) -> None:
        participant = walk_to_regenerated(client)
        rows = client.get(f"/api/log?participant={participant}").json()
        assert rows["responses"][0]["view_id"] == rows["viewings"][0]["view_id"]


class TestVisual:
    def test_fewer_points_than_the_floor_are_refused(self, client: TestClient) -> None:
        participant = enrol(client)
        client.post(
            "/api/viewing",
            json={"participant": participant, "step": "view_prepared", "started_at": "t0", "ended_at": "t1"},
        )
        answer(client, participant, "art")
        response = client.post("/api/visual", json={"participant": participant, "points": POINTS[:1]})
        assert response.status_code == 400
        assert "at least 2 points" in response.json()["error"]

    def test_the_response_never_says_what_a_point_hit(self, client: TestClient) -> None:
        # §4 matches once, on submit; telling the participant would turn the
        # task into hunting for a mask.
        participant = enrol(client)
        client.post(
            "/api/viewing",
            json={"participant": participant, "step": "view_prepared", "started_at": "t0", "ended_at": "t1"},
        )
        answer(client, participant, "art")
        body = client.post("/api/visual", json={"participant": participant, "points": POINTS}).json()
        assert set(body) == {"step", "points", "unclassified", "unclassified_proportion"}

    def test_the_matched_labels_and_the_unclassified_count_reach_the_log(self, client: TestClient) -> None:
        participant = enrol(client)
        client.post(
            "/api/viewing",
            json={"participant": participant, "step": "view_prepared", "started_at": "t0", "ended_at": "t1"},
        )
        answer(client, participant, "art")
        client.post(
            "/api/visual",
            json={"participant": participant, "points": [*POINTS, {"x": 0.95, "y": 0.95}]},
        )
        events = client.get(f"/api/log?participant={participant}").json()["events"]
        submitted = next(event for event in events if event["type"] == "visual.submitted")
        assert submitted["summary"]["unclassified"] == 1
        assert submitted["points"][-1]["label"] == "unclassified"
        assert submitted["points"][-1]["x"] == 0.95


class TestAuditory:
    def test_a_selection_the_segment_does_not_have_is_refused(self, client: TestClient) -> None:
        participant = enrol(client)
        client.post(
            "/api/viewing",
            json={"participant": participant, "step": "view_prepared", "started_at": "t0", "ended_at": "t1"},
        )
        answer(client, participant, "art")
        client.post("/api/visual", json={"participant": participant, "points": POINTS})
        response = client.post(
            "/api/auditory", json={"participant": participant, "selected": ["helicopter"], "lanes": {}}
        )
        assert response.status_code == 400

    def test_selecting_without_playing_is_computed_by_the_server(self, client: TestClient) -> None:
        participant = enrol(client)
        client.post(
            "/api/viewing",
            json={"participant": participant, "step": "view_prepared", "started_at": "t0", "ended_at": "t1"},
        )
        answer(client, participant, "art")
        client.post("/api/visual", json={"participant": participant, "points": POINTS})
        body = client.post(
            "/api/auditory",
            json={
                "participant": participant,
                "selected": ["traffic", "bird"],
                "lanes": {
                    "traffic": {"plays": 2, "listened_ms": 900},
                    "bird": {"plays": 0, "listened_ms": 0},
                },
            },
        ).json()
        assert body["selected_without_playback"] == ["bird"]

    def test_lane_statistics_that_are_not_statistics_are_refused(self, client: TestClient) -> None:
        # Every other input on this surface is shape-checked; this one reads a
        # count out of each lane, and a malformed batch must read like the
        # neighbouring refusals rather than as a server error.
        participant = enrol(client)
        client.post(
            "/api/viewing",
            json={"participant": participant, "step": "view_prepared", "started_at": "t0", "ended_at": "t1"},
        )
        answer(client, participant, "art")
        client.post("/api/visual", json={"participant": participant, "points": POINTS})
        response = client.post(
            "/api/auditory",
            json={"participant": participant, "selected": ["traffic"], "lanes": {"traffic": "played"}},
        )
        assert response.status_code == 400
        assert response.json()["invalid"] == ["traffic"]

    def test_a_lane_without_a_count_reads_as_unplayed(self, client: TestClient) -> None:
        """ "No count" is not evidence a playback happened."""
        participant = enrol(client)
        client.post(
            "/api/viewing",
            json={"participant": participant, "step": "view_prepared", "started_at": "t0", "ended_at": "t1"},
        )
        answer(client, participant, "art")
        client.post("/api/visual", json={"participant": participant, "points": POINTS})
        body = client.post(
            "/api/auditory",
            json={"participant": participant, "selected": ["traffic"], "lanes": {"traffic": {}}},
        ).json()
        assert body["selected_without_playback"] == ["traffic"]

    def test_an_empty_selection_is_allowed(self, client: TestClient) -> None:
        participant = enrol(client)
        client.post(
            "/api/viewing",
            json={"participant": participant, "step": "view_prepared", "started_at": "t0", "ended_at": "t1"},
        )
        answer(client, participant, "art")
        client.post("/api/visual", json={"participant": participant, "points": POINTS})
        response = client.post("/api/auditory", json={"participant": participant, "selected": []})
        assert response.status_code == 200
        assert response.json()["step"] == "regenerating"


class TestRegeneration:
    def test_the_track_is_written_once_and_returned_again_on_a_reload(self, client: TestClient) -> None:
        participant = walk_to_regenerated(client)
        again = client.post("/api/regenerate", json={"participant": participant}).json()
        assert again["cached"] is True
        rows = client.get(f"/api/log?participant={participant}").json()["events"]
        assert sum(1 for row in rows if row["type"] == "regeneration.written") == 1

    def test_the_prompt_and_the_output_reach_the_log(self, client: TestClient) -> None:
        participant = walk_to_regenerated(client)
        events = client.get(f"/api/log?participant={participant}").json()["events"]
        written = next(event for event in events if event["type"] == "regeneration.written")
        assert written["prompt"]
        assert written["raw_output"]
        assert written["auditory_labels"] == ["Traffic"]
        assert written["visual_labels"]
        assert written["fallback"] is False
        assert written["duration_ms"] >= 0

    def test_the_log_says_which_writer_produced_each_slot(self, client: TestClient) -> None:
        # The app hands `regenerate` the CachedWriter, not the writer inside
        # it. If the attribution does not survive that hop, every slot reads
        # as "unknown" and a caption Gemma wrote is indistinguishable from one
        # its template fallback wrote after two budget overruns.
        participant = walk_to_regenerated(client)
        events = client.get(f"/api/log?participant={participant}").json()["events"]
        written = next(event for event in events if event["type"] == "regeneration.written")
        assert written["writers"] == ["template"] * len(written["track"])
        assert written["writer"] == "template"

    def test_the_regenerated_track_has_the_prepared_track_s_timings(
        self, client: TestClient, document: dict[str, Any]
    ) -> None:
        # §6 writes text into fixed slots. If generation could move a cue, a
        # difference between the two survey pages could be a difference in
        # when the captions appeared.
        participant = walk_to_regenerated(client)
        detail = client.get(f"/api/step/view_regenerated?participant={participant}").json()
        prepared = document["segments"][detail["segment"]]["prepared_track"]
        assert [(c["start_ms"], c["end_ms"]) for c in detail["captions"]] == [
            (c["start_ms"], c["end_ms"]) for c in prepared
        ]


class TestSurface:
    def test_the_page_and_its_files_are_served(self, client: TestClient) -> None:
        assert client.get("/").status_code == 200
        for name in ("identity.css", "regen.css", "regen.js"):
            assert client.get(f"/{name}").status_code == 200

    def test_an_unknown_file_is_a_404(self, client: TestClient) -> None:
        assert client.get("/secrets.env").status_code == 404

    def test_the_strings_and_the_scale_are_served_from_one_place(self, client: TestClient) -> None:
        body = client.get("/api/strings").json()
        assert body["scale"]["points"] == 7
        assert len(body["scale"]["anchors"]) == 2
        assert len(body["steps"]) == 6

    def test_media_is_served_by_segment(self, client: TestClient) -> None:
        assert client.get("/media/still/A").status_code == 200
        assert client.get("/media/stem/A/traffic").status_code == 200
        assert client.get("/media/stem/A/helicopter").status_code == 404

    def test_the_masks_are_not_reachable(self, client: TestClient) -> None:
        participant = enrol(client)
        served = client.get(f"/api/step/auditory?participant={participant}")
        assert "mask" not in json.dumps(served.json())
