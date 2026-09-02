"""The HTTP surface: what it serves, what it refuses, and what it never returns."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from dpo.console.app import CAPTION_FAILED, STATIC_FILES, build_app, side_for, subset_key
from dpo.console.copy import STRINGS
from dpo.console.document import configuration_of, field_of
from dpo.console.log import MEASUREMENT_EVENT, SNAPSHOT_SCHEMA
from dpo.console.requests import ConsoleTemplateWriter

PARTICIPANT = "P01"


class FailsOnMore:
    """Writes single-source auditions, then fails.

    ``build_app`` warms every audition before it returns, so a writer that
    fails outright makes the server refuse to start — which is right, and
    which is why the 502 route needs a writer that gets that far and then
    breaks on a real request.
    """

    def write(self, request: Any) -> str:
        from dpo.caption.writer import WriterError

        if len(request.sources) <= 1:
            return "One source alone."
        raise WriterError("no model here")


@pytest.fixture
def client(document: dict[str, Any], tmp_path: Path) -> TestClient:
    app = build_app(document, tmp_path / "media", tmp_path / "out", ConsoleTemplateWriter())
    return TestClient(app)


@pytest.fixture
def shot_ids(document: dict[str, Any]) -> tuple[str, str]:
    clip = document["clips"][0]
    return clip["clip_id"], clip["shots"][0]["shot_id"]


def _sources(client: TestClient, clip_id: str, shot_id: str) -> list[dict[str, Any]]:
    answer = client.get(f"/api/shot/{clip_id}/{shot_id}?participant={PARTICIPANT}")
    assert answer.status_code == 200
    return answer.json()["sources"]


class TestPage:
    def test_the_page_and_its_three_files_are_served(self, client: TestClient) -> None:
        assert client.get("/").status_code == 200
        for name in STATIC_FILES:
            assert client.get("/" + name).status_code == 200

    def test_each_file_is_served_as_itself_with_its_own_media_type(self, client: TestClient) -> None:
        """console.css and console.js share a stem; only the whole name tells them apart."""
        for name, media_type in STATIC_FILES.items():
            answer = client.get("/" + name)
            assert answer.headers["content-type"] == media_type
            assert answer.text == files("dpo.console").joinpath(name).read_text(encoding="utf-8")

    def test_nothing_else_is_served(self, client: TestClient) -> None:
        assert client.get("/secrets.css").status_code == 404


class TestGating:
    def test_every_api_route_needs_a_participant(self, client: TestClient, shot_ids: Any) -> None:
        clip_id, shot_id = shot_ids
        for path in (
            "/api/session",
            f"/api/shot/{clip_id}/{shot_id}",
            f"/api/check/{clip_id}",
            "/api/state",
            "/api/log",
        ):
            assert client.get(path).status_code == 400

    def test_a_participant_id_of_the_wrong_shape_is_refused(self, client: TestClient) -> None:
        assert client.get("/api/session?participant=../etc").status_code == 400


class TestShot:
    def test_no_measured_number_is_ever_returned(self, client: TestClient, shot_ids: Any) -> None:
        clip_id, shot_id = shot_ids
        body = client.get(f"/api/shot/{clip_id}/{shot_id}?participant={PARTICIPANT}").text
        for measured in ("share", "presence", "confidence", "energy", "anchoring", "divergence"):
            assert measured not in body

    def test_bands_and_registers_arrive_as_phrases(self, client: TestClient, shot_ids: Any) -> None:
        for source in _sources(client, *shot_ids):
            assert source["band"] and source["register"]
            assert not any(character.isdigit() for character in source["band"] + source["register"])

    def test_every_admitted_subset_has_its_regimes_precomputed(
        self, client: TestClient, shot_ids: Any, document: dict[str, Any]
    ) -> None:
        clip_id, shot_id = shot_ids
        answer = client.get(f"/api/shot/{clip_id}/{shot_id}?participant={PARTICIPANT}").json()
        ids = [source["id"] for source in answer["sources"]]
        # 2**n - 1 non-empty subsets, so muting never needs a round trip.
        assert len(answer["regimes"]) == 2 ** len(ids) - 1
        assert subset_key(ids) in answer["regimes"]

    def test_an_audition_is_ready_for_every_source(self, client: TestClient, shot_ids: Any) -> None:
        clip_id, shot_id = shot_ids
        answer = client.get(f"/api/shot/{clip_id}/{shot_id}?participant={PARTICIPANT}").json()
        assert set(answer["auditions"]) == {source["id"] for source in answer["sources"]}

    def test_an_unknown_shot_is_a_404(self, client: TestClient, shot_ids: Any) -> None:
        clip_id, _ = shot_ids
        assert client.get(f"/api/shot/{clip_id}/s99?participant={PARTICIPANT}").status_code == 404


class TestCaption:
    def test_identical_settings_return_the_identical_caption_from_the_cache(
        self, client: TestClient, shot_ids: Any
    ) -> None:
        clip_id, shot_id = shot_ids
        ids = [source["id"] for source in _sources(client, clip_id, shot_id)]
        payload = {
            "participant": PARTICIPANT,
            "clip_id": clip_id,
            "shot_id": shot_id,
            "settings": {"grain": "itemized", "admitted": ids, "regime": 0},
        }
        first = client.post("/api/caption", json=payload).json()
        second = client.post("/api/caption", json=payload).json()
        assert first["caption"] == second["caption"]
        assert second["cached"] is True

    def test_a_regime_the_set_cannot_reach_is_a_400_not_a_caption(
        self, client: TestClient, shot_ids: Any
    ) -> None:
        clip_id, shot_id = shot_ids
        ids = [source["id"] for source in _sources(client, clip_id, shot_id)]
        answer = client.post(
            "/api/caption",
            json={
                "participant": PARTICIPANT,
                "clip_id": clip_id,
                "shot_id": shot_id,
                "settings": {"grain": "itemized", "admitted": ids, "regime": 99},
            },
        )
        assert answer.status_code == 400

    def test_a_writer_that_cannot_start_stops_the_server_rather_than_the_participant(
        self, document: dict[str, Any], tmp_path: Path
    ) -> None:
        from dpo.caption.writer import WriterError

        class Dead:
            def write(self, request: Any) -> str:
                raise WriterError("no model here")

        with pytest.raises(WriterError):
            build_app(document, tmp_path / "m", tmp_path / "o", Dead())

    def test_a_failing_writer_becomes_the_one_error_string_the_copy_deck_has(
        self, document: dict[str, Any], tmp_path: Path, shot_ids: Any
    ) -> None:
        client = TestClient(build_app(document, tmp_path / "m", tmp_path / "o", FailsOnMore()))
        clip_id, shot_id = shot_ids
        ids = [source["id"] for source in _sources(client, clip_id, shot_id)]
        answer = client.post(
            "/api/caption",
            json={
                "participant": PARTICIPANT,
                "clip_id": clip_id,
                "shot_id": shot_id,
                "settings": {"grain": "itemized", "admitted": ids, "regime": 0},
            },
        )
        assert answer.status_code == 502
        assert answer.json()["error"] == CAPTION_FAILED == STRINGS["error"]

    def test_the_stamp_is_on_the_caption(self, client: TestClient, shot_ids: Any) -> None:
        clip_id, shot_id = shot_ids
        ids = [source["id"] for source in _sources(client, clip_id, shot_id)]
        answer = client.post(
            "/api/caption",
            json={
                "participant": PARTICIPANT,
                "clip_id": clip_id,
                "shot_id": shot_id,
                "settings": {"grain": "scene", "admitted": ids, "regime": 0},
            },
        ).json()
        assert len(answer["config_hash"]) == 12


class TestMeasurements:
    def test_opening_a_shot_writes_section_nine_to_the_log(self, client: TestClient, shot_ids: Any) -> None:
        clip_id, shot_id = shot_ids
        _sources(client, clip_id, shot_id)
        events = client.get(f"/api/log?participant={PARTICIPANT}").json()["events"]
        measured = [event for event in events if event["type"] == MEASUREMENT_EVENT]
        assert len(measured) == 1
        assert set(measured[0]["measurements"]) == {"anchoring", "divergence", "residuals"}

    def test_reopening_a_shot_does_not_restate_a_constant(self, client: TestClient, shot_ids: Any) -> None:
        clip_id, shot_id = shot_ids
        _sources(client, clip_id, shot_id)
        _sources(client, clip_id, shot_id)
        events = client.get(f"/api/log?participant={PARTICIPANT}").json()["events"]
        assert len([event for event in events if event["type"] == MEASUREMENT_EVENT]) == 1

    def test_the_measurements_match_the_field_they_were_taken_from(
        self, client: TestClient, shot_ids: Any, document: dict[str, Any]
    ) -> None:
        clip_id, shot_id = shot_ids
        _sources(client, clip_id, shot_id)
        events = client.get(f"/api/log?participant={PARTICIPANT}").json()["events"]
        logged = next(event for event in events if event["type"] == MEASUREMENT_EVENT)["measurements"]
        field = field_of(document["clips"][0]["shots"][0], configuration_of(document))
        assert logged["anchoring"] == pytest.approx(field.anchoring())
        assert logged["divergence"] == pytest.approx(field.divergence())


class TestCheck:
    def _commit_every_shot(self, client: TestClient, document: dict[str, Any]) -> None:
        clip = document["clips"][0]
        committed: dict[str, Any] = {clip["clip_id"]: {}}
        for shot in clip["shots"]:
            ids = [source["id"] for source in _sources(client, clip["clip_id"], shot["shot_id"])]
            answer = client.post(
                "/api/caption",
                json={
                    "participant": PARTICIPANT,
                    "clip_id": clip["clip_id"],
                    "shot_id": shot["shot_id"],
                    "settings": {"grain": "itemized", "admitted": ids, "regime": 0},
                },
            ).json()
            committed[clip["clip_id"]][shot["shot_id"]] = {"caption": answer["caption"], "key": answer["key"]}
        client.post(
            "/api/events",
            json={
                "participant": PARTICIPANT,
                "events": [],
                "snapshot": {"schema": SNAPSHOT_SCHEMA, "committed": committed},
            },
        )

    def test_the_check_follows_the_endpoint(
        self, client: TestClient, document: dict[str, Any], shot_ids: Any
    ) -> None:
        clip_id, _ = shot_ids
        answer = client.get(f"/api/check/{clip_id}?participant={PARTICIPANT}")
        assert answer.status_code == 409
        # The body names where the session stopped.
        assert answer.json()["shot_id"] == document["clips"][0]["shots"][0]["shot_id"]

    def test_each_shot_offers_the_committed_caption_against_the_default_policy(
        self, client: TestClient, document: dict[str, Any]
    ) -> None:
        self._commit_every_shot(client, document)
        clip = document["clips"][0]
        answer = client.get(f"/api/check/{clip['clip_id']}?participant={PARTICIPANT}").json()
        for pair, shot in zip(answer["shots"], clip["shots"], strict=True):
            assert shot["default_caption"] in (pair["a"], pair["b"])
            assert pair["a"] != pair["b"]

    def test_the_cards_are_never_labelled_own_or_default(
        self, client: TestClient, document: dict[str, Any]
    ) -> None:
        self._commit_every_shot(client, document)
        clip_id = document["clips"][0]["clip_id"]
        body = client.get(f"/api/check/{clip_id}?participant={PARTICIPANT}").text
        assert "own" not in body and "default_caption" not in body


class TestSideAssignment:
    def test_a_side_is_stable_for_one_participant_across_reloads(self) -> None:
        assert side_for("P01", "c", "s1", "abc") == side_for("P01", "c", "s1", "abc")

    def test_two_participants_do_not_share_one_layout(self) -> None:
        sides = {side_for(f"P{index:02d}", "c", "s1", "abc") for index in range(20)}
        assert sides == {"own", "default"}

    def test_a_recalibration_reassigns_the_sides(self) -> None:
        assignments = [side_for("P01", "c", f"s{index}", "abc") for index in range(30)]
        other = [side_for("P01", "c", f"s{index}", "xyz") for index in range(30)]
        assert assignments != other


class TestLog:
    def test_every_line_carries_the_stamp_from_the_server_not_the_browser(self, client: TestClient) -> None:
        client.post(
            "/api/events",
            json={
                "participant": PARTICIPANT,
                "events": [{"type": "screen.enter", "screen": "author", "config_hash": "forged"}],
                "snapshot": {"schema": SNAPSHOT_SCHEMA},
            },
        )
        events = client.get(f"/api/log?participant={PARTICIPANT}").json()["events"]
        assert events[0]["config_hash"] != "forged"

    def test_an_event_without_a_type_is_refused(self, client: TestClient) -> None:
        answer = client.post(
            "/api/events", json={"participant": PARTICIPANT, "events": [{"screen": "author"}]}
        )
        assert answer.status_code == 400

    def test_the_snapshot_comes_back_for_a_resume(self, client: TestClient) -> None:
        snapshot = {"schema": SNAPSHOT_SCHEMA, "screen": "author", "shot_index": 1}
        client.post("/api/events", json={"participant": PARTICIPANT, "events": [], "snapshot": snapshot})
        assert client.get(f"/api/state?participant={PARTICIPANT}").json()["snapshot"] == snapshot

    def test_the_download_is_the_events_and_the_snapshot_in_one_document(self, client: TestClient) -> None:
        client.post("/api/events", json={"participant": PARTICIPANT, "events": [], "snapshot": None})
        answer = client.get(f"/api/log?participant={PARTICIPANT}").json()
        assert answer["schema"] == "dpo.caption-console-log/v1"
        assert set(answer) >= {"events", "snapshot", "config_hash", "session_id", "participant"}


class TestDownload:
    def test_the_log_is_served_as_an_attachment_so_the_done_screen_stays(
        self, client: TestClient, shot_ids: Any
    ) -> None:
        _sources(client, *shot_ids)
        response = client.get(f"/api/log?participant={PARTICIPANT}")
        assert response.status_code == 200
        assert response.headers["content-disposition"].startswith("attachment;")
        assert PARTICIPANT in response.headers["content-disposition"]
        assert response.json()["events"]


class TestSession:
    def test_the_page_gets_its_copy_from_the_one_place_it_lives(self, client: TestClient) -> None:
        answer = client.get(f"/api/session?participant={PARTICIPANT}").json()
        assert answer["strings"] == STRINGS
        assert answer["strings"]["criterion"] == "Is this the caption you'd want for this moment?"

    def test_the_phase_rail_is_the_table_three_string(self, client: TestClient) -> None:
        answer = client.get(f"/api/session?participant={PARTICIPANT}").json()
        assert " · ".join(answer["strings"]["phase_rail"]) == "1 watch · 2 author · 3 watch again · 4 check"


class TestScope:
    def test_no_reroll_route_exists(self, client: TestClient) -> None:
        paths = {route.path for route in client.app.routes}  # type: ignore[attr-defined]
        assert not any("reroll" in path or "regenerate" in path for path in paths)

    def test_the_cache_file_is_written_under_the_out_directory(
        self, document: dict[str, Any], tmp_path: Path, shot_ids: Any
    ) -> None:
        out = tmp_path / "out"
        client = TestClient(build_app(document, tmp_path / "media", out, ConsoleTemplateWriter()))
        clip_id, shot_id = shot_ids
        _sources(client, clip_id, shot_id)
        entries = json.loads((out / "captions.json").read_text(encoding="utf-8"))
        assert entries  # the auditions were written before the first request
