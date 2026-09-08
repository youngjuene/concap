"""The HTTP surface: the six steps in order, and what each one refuses."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from dpo.regen.app import build_app
from dpo.regen.config import SOUND_FAMILIES
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


def hear(client: TestClient, participant: str, heard: tuple[str, ...] = ("things",)) -> Any:
    """Answer §5: every family, with `heard` the ones reported as heard."""
    return client.post(
        "/api/auditory",
        json={
            "participant": participant,
            "heard": {family: family in heard for family in SOUND_FAMILIES},
        },
    )


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
    hear(client, participant)
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


class TestLanguage:
    """§9.3: a study may offer two, and a session reads one of them."""

    def test_a_session_opens_in_the_studys_first_language(self, client: TestClient) -> None:
        body = client.post("/api/session", json={}).json()
        assert body["language"] == "en"
        assert body["languages"] == ["en", "ko"]
        assert body["language_locked"] is False

    def test_the_other_language_can_be_chosen_before_the_first_clip(self, client: TestClient) -> None:
        participant = enrol(client)
        assert client.post("/api/language", json={"participant": participant, "language": "ko"}).json() == {
            "language": "ko",
            "language_locked": False,
        }
        detail = client.get(f"/api/step/view_prepared?participant={participant}").json()
        assert all(cue["language"] == "ko" for cue in detail["captions"])
        assert all(cue["text"].endswith("소리") for cue in detail["captions"])

    def test_a_language_the_study_does_not_offer_is_refused(self, client: TestClient) -> None:
        participant = enrol(client)
        response = client.post("/api/language", json={"participant": participant, "language": "fr"})
        assert response.status_code == 400
        assert response.json()["asked"] == "fr"

    def test_it_is_fixed_once_a_clip_has_played(self, client: TestClient) -> None:
        # §8 is compared against §3. A session read half in one language and
        # half in the other has moved something the study measures.
        participant = enrol(client)
        client.post(
            "/api/viewing",
            json={"participant": participant, "step": "view_prepared", "started_at": "t0", "ended_at": "t1"},
        )
        response = client.post("/api/language", json={"participant": participant, "language": "ko"})
        assert response.status_code == 409
        assert response.json()["language"] == "en"
        assert client.post("/api/session", json={"participant": participant}).json()["language_locked"]

    def test_the_viewing_row_says_which_language_was_read(self, client: TestClient) -> None:
        participant = enrol(client)
        client.post("/api/language", json={"participant": participant, "language": "ko"})
        client.post(
            "/api/viewing",
            json={"participant": participant, "step": "view_prepared", "started_at": "t0", "ended_at": "t1"},
        )
        row = client.get(f"/api/log?participant={participant}").json()["viewings"][0]
        assert row["language"] == "ko"
        assert all(cue["language"] == "ko" for cue in row["captions"])

    def test_the_regenerated_track_is_written_in_the_language_on_screen(self, client: TestClient) -> None:
        participant = enrol(client)
        client.post("/api/language", json={"participant": participant, "language": "ko"})
        client.post(
            "/api/viewing",
            json={"participant": participant, "step": "view_prepared", "started_at": "t0", "ended_at": "t1"},
        )
        answer(client, participant, "art")
        client.post("/api/visual", json={"participant": participant, "points": POINTS})
        hear(client, participant)
        body = client.post("/api/regenerate", json={"participant": participant}).json()
        assert all(cue["language"] == "ko" for cue in body["track"])
        events = client.get(f"/api/log?participant={participant}").json()["events"]
        written = next(event for event in events if event["type"] == "regeneration.written")
        assert written["language"] == "ko"
        # The template writer works in English, so a Korean session falls back
        # rather than showing prose in the wrong language — which is the check
        # doing its job, not a defect.
        assert written["fallback"] is True
        assert "not in the language on screen" in written["fallback_reason"]


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
    """§5 asks the same five families of everyone, and requires all five.

    The lane tests this replaces asserted a selection among the sources a clip
    carried, per-lane playback counts, and a selected-without-playing flag.
    None of those exist now: the screen is a fixed judgment, so what has to be
    guarded is that it stays fixed, that a blank is not read as a denial, and
    that a family the clip does not contain is answerable.
    """

    def _at_auditory(self, client: TestClient) -> str:
        participant = enrol(client)
        client.post(
            "/api/viewing",
            json={"participant": participant, "step": "view_prepared", "started_at": "t0", "ended_at": "t1"},
        )
        answer(client, participant, "art")
        client.post("/api/visual", json={"participant": participant, "points": POINTS})
        return participant

    def test_the_same_five_families_are_asked_whatever_the_clip_holds(self, client: TestClient) -> None:
        participant = self._at_auditory(client)
        detail = client.get(f"/api/step/auditory?participant={participant}").json()
        assert detail["families"] == list(SOUND_FAMILIES)
        assert "stems" not in detail, "the sources the clip carries are not the question any more"

    def test_a_family_the_clip_does_not_contain_can_still_be_reported(self, client: TestClient) -> None:
        # The point of the redesign. The fixture's stems are traffic and bird;
        # nothing in it is music, and claiming music has to be possible or a
        # false alarm cannot be measured.
        participant = self._at_auditory(client)
        response = hear(client, participant, heard=("music",))
        assert response.status_code == 200
        assert response.json()["heard"] == ["music"]

    def test_every_family_has_to_be_answered(self, client: TestClient) -> None:
        # A blank is a participant who did not answer, which is not the same
        # as one who did not hear; §6 must not be conditioned on the
        # difference between a denial and a shrug.
        participant = self._at_auditory(client)
        response = client.post("/api/auditory", json={"participant": participant, "heard": {"human": True}})
        assert response.status_code == 400
        assert set(response.json()["missing"]) == set(SOUND_FAMILIES) - {"human"}

    def test_hearing_none_of_them_is_a_complete_answer(self, client: TestClient) -> None:
        participant = self._at_auditory(client)
        response = hear(client, participant, heard=())
        assert response.status_code == 200
        assert response.json()["heard"] == []
        assert response.json()["step"] == "regenerating"

    def test_a_family_the_study_does_not_ask_about_is_refused(self, client: TestClient) -> None:
        participant = self._at_auditory(client)
        payload = {family: True for family in SOUND_FAMILIES}
        payload["helicopter"] = True
        response = client.post("/api/auditory", json={"participant": participant, "heard": payload})
        assert response.status_code == 400
        assert response.json()["unknown"] == ["helicopter"]

    def test_an_answer_that_is_not_yes_or_no_is_refused(self, client: TestClient) -> None:
        # Every other input on this surface is shape-checked, and "maybe" has
        # to read like the neighbouring refusals rather than as a server error.
        participant = self._at_auditory(client)
        payload: dict[str, Any] = {family: True for family in SOUND_FAMILIES}
        payload["music"] = "maybe"
        response = client.post("/api/auditory", json={"participant": participant, "heard": payload})
        assert response.status_code == 400
        assert response.json()["invalid"] == ["music"]

    def test_the_log_records_both_answers_and_what_the_clip_carried(self, client: TestClient) -> None:
        """A false alarm has to be readable without joining to the document."""
        participant = self._at_auditory(client)
        hear(client, participant, heard=("music", "things"))
        events = client.get(f"/api/log?participant={participant}").json()["events"]
        submitted = [row for row in events if row["type"] == "auditory.submitted"][-1]
        assert submitted["heard"] == ["things", "music"], "reported in the study's order, not the page's"
        assert set(submitted["not_heard"]) == set(SOUND_FAMILIES) - {"music", "things"}
        assert submitted["present"], "what the clip actually carries is recorded beside the answer"

    def test_what_the_clip_carried_is_written_in_the_answers_vocabulary(self, client: TestClient) -> None:
        """The three fields have to be comparable, which is the point of the row.

        The document stores AudioSet's own class names in ``stem.parent``; §5
        answers in family keys. Written straight through, the row read
        ``heard: ["things"]`` beside ``present: ["Sounds of things"]`` — two
        lists with nothing in common, in a row whose purpose is comparing them.
        """
        participant = self._at_auditory(client)
        hear(client, participant, heard=("music",))
        events = client.get(f"/api/log?participant={participant}").json()["events"]
        submitted = [row for row in events if row["type"] == "auditory.submitted"][-1]
        assert set(submitted["present"]) <= set(SOUND_FAMILIES), submitted["present"]
        assert "things" in submitted["present"], "the fixture's traffic stem is a sound of things"
        # And the comparison the row exists for: claimed but not there.
        false_alarm = set(submitted["heard"]) - set(submitted["present"])
        assert false_alarm == {"music"}

    def test_a_family_the_study_does_not_ask_about_is_kept_not_dropped(
        self, document: dict[str, Any], media_dir: Path, tmp_path: Path
    ) -> None:
        # AudioSet has top-level classes §5 has no row for. One in a clip is
        # not a false alarm the participant could have made, and dropping it
        # silently would leave the log looking as though the clip held nothing.
        document["segments"]["A"]["stems"][0]["parent"] = "Source-ambiguous sounds"
        client = TestClient(build_app(document, media_dir, tmp_path / "out", RegenTemplateWriter()))
        participant = self._at_auditory(client)
        hear(client, participant, heard=())
        events = client.get(f"/api/log?participant={participant}").json()["events"]
        submitted = [row for row in events if row["type"] == "auditory.submitted"][-1]
        assert submitted["present_unasked"] == ["Source-ambiguous sounds"]
        assert "Source-ambiguous sounds" not in submitted["present"]

    def test_what_reaches_the_regeneration_does_not_depend_on_the_language_read(
        self, client: TestClient
    ) -> None:
        """Two participants reporting the same families hand §6 the same input.

        The display label is chrome and changes with the interface language;
        the prose form the writer is given must not, or the same report would
        produce a different caption in each language.
        """

        def walked(language: str | None) -> list[str]:
            participant = enrol(client)
            if language:
                client.post("/api/language", json={"participant": participant, "language": language})
            client.post(
                "/api/viewing",
                json={
                    "participant": participant,
                    "step": "view_prepared",
                    "started_at": "t0",
                    "ended_at": "t1",
                },
            )
            answer(client, participant, "art")
            client.post("/api/visual", json={"participant": participant, "points": POINTS})
            hear(client, participant, heard=("human",))
            client.post("/api/regenerate", json={"participant": participant})
            events = client.get(f"/api/log?participant={participant}").json()["events"]
            written = next(row for row in events if row["type"] == "regeneration.written")
            return list(written["auditory_labels"])

        assert walked(None) == walked("ko") == [SOUND_FAMILIES["human"]]


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
        assert written["auditory_labels"] == [SOUND_FAMILIES["things"]]
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

    def test_the_chrome_is_served_in_every_language_the_study_offers(self, client: TestClient) -> None:
        """All of them at once, so a switch redraws rather than re-fetches.

        §9.3 lets the participant change language until the first clip plays.
        Sending only the language they are currently reading would put a round
        trip between the press and the redraw, and would stop boot asking for
        the copy and the enrolment together — the copy would have to wait to
        learn which language the enrolment assigned.
        """
        body = client.get("/api/strings").json()
        assert set(body["strings"]) == {"en", "ko"}, "the fixture's study offers both"
        assert body["strings"]["en"]["app_title"] == "Sound captions"
        assert body["strings"]["ko"]["app_title"] != body["strings"]["en"]["app_title"]
        assert body["strings"]["ko"]["steps"][2] != body["strings"]["en"]["steps"][2]

    def test_a_study_in_one_language_is_served_one_tree(
        self, document: dict[str, Any], media_dir: Path, tmp_path: Path
    ) -> None:
        english = dict(document)
        english["config"] = {
            **document["config"],
            "calibration": {**document["config"]["calibration"], "languages": ["en"]},
        }
        client = TestClient(build_app(english, media_dir, tmp_path / "out", RegenTemplateWriter()))
        assert set(client.get("/api/strings").json()["strings"]) == {"en"}

    def test_media_is_served_by_segment(self, client: TestClient) -> None:
        assert client.get("/media/frame/A/0").status_code == 200
        assert client.get("/media/frame/A/4").status_code == 200
        assert client.get("/media/frame/A/5").status_code == 404, "the strip has five frames"
        assert client.get("/media/stem/A/traffic").status_code == 200
        assert client.get("/media/stem/A/helicopter").status_code == 404

    def test_the_strip_is_served_as_timings_only(self, client: TestClient) -> None:
        # §4 matches once, on submit. The page is told when each frame is from
        # and nothing about what is in it.
        participant = enrol(client)
        client.post(
            "/api/viewing",
            json={"participant": participant, "step": "view_prepared", "started_at": "t0", "ended_at": "t1"},
        )
        answer(client, participant, "art")
        detail = client.get(f"/api/step/visual?participant={participant}").json()
        assert [frame["index"] for frame in detail["frames"]] == [0, 1, 2, 3, 4]
        assert all(set(frame) == {"index", "at_ms"} for frame in detail["frames"])
        assert detail["minimum"] == 2, "the fixture's own calibration, echoed"

    def test_the_session_log_downloads_rather_than_replacing_the_page(self, client: TestClient) -> None:
        # The done screen navigates here. A bare JSON body replaces the kiosk
        # with the log's text, which is a state a fullscreen kiosk cannot be
        # brought back from — the same defect the console fixed.
        participant = enrol(client)
        response = client.get(f"/api/log?participant={participant}")
        assert response.status_code == 200
        assert response.headers["content-disposition"].startswith("attachment;")
        assert participant in response.headers["content-disposition"]

    def test_events_are_refused_for_a_participant_nobody_enrolled(self, client: TestClient) -> None:
        # Every other write goes through the roster first. Without this one, a
        # well-formed identifier opened its own events file, and a file in
        # --out with no roster entry beside it is a record of a session that
        # never happened.
        response = client.post(
            "/api/events", json={"participant": "ghost99", "events": [{"type": "injected"}]}
        )
        assert response.status_code == 404
        assert "enrolled" in response.json()["error"]

    def test_events_are_taken_for_a_participant_who_was_enrolled(self, client: TestClient) -> None:
        participant = enrol(client)
        response = client.post(
            "/api/events", json={"participant": participant, "events": [{"type": "step.entered"}]}
        )
        assert response.json() == {"written": 1}

    def test_the_masks_are_not_reachable(self, client: TestClient) -> None:
        participant = enrol(client)
        served = client.get(f"/api/step/auditory?participant={participant}")
        assert "mask" not in json.dumps(served.json())


class TestRegenerationProgress:
    """§6 reports cues written, for the screen waiting on it and the console."""

    def test_it_reads_as_idle_with_the_studys_slot_count_before_anything_runs(
        self, client: TestClient
    ) -> None:
        participant = enrol(client)
        body = client.get("/api/regenerate/progress", params={"participant": participant}).json()
        # The count is there even when nothing is running, so the waiting screen
        # can draw a bar of the right width rather than growing one.
        prepared = client.get(f"/api/step/view_prepared?participant={participant}").json()["segment"]
        assert body == {
            "writing": False,
            "done": 0,
            "total": 4,
            # The clip §6's wait is used to fetch: the other segment.
            "next_segment": "B" if prepared == "A" else "A",
        }

    def test_it_reads_as_idle_again_once_the_track_is_written(self, client: TestClient) -> None:
        participant = walk_to_regenerated(client)
        body = client.get("/api/regenerate/progress", params={"participant": participant}).json()
        assert body["writing"] is False

    def test_a_malformed_participant_is_refused_like_its_neighbours(self, client: TestClient) -> None:
        assert client.get("/api/regenerate/progress").status_code == 400

    def test_it_names_the_clip_the_next_viewing_will_play(self, client: TestClient) -> None:
        """§6's wait is what pays for the second clip's five to seven megabytes.

        The page cannot ask ``/api/step/view_regenerated`` yet — that step is
        gated and the answer would be a 409 — so the route the waiting screen
        is already polling carries it. It is the same fact that step returns a
        moment later, never a different one.
        """
        participant = walk_to_regenerated(client)
        waiting = client.get("/api/regenerate/progress", params={"participant": participant}).json()
        viewing = client.get(f"/api/step/view_regenerated?participant={participant}").json()
        assert waiting["next_segment"] == viewing["segment"]

    def test_it_is_the_segment_the_first_viewing_did_not_use(self, client: TestClient) -> None:
        participant = enrol(client)
        prepared = client.get(f"/api/step/view_prepared?participant={participant}").json()["segment"]
        body = client.get("/api/regenerate/progress", params={"participant": participant}).json()
        assert body["next_segment"] != prepared

    def test_an_unenrolled_participant_still_reads_as_idle(self, client: TestClient) -> None:
        # Best-effort, as the route has always been: no assignment yet is a
        # missing field, not a 404. A progress display may not be the thing
        # that ends a session.
        body = client.get("/api/regenerate/progress", params={"participant": "pfffffff"})
        assert body.status_code == 200
        assert "next_segment" not in body.json()
        assert body.json()["writing"] is False

    def test_it_reports_the_cues_written_while_the_model_is_still_in_the_call(
        self, document: dict[str, Any], media_dir: Path, tmp_path: Path
    ) -> None:
        """The point of the route: answered while /api/regenerate is in flight.

        `/api/regenerate` is one blocking call and a sync route, so it runs in
        the threadpool and this GET is served beside it. A writer that stops
        inside its second slot puts the run in a known place — one cue written
        — and the route has to say so while the second is still decoding.
        """
        inside_second = threading.Event()
        release = threading.Event()

        class Blocks:
            identity = "blocks"

            def __init__(self) -> None:
                self.calls = 0

            def write(self, request: Any) -> str:
                self.calls += 1
                if self.calls == 2:
                    inside_second.set()
                    release.wait(10)
                return "A car passes."

        app = build_app(document, media_dir, tmp_path / "out", Blocks(), items=load_items())
        walker, watcher = TestClient(app), TestClient(app)
        participant = enrol(walker)
        walker.post(
            "/api/viewing",
            json={
                "participant": participant,
                "step": "view_prepared",
                "started_at": "t0",
                "ended_at": "t1",
            },
        )
        answer(walker, participant, "art")
        walker.post("/api/visual", json={"participant": participant, "points": POINTS})
        hear(walker, participant)

        writing = threading.Thread(
            target=lambda: walker.post("/api/regenerate", json={"participant": participant})
        )
        writing.start()
        try:
            assert inside_second.wait(10), "the writer never reached its second slot"
            body = watcher.get("/api/regenerate/progress", params={"participant": participant}).json()
        finally:
            release.set()
            writing.join(20)
        assert {key: body[key] for key in ("writing", "done", "total")} == {
            "writing": True,
            "done": 1,
            "total": 4,
        }
        # And nothing is left in flight for a screen to keep reading.
        after = watcher.get("/api/regenerate/progress", params={"participant": participant}).json()
        assert after["writing"] is False
