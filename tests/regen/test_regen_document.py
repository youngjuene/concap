"""The session document: both segments prepared, both tracks under one schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dpo.regen.document import (
    RegenDocumentError,
    load_regen_document,
    objects_of,
    slug,
    track_of,
    validate_regen_document,
)


class TestValid:
    def test_the_fixture_validates(self, document: dict[str, Any]) -> None:
        validate_regen_document(document)

    def test_both_segments_carry_both_tracks(self, document: dict[str, Any]) -> None:
        # Either segment can be either condition (§1), so a document that only
        # prepared A would fail for odd-numbered participants — in front of them.
        for name in ("A", "B"):
            assert len(track_of(document, name, "prepared_track")) == 4
            assert len(track_of(document, name, "fallback_track")) == 4

    def test_it_round_trips_through_disk(self, document: dict[str, Any], tmp_path: Path) -> None:
        path = tmp_path / "session.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        assert load_regen_document(path)["session_id"] == document["session_id"]


class TestRefusals:
    def test_a_missing_segment_is_refused(self, document: dict[str, Any]) -> None:
        del document["segments"]["B"]
        with pytest.raises(RegenDocumentError, match="segments"):
            validate_regen_document(document)

    def test_the_same_clip_twice_is_refused(self, document: dict[str, Any]) -> None:
        document["segments"]["B"]["clip_id"] = document["segments"]["A"]["clip_id"]
        with pytest.raises(RegenDocumentError, match="the same clip as segment A"):
            validate_regen_document(document)

    def test_tracks_of_different_lengths_are_refused(self, document: dict[str, Any]) -> None:
        document["segments"]["A"]["fallback_track"].pop()
        with pytest.raises(RegenDocumentError, match="cue slots"):
            validate_regen_document(document)

    def test_an_unauthored_caption_is_refused(self, document: dict[str, Any]) -> None:
        # What keeps a scaffold from reaching a participant.
        document["segments"]["A"]["prepared_track"][0]["text"] = ""
        with pytest.raises(RegenDocumentError, match=r"prepared_track\[0\].text"):
            validate_regen_document(document)

    def test_a_track_running_past_the_clip_is_refused(self, document: dict[str, Any]) -> None:
        document["segments"]["A"]["prepared_track"][-1]["end_ms"] = 99999
        with pytest.raises(RegenDocumentError, match="past the segment"):
            validate_regen_document(document)

    def test_a_media_reference_escaping_the_media_directory_is_refused(
        self, document: dict[str, Any]
    ) -> None:
        document["segments"]["A"]["video"] = "../../etc/passwd"
        with pytest.raises(RegenDocumentError, match="inside the media directory"):
            validate_regen_document(document)

    def test_a_segment_without_its_sound_is_refused(self, document: dict[str, Any]) -> None:
        """§6 reads the clip's audio from a wav beside it, not out of the container.

        A document that declares no audio is one whose every regeneration would
        fall back, so it is refused on load rather than four slots into a
        participant's session.
        """
        del document["segments"]["A"]["audio"]
        with pytest.raises(RegenDocumentError, match="segments.A.audio"):
            validate_regen_document(document)

    def test_the_sound_reference_may_not_escape_the_media_directory_either(
        self, document: dict[str, Any]
    ) -> None:
        document["segments"]["A"]["audio"] = "/etc/passwd"
        with pytest.raises(RegenDocumentError, match="inside the media directory"):
            validate_regen_document(document)

    def test_a_stem_without_a_colour_in_hex_is_refused(self, document: dict[str, Any]) -> None:
        document["segments"]["A"]["stems"][0]["colour"] = "blue"
        with pytest.raises(RegenDocumentError, match="#rrggbb"):
            validate_regen_document(document)

    def test_a_repeated_object_id_is_refused(self, document: dict[str, Any]) -> None:
        objects = document["segments"]["A"]["frames"][0]["objects"]
        objects[1]["id"] = objects[0]["id"]
        with pytest.raises(RegenDocumentError, match="declared twice"):
            validate_regen_document(document)

    def test_a_frame_past_the_end_of_the_clip_is_refused(self, document: dict[str, Any]) -> None:
        document["segments"]["A"]["frames"][-1]["at_ms"] = 99000
        with pytest.raises(RegenDocumentError, match="past the segment"):
            validate_regen_document(document)

    def test_a_strip_out_of_clock_order_is_refused(self, document: dict[str, Any]) -> None:
        frames = document["segments"]["A"]["frames"]
        frames[1]["at_ms"], frames[2]["at_ms"] = frames[2]["at_ms"], frames[1]["at_ms"]
        with pytest.raises(RegenDocumentError, match="not after the previous"):
            validate_regen_document(document)

    def test_a_frame_without_masks_is_refused(self, document: dict[str, Any]) -> None:
        document["segments"]["A"]["frames"][0]["objects"] = []
        with pytest.raises(RegenDocumentError, match="objects"):
            validate_regen_document(document)

    def test_a_strip_of_one_frame_is_refused(self, document: dict[str, Any]) -> None:
        document["segments"]["A"]["frames"] = document["segments"]["A"]["frames"][:1]
        with pytest.raises(RegenDocumentError, match="at least"):
            validate_regen_document(document)


class TestTheFamilyOnAStem:
    """§5 records which families a clip carries, so the parent has to survive.

    What the browser is *given* is asserted at the routes now
    (``tests/regen/test_regen_app.py``): the helper these tests used to call,
    ``participant_document``, stopped being what the app serves when §5 became
    five fixed families, and went with the screen it fed.
    """

    def test_a_family_that_is_not_words_is_refused(self, document: dict[str, Any]) -> None:
        document["segments"]["A"]["stems"][0]["parent"] = 7
        with pytest.raises(RegenDocumentError, match="parent"):
            validate_regen_document(document)


class TestSlug:
    """Source vocabularies are written for people; ids are not."""

    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("Traffic noise, roadway noise", "traffic_noise_roadway_noise"),
            ("Police car (siren)", "police_car_siren"),
            ("Vehicle horn, car horn, honking", "vehicle_horn_car_horn_honking"),
            ("Road transport", "road_transport"),
            ("Beep, bleep", "beep_bleep"),
            ("  Bird  ", "bird"),
        ],
    )
    def test_a_label_becomes_an_id_the_document_accepts(self, label: str, expected: str) -> None:
        assert slug(label) == expected

    def test_a_label_with_no_usable_character_is_refused(self) -> None:
        with pytest.raises(RegenDocumentError, match="identifier"):
            slug("(,)")

    def test_two_labels_that_slug_alike_are_caught_as_a_repeated_id(self, document: dict[str, Any]) -> None:
        # Not slug's job — a collision is a property of the set a segment
        # declares, and the document already refuses one id declared twice.
        stems = document["segments"]["A"]["stems"]
        stems[0]["id"] = slug("Car, horn")
        stems[1]["id"] = slug("Car (horn)")
        with pytest.raises(RegenDocumentError, match="declared twice"):
            validate_regen_document(document)


class TestResolution:
    def test_objects_resolve_under_the_media_directory(
        self, document: dict[str, Any], media_dir: Path
    ) -> None:
        objects = objects_of(document, media_dir, "A")
        assert sorted(objects) == [0, 1, 2, 3, 4]
        for entries in objects.values():
            assert [entry.label for entry in entries] == ["Building", "Person"]
            assert all(entry.path.is_file() for entry in entries)

    def test_each_frame_resolves_to_its_own_masks(self, document: dict[str, Any], media_dir: Path) -> None:
        objects = objects_of(document, media_dir, "A")
        assert len({entries[0].path for entries in objects.values()}) == len(objects)
