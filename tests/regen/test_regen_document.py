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
    participant_document,
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

    def test_a_stem_without_a_colour_in_hex_is_refused(self, document: dict[str, Any]) -> None:
        document["segments"]["A"]["stems"][0]["colour"] = "blue"
        with pytest.raises(RegenDocumentError, match="#rrggbb"):
            validate_regen_document(document)

    def test_a_repeated_object_id_is_refused(self, document: dict[str, Any]) -> None:
        document["segments"]["A"]["objects"][1]["id"] = document["segments"]["A"]["objects"][0]["id"]
        with pytest.raises(RegenDocumentError, match="declared twice"):
            validate_regen_document(document)


class TestWhatTheBrowserGets:
    def test_the_masks_never_leave_the_server(self, document: dict[str, Any]) -> None:
        # §4 matches once, on submit. A browser holding the masks could match
        # on every click, which is the thing that rule prevents.
        served = participant_document(document, "A")
        assert all(set(entry) == {"id"} for entry in served["objects"])
        assert "mask" not in json.dumps(served)

    def test_the_fallback_track_never_leaves_the_server(self, document: dict[str, Any]) -> None:
        assert "Fallback" not in json.dumps(participant_document(document, "A"))

    def test_the_lanes_carry_what_they_need_to_be_drawn_and_played(self, document: dict[str, Any]) -> None:
        stem = participant_document(document, "A")["stems"][0]
        assert set(stem) == {"id", "label", "colour", "gain", "waveform"}


class TestResolution:
    def test_objects_resolve_under_the_media_directory(
        self, document: dict[str, Any], media_dir: Path
    ) -> None:
        objects = objects_of(document, media_dir, "A")
        assert [entry.label for entry in objects] == ["Building", "Person"]
        assert all(entry.path.is_file() for entry in objects)
