"""The document schema: what it refuses, and what it lets the browser hold."""

from __future__ import annotations

from typing import Any

import pytest

from dpo.caption.writer import CAPTION_MAX_CHARS
from dpo.console.document import (
    ConsoleDocumentError,
    configuration_of,
    field_of,
    load_console_document,
    participant_document,
    validate_console_document,
)


def _first_shot(document: dict[str, Any]) -> dict[str, Any]:
    return document["clips"][0]["shots"][0]


class TestValidation:
    def test_the_fixture_is_valid(self, document: dict[str, Any]) -> None:
        validate_console_document(document)

    def test_a_wrong_schema_is_named(self, document: dict[str, Any]) -> None:
        document["schema"] = "dpo.caption-session/v1"
        with pytest.raises(ConsoleDocumentError, match="^schema"):
            validate_console_document(document)

    def test_the_offending_path_is_named_all_the_way_down(self, document: dict[str, Any]) -> None:
        _first_shot(document)["sources"][1]["share"] = 2.0
        with pytest.raises(ConsoleDocumentError, match=r"^clips\[0\]\.shots\[0\]\.sources\[1\]\.share"):
            validate_console_document(document)

    def test_confidence_and_energy_are_both_required_so_the_axis_stays_acoustic(
        self, document: dict[str, Any]
    ) -> None:
        """w_g = c_g * e_g separates detection from prominence; neither defaults to one."""
        del _first_shot(document)["sources"][0]["energy"]
        with pytest.raises(ConsoleDocumentError, match=r"sources\[0\]\.energy"):
            validate_console_document(document)

    def test_an_unauthored_caption_is_refused(self, document: dict[str, Any]) -> None:
        _first_shot(document)["default_caption"] = ""
        with pytest.raises(ConsoleDocumentError, match="default_caption"):
            validate_console_document(document)

    def test_a_caption_too_long_for_the_band_is_refused(self, document: dict[str, Any]) -> None:
        _first_shot(document)["raw_caption"] = "x" * (CAPTION_MAX_CHARS + 1)
        with pytest.raises(ConsoleDocumentError, match="reserves two lines"):
            validate_console_document(document)

    def test_shots_must_be_contiguous_from_zero(self, document: dict[str, Any]) -> None:
        document["clips"][0]["shots"][1]["start_ms"] = 5000
        with pytest.raises(ConsoleDocumentError, match="contiguous"):
            validate_console_document(document)

    def test_a_duplicate_source_id_within_a_shot_is_refused(self, document: dict[str, Any]) -> None:
        sources = _first_shot(document)["sources"]
        sources[1]["id"] = sources[0]["id"]
        with pytest.raises(ConsoleDocumentError, match="duplicates"):
            validate_console_document(document)

    def test_a_shot_needs_at_least_one_source(self, document: dict[str, Any]) -> None:
        _first_shot(document)["sources"] = []
        with pytest.raises(ConsoleDocumentError, match="at least 1"):
            validate_console_document(document)

    def test_an_opening_grain_outside_the_four_is_refused(self, document: dict[str, Any]) -> None:
        document["clips"][0]["opening"]["grain"] = "verbose"
        with pytest.raises(ConsoleDocumentError, match="opening.grain"):
            validate_console_document(document)

    def test_an_opening_admitted_set_naming_an_unknown_source_is_refused(
        self, document: dict[str, Any]
    ) -> None:
        document["clips"][0]["opening"]["admitted"] = ["nothing_like_this"]
        with pytest.raises(ConsoleDocumentError, match="unknown source"):
            validate_console_document(document)

    def test_a_config_from_a_different_method_is_refused_by_the_document(
        self, document: dict[str, Any]
    ) -> None:
        document["config"]["method_constants"]["salience"] = "w_g = c_g"
        with pytest.raises(ConsoleDocumentError, match="^config"):
            validate_console_document(document)

    def test_loading_names_the_file_when_it_is_not_json(self, tmp_path: Any) -> None:
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ConsoleDocumentError, match="not JSON"):
            load_console_document(path)


class TestParticipantDocument:
    def test_no_measured_quantity_reaches_the_browser(self, document: dict[str, Any]) -> None:
        narrowed = participant_document(document)
        text = repr(narrowed)
        for measured in ("share", "presence", "confidence", "energy", "sources"):
            assert measured not in text

    def test_the_raw_caption_travels_because_the_first_viewing_plays_it(
        self, document: dict[str, Any]
    ) -> None:
        shot = participant_document(document)["clips"][0]["shots"][0]
        assert shot["raw_caption"] == _first_shot(document)["raw_caption"]

    def test_the_default_policy_caption_does_not_travel_because_the_check_follows_the_endpoint(
        self, document: dict[str, Any]
    ) -> None:
        shot = participant_document(document)["clips"][0]["shots"][0]
        assert "default_caption" not in shot

    def test_the_stamp_travels_so_a_stale_tab_can_be_told_apart(self, document: dict[str, Any]) -> None:
        assert len(participant_document(document)["config_hash"]) == 12


class TestField:
    def test_the_field_is_built_from_the_configuration_in_force_not_stored(
        self, document: dict[str, Any], configuration: Any
    ) -> None:
        from dataclasses import replace

        from dpo.console.config import Calibration

        loose = replace(configuration, calibration=Calibration(half_saturation=0.5))
        tight = replace(configuration, calibration=Calibration(half_saturation=0.001))
        shot = _first_shot(document)
        assert field_of(shot, loose).visibility["idling"] < field_of(shot, tight).visibility["idling"]

    def test_a_group_of_labels_stays_one_source(self, document: dict[str, Any]) -> None:
        field = field_of(_first_shot(document), configuration_of(document))
        footsteps = next(source for source in field.sources if source.id == "footsteps")
        assert footsteps.labels == ("Footsteps", "Speech")
