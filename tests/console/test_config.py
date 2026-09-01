"""The versioned configuration artifact and the stamp every result carries."""

from __future__ import annotations

import pytest

from dpo.console.config import (
    CONFIG_SCHEMA,
    METHOD_CONSTANTS,
    Band,
    Calibration,
    ConfigError,
    Configuration,
    load_configuration,
    phrase_for,
)


def _configuration(**overrides: object) -> Configuration:
    return Configuration(
        study_id=str(overrides.pop("study_id", "street2026")),
        corpus_id=str(overrides.pop("corpus_id", "amsterdam")),
        calibration=Calibration(**overrides),  # type: ignore[arg-type]
    )


class TestHash:
    def test_the_same_configuration_hashes_the_same_way_twice(self) -> None:
        assert _configuration().hash == _configuration().hash

    def test_a_changed_calibration_changes_the_stamp(self) -> None:
        assert _configuration(half_saturation=0.02).hash != _configuration(half_saturation=0.03).hash

    def test_two_corpora_calibrated_alike_are_still_two_corpora(self) -> None:
        assert _configuration(corpus_id="amsterdam").hash != _configuration(corpus_id="singapore").hash

    def test_a_changed_band_phrase_changes_the_stamp(self) -> None:
        other = (Band(0.5, "mostly there"), Band(0.0, "out of frame"))
        assert _configuration(visibility_bands=other).hash != _configuration().hash

    def test_the_stamp_goes_on_every_result_that_leaves(self) -> None:
        stamped = _configuration().stamped({"caption": "A siren passes."})
        assert stamped["config_hash"] == _configuration().hash
        assert stamped["caption"] == "A siren passes."

    def test_the_artifact_carries_the_method_constants_it_was_computed_under(self) -> None:
        artifact = _configuration().artifact()
        assert artifact["schema"] == CONFIG_SCHEMA
        assert artifact["method_constants"] == dict(METHOD_CONSTANTS)


class TestBands:
    def test_a_value_takes_the_phrase_of_the_highest_band_it_reaches(self) -> None:
        bands = (Band(0.66, "high"), Band(0.33, "middle"), Band(0.0, "low"))
        assert phrase_for(0.9, bands) == "high"
        assert phrase_for(0.4, bands) == "middle"
        assert phrase_for(0.0, bands) == "low"

    def test_bands_are_read_in_value_order_not_declaration_order(self) -> None:
        bands = (Band(0.0, "low"), Band(0.66, "high"), Band(0.33, "middle"))
        assert phrase_for(0.9, bands) == "high"

    def test_a_configuration_with_no_band_at_zero_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="band starting at zero"):
            Calibration(visibility_bands=(Band(0.5, "seen"),))

    def test_the_configuration_reads_both_axes_through_its_own_bands(self) -> None:
        configuration = _configuration()
        assert configuration.visibility_phrase(0.0) == "out of frame"
        assert configuration.register_phrase(1.0) == "throughout"


class TestCalibration:
    @pytest.mark.parametrize(
        "field,value",
        [
            ("half_saturation", 0.0),
            ("half_saturation", 1.0),
            ("iou_threshold", 1.5),
            ("min_regime", 1.0),
            ("cut_threshold", 0.0),
            ("minimum_shot_ms", 0),
            ("stride_ms", 0),
        ],
    )
    def test_a_value_outside_its_range_is_refused_at_construction(self, field: str, value: object) -> None:
        with pytest.raises(ConfigError):
            Calibration(**{field: value})  # type: ignore[arg-type]


class TestLoading:
    def test_an_artifact_round_trips(self) -> None:
        original = _configuration(half_saturation=0.05)
        assert load_configuration(original.artifact()).hash == original.hash

    def test_an_artifact_from_a_different_method_is_refused_not_warned(self) -> None:
        artifact = _configuration().artifact()
        artifact["method_constants"] = {**artifact["method_constants"], "ordering_saturates": False}
        with pytest.raises(ConfigError, match="method constants"):
            load_configuration(artifact)

    def test_an_artifact_of_another_schema_is_refused(self) -> None:
        artifact = {**_configuration().artifact(), "schema": "something/v1"}
        with pytest.raises(ConfigError, match=CONFIG_SCHEMA):
            load_configuration(artifact)

    def test_a_missing_field_names_itself(self) -> None:
        artifact = _configuration().artifact()
        del artifact["corpus_id"]
        with pytest.raises(ConfigError, match="missing or misnames"):
            load_configuration(artifact)
