"""The configuration artifact, its stamp, and the one slot count §9.4 requires."""

from __future__ import annotations

import pytest

from dpo.regen.config import (
    CONFIG_SCHEMA,
    METHOD_CONSTANTS,
    Calibration,
    ConfigError,
    Configuration,
    Scale,
    load_configuration,
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

    def test_a_changed_slot_count_changes_the_stamp(self) -> None:
        assert _configuration(cue_slots=4).hash != _configuration(cue_slots=6).hash

    def test_two_corpora_calibrated_alike_are_still_two_corpora(self) -> None:
        assert _configuration(corpus_id="amsterdam").hash != _configuration(corpus_id="singapore").hash

    def test_a_changed_anchor_word_changes_the_stamp(self) -> None:
        other = Scale(points=7, anchors=("Never", "Always"))
        assert _configuration(scale=other).hash != _configuration().hash

    def test_the_stamp_travels_on_every_result(self) -> None:
        configuration = _configuration()
        assert configuration.stamped({"a": 1})["config_hash"] == configuration.hash


class TestRoundTrip:
    def test_an_artifact_reloads_to_the_same_stamp(self) -> None:
        original = _configuration(cue_slots=5, languages=("ko", "en"))
        assert load_configuration(original.artifact()).hash == original.hash

    def test_a_foreign_schema_is_refused(self) -> None:
        artifact = {**_configuration().artifact(), "schema": "something/v1"}
        with pytest.raises(ConfigError, match=CONFIG_SCHEMA):
            load_configuration(artifact)

    def test_an_artifact_from_a_different_method_is_refused(self) -> None:
        # The failure the stamp exists to prevent: two studies sharing a hash
        # while the code that computed them differs.
        artifact = _configuration().artifact()
        artifact["method_constants"] = {**METHOD_CONSTANTS, "point_matching": "largest containing mask wins"}
        with pytest.raises(ConfigError, match="method constants"):
            load_configuration(artifact)


class TestScale:
    def test_answers_outside_the_range_are_not_accepted(self) -> None:
        scale = Scale(points=7)
        assert scale.accepts(1) and scale.accepts(7)
        assert not scale.accepts(0)
        assert not scale.accepts(8)

    def test_a_boolean_is_not_an_answer(self) -> None:
        # True == 1 in Python, and a page that posts a checkbox must not have
        # it silently read as the bottom of the scale.
        assert not Scale().accepts(True)

    @pytest.mark.parametrize("points", [1, 0, -3])
    def test_a_scale_needs_two_points(self, points: int) -> None:
        with pytest.raises(ConfigError):
            Scale(points=points)


class TestCalibration:
    def test_one_mark_is_enough_to_go_on(self) -> None:
        # The floor exists to stop an empty submission and nothing more. A
        # higher one makes a participant invent marks to get past the screen,
        # which corrupts the measure rather than thinning it.
        assert Calibration().minimum_points == 1

    @pytest.mark.parametrize(
        "overrides",
        [
            {"cue_slots": 0},
            {"slot_max_chars": 0},
            {"slot_max_lines": 0},
            {"minimum_points": 0},
            {"latency_ceiling_ms": 0},
            {"languages": ()},
            {"languages": ("  ",)},
            {"languages": ("en", "en")},
        ],
    )
    def test_a_calibration_a_study_could_not_run_under_is_refused(self, overrides: dict[str, object]) -> None:
        with pytest.raises(ConfigError):
            Calibration(**overrides)  # type: ignore[arg-type]
