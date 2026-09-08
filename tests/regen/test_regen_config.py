"""The configuration artifact, its stamp, and the one slot count §9.4 requires."""

from __future__ import annotations

import pytest

from dpo.regen.config import (
    CONFIG_SCHEMA,
    METHOD_CONSTANTS,
    SOUND_FAMILIES,
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


class TestTheSoundFamiliesAreHashed:
    """§5's vocabulary is a study input, so the stamp has to cover it.

    The prose beside each family is handed straight to §6's writer, so editing
    one changes the captions a participant is shown. For a while the artifact
    hashed only ``method_constants``, which declares the *shape* of §5's
    question — five families, heard or not — and says nothing about which five
    or what they are called. Two studies asking about different things would
    have shared one stamp.
    """

    def test_they_are_in_the_artifact(self) -> None:
        assert _configuration().artifact()["sound_families"] == dict(SOUND_FAMILIES)

    def test_changing_a_family_s_prose_changes_the_stamp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The prose is what §6 puts in a caption, so this edits the stimulus.
        before = _configuration().hash
        monkeypatch.setattr(
            "dpo.regen.config.SOUND_FAMILIES", {**SOUND_FAMILIES, "natural": "rain and thunder"}
        )
        assert _configuration().hash != before

    def test_adding_a_family_changes_the_stamp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        before = _configuration().hash
        monkeypatch.setattr("dpo.regen.config.SOUND_FAMILIES", {**SOUND_FAMILIES, "silence": "quiet"})
        assert _configuration().hash != before

    def test_an_artifact_with_other_families_is_refused(self) -> None:
        # The same refusal method_constants gets: an artifact written by a
        # build that asked about different sounds cannot be loaded here and
        # stamped as though it were this one.
        artifact = _configuration().artifact()
        artifact["sound_families"] = {**SOUND_FAMILIES, "silence": "quiet"}
        with pytest.raises(ConfigError, match="sound families"):
            load_configuration(artifact)

    def test_an_artifact_written_before_they_were_hashed_is_refused(self) -> None:
        artifact = _configuration().artifact()
        del artifact["sound_families"]
        with pytest.raises(ConfigError, match="sound families"):
            load_configuration(artifact)
