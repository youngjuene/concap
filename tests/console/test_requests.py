"""Settings become a request: the cache key, what is refused, and what the model is told."""

from __future__ import annotations

from typing import Any

import pytest

from dpo.console.config import Configuration
from dpo.console.document import configuration_of, field_of
from dpo.console.quantities import UNNAMED_GRAIN, Field
from dpo.console.requests import (
    ConsoleTemplateWriter,
    RequestBuilder,
    Settings,
    SettingsError,
    audition_settings,
    every_audition,
    validate_settings,
)


@pytest.fixture
def field(document: dict[str, Any]) -> Field:
    return field_of(document["clips"][0]["shots"][0], configuration_of(document))


@pytest.fixture
def builder(document: dict[str, Any]) -> RequestBuilder:
    return RequestBuilder(configuration_of(document))


def _all(field: Field) -> list[str]:
    return [source.id for source in field.sources]


class TestSettingsKey:
    def test_the_key_is_shot_regime_admitted_and_grain(self, field: Field) -> None:
        settings = Settings(grain="itemized", admitted=("b", "a"), regime=2)
        assert settings.key == "itemized|2|a,b"

    def test_the_admitted_set_is_sorted_so_a_reordering_is_not_a_new_caption(self) -> None:
        assert Settings("itemized", ("a", "b"), 0).key == Settings("itemized", ("b", "a"), 0).key

    def test_at_the_atmospheric_grain_the_key_drops_what_has_no_surface(self, field: Field) -> None:
        """Nothing is named, so admission and the regime cannot distinguish two captions."""
        one = validate_settings(field, {"grain": UNNAMED_GRAIN, "admitted": _all(field), "regime": 0})
        two = validate_settings(field, {"grain": UNNAMED_GRAIN, "admitted": [_all(field)[0]], "regime": 0})
        assert one.key == two.key == UNNAMED_GRAIN
        assert one.admitted == () and one.regime == 0


class TestValidation:
    def test_a_grain_outside_the_four_is_refused(self, field: Field) -> None:
        with pytest.raises(SettingsError, match="grain must be one of"):
            validate_settings(field, {"grain": "verbose", "admitted": _all(field), "regime": 0})

    def test_a_regime_the_admitted_set_cannot_reach_is_refused(self, field: Field) -> None:
        with pytest.raises(SettingsError, match="not one of the"):
            validate_settings(field, {"grain": "itemized", "admitted": _all(field), "regime": 99})

    def test_an_unknown_source_is_refused(self, field: Field) -> None:
        with pytest.raises(SettingsError, match="unknown source"):
            validate_settings(field, {"grain": "itemized", "admitted": ["ghost"], "regime": 0})

    def test_a_repeated_source_is_refused(self, field: Field) -> None:
        first = _all(field)[0]
        with pytest.raises(SettingsError, match="repeats"):
            validate_settings(field, {"grain": "itemized", "admitted": [first, first], "regime": 0})

    def test_an_empty_admitted_set_is_refused_at_a_named_grain(self, field: Field) -> None:
        with pytest.raises(SettingsError, match="must not be empty"):
            validate_settings(field, {"grain": "itemized", "admitted": [], "regime": 0})

    def test_a_boolean_regime_is_not_an_integer_index(self, field: Field) -> None:
        with pytest.raises(SettingsError, match="integer index"):
            validate_settings(field, {"grain": "itemized", "admitted": _all(field), "regime": True})


class TestSpecs:
    def test_sources_come_in_the_regime_order_with_phrases_not_numbers(
        self, field: Field, builder: RequestBuilder
    ) -> None:
        settings = validate_settings(field, {"grain": "itemized", "admitted": _all(field), "regime": 0})
        specs = builder.specs(field, settings)
        assert [spec.id for spec in specs] == field.regimes(settings.admitted)[0]["order"]
        for spec in specs:
            assert not any(character.isdigit() for phrase in spec.phrases for character in phrase)

    def test_the_two_ends_of_the_axis_order_the_specs_differently(
        self, field: Field, builder: RequestBuilder
    ) -> None:
        regimes = field.regimes(_all(field))
        eye = builder.specs(field, Settings("itemized", tuple(_all(field)), 0))
        ear = builder.specs(field, Settings("itemized", tuple(_all(field)), len(regimes) - 1))
        assert [spec.id for spec in eye] != [spec.id for spec in ear]

    def test_the_atmospheric_grain_carries_no_sources_at_all(
        self, field: Field, builder: RequestBuilder
    ) -> None:
        assert builder.specs(field, Settings(UNNAMED_GRAIN, (), 0)) == ()


class TestInstruction:
    def test_the_instruction_names_the_sources_in_order_and_forbids_others(
        self, field: Field, builder: RequestBuilder, document: dict[str, Any]
    ) -> None:
        settings = validate_settings(field, {"grain": "itemized", "admitted": _all(field), "regime": 0})
        request = builder.build("amsterdam_006", document["clips"][0]["shots"][0], field, settings)
        text = builder.instruction(request)
        assert "no other source" in text
        positions = [text.index(spec.prose) for spec in request.sources]
        assert positions == sorted(positions)

    def test_the_out_of_frame_rule_quotes_the_configuration_not_a_literal(
        self, field: Field, document: dict[str, Any]
    ) -> None:
        from dpo.console.config import Band, Calibration

        renamed = Configuration(
            "s",
            "c",
            Calibration(visibility_bands=(Band(0.5, "well seen"), Band(0.0, "nowhere in shot"))),
        )
        builder = RequestBuilder(renamed)
        settings = Settings("itemized", tuple(_all(field)), 0)
        request = builder.build("amsterdam_006", document["clips"][0]["shots"][0], field, settings)
        assert "nowhere in shot" in builder.instruction(request)
        assert "out of frame" not in builder.instruction(request)

    def test_the_model_is_never_told_what_produced_the_order(
        self, field: Field, builder: RequestBuilder, document: dict[str, Any]
    ) -> None:
        settings = Settings("itemized", tuple(_all(field)), 0)
        text = builder.instruction(builder.build("c", document["clips"][0]["shots"][0], field, settings))
        for leak in ("alpha", "α", "weight", "participant", "salience", "visibility ratio"):
            assert leak not in text

    def test_the_atmospheric_instruction_forbids_naming_anything(
        self, field: Field, builder: RequestBuilder, document: dict[str, Any]
    ) -> None:
        settings = Settings(UNNAMED_GRAIN, (), 0)
        text = builder.instruction(builder.build("c", document["clips"][0]["shots"][0], field, settings))
        assert "no noun naming any source" in text


class TestTemplateWriter:
    def test_each_grain_writes_a_different_sentence(
        self, field: Field, builder: RequestBuilder, document: dict[str, Any]
    ) -> None:
        writer = ConsoleTemplateWriter()
        shot = document["clips"][0]["shots"][0]
        captions = {
            grain: writer.write(
                builder.build(
                    "c",
                    shot,
                    field,
                    validate_settings(field, {"grain": grain, "admitted": _all(field), "regime": 0}),
                )
            )
            for grain in ("itemized", "grouped", "scene", UNNAMED_GRAIN)
        }
        assert len(set(captions.values())) == 4

    def test_the_atmospheric_caption_names_no_source(
        self, field: Field, builder: RequestBuilder, document: dict[str, Any]
    ) -> None:
        writer = ConsoleTemplateWriter()
        caption = writer.write(
            builder.build("c", document["clips"][0]["shots"][0], field, Settings(UNNAMED_GRAIN, (), 0))
        )
        for source in field.sources:
            assert source.prose not in caption

    def test_a_named_grain_with_nothing_admitted_is_an_error_not_an_empty_caption(self) -> None:
        from dpo.caption.writer import CaptionRequest

        request = CaptionRequest("c", "s", "shaped", "itemized", "k", (), (), "", "", None)
        with pytest.raises(SettingsError):
            ConsoleTemplateWriter().write(request)


class TestAuditions:
    def test_an_audition_is_one_source_alone_and_a_real_point_in_the_space(self, field: Field) -> None:
        settings = audition_settings("siren")
        assert settings.admitted == ("siren",) and settings.grain == "itemized"

    def test_every_source_gets_one(self, field: Field) -> None:
        assert set(every_audition(field)) == {source.id for source in field.sources}
