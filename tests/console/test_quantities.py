"""The measured half: grouping, the quantities, the regimes, and §9."""

from __future__ import annotations

import pytest

from dpo.console.quantities import (
    Field,
    QuantityError,
    Source,
    composition,
    cut_points,
    group_labels,
    iou,
    mean_visual_share,
    normalize,
    presence_rate,
    salience,
    segments,
    subsets_of,
    total_variation,
    union_occupancy,
    visibility,
)


def _source(source_id: str, share: float, presence: float, confidence: float, energy: float) -> Source:
    return Source(
        id=source_id,
        labels=(source_id.title(),),
        prose=source_id,
        share=share,
        presence=presence,
        confidence=confidence,
        energy=energy,
    )


def _field(*sources: Source, r0: float = 0.02, min_regime: float = 0.02) -> Field:
    return Field(sources=sources, half_saturation=r0, min_regime=min_regime)


# ---- §4 grouping ------------------------------------------------------------


class TestGrouping:
    def test_two_labels_over_the_same_pixels_become_one_source(self) -> None:
        occupancy = {
            "speech": [True, True, True, False],
            "footsteps": [True, True, False, False],
            "siren": [False, False, False, True],
        }
        groups = group_labels(occupancy, 0.5)
        assert sorted(groups, key=len) == [("siren",), ("speech", "footsteps")]

    def test_agreement_is_transitive_because_a_source_is_one_thing(self) -> None:
        # a↔b and b↔c both clear the threshold; a↔c on its own does not.
        occupancy = {
            "a": [True, True, True, False, False, False],
            "b": [False, True, True, True, False, False],
            "c": [False, False, True, True, True, False],
        }
        assert iou(occupancy["a"], occupancy["c"]) < 0.5
        assert group_labels(occupancy, 0.4) == [("a", "b", "c")]

    def test_two_absent_labels_do_not_group_on_both_being_absent(self) -> None:
        occupancy = {"a": [False, False], "b": [False, False]}
        assert iou(occupancy["a"], occupancy["b"]) == 0.0
        assert group_labels(occupancy, 0.5) == [("a",), ("b",)]

    def test_masks_of_different_extents_are_refused_rather_than_zipped_short(self) -> None:
        with pytest.raises(QuantityError):
            iou([True, False], [True])

    def test_grouping_is_a_function_of_input_order_not_dict_iteration(self) -> None:
        occupancy = {"z": [True, False], "a": [True, False], "m": [False, True]}
        assert group_labels(occupancy, 0.5) == [("z", "a"), ("m",)]

    def test_the_group_occupies_the_union_so_area_is_counted_once(self) -> None:
        assert union_occupancy([[True, False, False], [False, True, False]]) == [True, True, False]

    def test_a_threshold_outside_the_unit_interval_is_refused(self) -> None:
        with pytest.raises(QuantityError):
            group_labels({"a": [True]}, 1.5)


# ---- §5 quantities ----------------------------------------------------------


class TestQuantities:
    def test_mean_visual_share_counts_the_frames_the_source_is_absent_from(self) -> None:
        assert mean_visual_share([0.4, 0.0, 0.0, 0.0]) == pytest.approx(0.1)

    def test_presence_rate_is_the_fraction_of_frames_it_occupies_anything(self) -> None:
        assert presence_rate([0.4, 0.0, 0.1, 0.0]) == pytest.approx(0.5)

    def test_salience_is_a_product_so_confidence_alone_cannot_carry_it(self) -> None:
        # A distant siren: high confidence, negligible energy.
        assert salience(0.99, 0.01) < salience(0.5, 0.5)

    def test_visibility_saturates_rather_than_thresholding(self) -> None:
        r0 = 0.02
        assert visibility(r0, r0) == pytest.approx(0.5)
        assert visibility(0.0, r0) == 0.0
        # Doubling a large area moves visibility far less than doubling a small one.
        assert visibility(0.4, r0) - visibility(0.2, r0) < visibility(0.02, r0) - visibility(0.01, r0)

    def test_a_non_positive_half_saturation_is_refused(self) -> None:
        with pytest.raises(QuantityError):
            visibility(0.1, 0.0)


# ---- §6 normalization -------------------------------------------------------


class TestNormalization:
    def test_normalize_is_share_of_the_total(self) -> None:
        assert normalize([1.0, 3.0]) == [0.25, 0.75]

    def test_an_all_zero_axis_normalizes_to_zeros_rather_than_dividing(self) -> None:
        assert normalize([0.0, 0.0]) == [0.0, 0.0]

    def test_normalizing_over_the_full_set_keeps_removal_subtractive(self) -> None:
        """Muting one source must not reweight the two the participant never touched."""
        field = _field(
            _source("a", 0.10, 0.9, 1.0, 0.5),
            _source("b", 0.05, 0.5, 1.0, 0.3),
            _source("c", 0.20, 0.8, 1.0, 0.9),
        )
        before = field.score("a", 0.5)
        # The field is built once from the full set; filtering happens after.
        assert field.regimes(["a", "b"]) and field.regimes(["a", "b", "c"])
        assert field.score("a", 0.5) == before


# ---- §7 rank ----------------------------------------------------------------


class TestRegimes:
    def test_one_source_is_one_regime_over_the_whole_axis(self) -> None:
        field = _field(_source("a", 0.1, 0.5, 1.0, 0.5))
        assert field.regimes(["a"]) == [{"order": ["a"], "span": [0.0, 1.0]}]

    def test_sources_that_never_cross_give_one_regime(self) -> None:
        field = _field(_source("loud", 0.4, 0.9, 1.0, 0.9), _source("quiet", 0.05, 0.2, 1.0, 0.1))
        assert len(field.regimes(["loud", "quiet"])) == 1

    def test_the_endpoints_are_the_visibility_first_and_salience_first_orders(self) -> None:
        # Seen but barely heard against heard but barely seen.
        field = _field(_source("seen", 0.5, 0.9, 1.0, 0.05), _source("heard", 0.002, 0.3, 1.0, 0.9))
        regimes = field.regimes(["seen", "heard"])
        assert regimes[0]["order"][0] == "seen"
        assert regimes[-1]["order"][0] == "heard"

    def test_spans_partition_the_axis(self) -> None:
        field = _field(
            _source("a", 0.30, 0.9, 1.0, 0.10),
            _source("b", 0.10, 0.6, 1.0, 0.40),
            _source("c", 0.02, 0.3, 1.0, 0.90),
        )
        regimes = field.regimes(["a", "b", "c"])
        assert regimes[0]["span"][0] == 0.0
        assert regimes[-1]["span"][1] == 1.0
        for earlier, later in zip(regimes, regimes[1:], strict=False):
            assert earlier["span"][1] == pytest.approx(later["span"][0])

    def test_regimes_narrower_than_the_calibration_are_folded_away(self) -> None:
        sources = [
            _source(f"s{index}", 0.05 + index * 1e-5, 0.5, 1.0, 0.5 - index * 1e-5 + index * index * 1e-6)
            for index in range(5)
        ]
        loose = Field(sources=tuple(sources), half_saturation=0.02, min_regime=0.0)
        tight = Field(sources=tuple(sources), half_saturation=0.02, min_regime=0.05)
        ids = [source.id for source in sources]
        assert len(tight.regimes(ids)) <= len(loose.regimes(ids))
        assert all(r["span"][1] - r["span"][0] >= 0.05 - 1e-12 for r in tight.regimes(ids))

    def test_folding_never_loses_the_two_endpoint_orders(self) -> None:
        sources = [
            _source(f"s{index}", 0.05 + index * 1e-5, 0.5, 1.0, 0.5 - index * 1e-5 + index * index * 1e-6)
            for index in range(5)
        ]
        ids = [source.id for source in sources]
        raw = Field(sources=tuple(sources), half_saturation=0.02, min_regime=0.0).regimes(ids)
        folded = Field(sources=tuple(sources), half_saturation=0.02, min_regime=0.1).regimes(ids)
        assert folded[0]["order"] == raw[0]["order"]
        assert folded[-1]["order"] == raw[-1]["order"]

    def test_a_regime_ranks_only_the_admitted_sources(self) -> None:
        field = _field(_source("a", 0.2, 0.8, 1.0, 0.4), _source("b", 0.1, 0.4, 1.0, 0.9))
        assert field.regimes(["a"])[0]["order"] == ["a"]

    def test_subsets_of_covers_every_non_empty_admitted_set(self) -> None:
        assert subsets_of(["a", "b"]) == [("a",), ("b",), ("a", "b")]

    def test_two_sources_of_one_shot_may_not_share_an_id(self) -> None:
        with pytest.raises(QuantityError):
            _field(_source("a", 0.1, 0.5, 1.0, 0.5), _source("a", 0.2, 0.5, 1.0, 0.5))


# ---- §9 measurements --------------------------------------------------------


class TestMeasurements:
    def test_anchoring_is_one_when_everything_heard_is_fully_visible(self) -> None:
        # A source occupying far more than r0 saturates toward one.
        field = _field(_source("a", 100.0, 1.0, 1.0, 1.0))
        assert field.anchoring() == pytest.approx(1.0, abs=1e-3)

    def test_anchoring_is_near_zero_when_what_is_heard_cannot_be_seen(self) -> None:
        field = _field(_source("offscreen", 0.0, 0.0, 1.0, 1.0))
        assert field.anchoring() == 0.0

    def test_anchoring_is_undefined_when_nothing_is_heard(self) -> None:
        assert _field(_source("a", 0.2, 0.8, 0.0, 0.0)).anchoring() is None

    def test_divergence_is_zero_when_the_modalities_agree(self) -> None:
        field = _field(_source("a", 0.1, 0.5, 1.0, 0.5), _source("b", 0.1, 0.5, 1.0, 0.5))
        assert field.divergence() == pytest.approx(0.0)

    def test_divergence_is_undefined_when_nothing_is_visible(self) -> None:
        """A meaningful state rather than an error (§9)."""
        field = _field(_source("a", 0.0, 0.0, 1.0, 0.5), _source("b", 0.0, 0.0, 1.0, 0.5))
        assert field.divergence() is None

    def test_residuals_locate_the_disagreement_and_sum_to_zero(self) -> None:
        field = _field(_source("seen", 0.4, 0.9, 1.0, 0.1), _source("heard", 0.005, 0.2, 1.0, 0.9))
        residuals = field.residuals()
        assert residuals["heard"] > 0 > residuals["seen"]
        assert sum(residuals.values()) == pytest.approx(0.0)

    def test_measurements_gathers_the_three_without_a_screen_ever_seeing_them(self) -> None:
        field = _field(_source("a", 0.2, 0.8, 1.0, 0.4))
        assert set(field.measurements()) == {"anchoring", "divergence", "residuals"}


# ---- §3.1 segmentation ------------------------------------------------------


class TestSegmentation:
    def test_composition_normalizes_pixel_shares(self) -> None:
        assert composition({"road": 1.0, "sky": 3.0}) == {"road": 0.25, "sky": 0.75}

    def test_total_variation_is_the_measure_that_also_reports_divergence(self) -> None:
        assert total_variation({"a": 1.0}, {"a": 0.0, "b": 1.0}) == pytest.approx(1.0)
        assert total_variation({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}) == 0.0

    def test_a_cut_goes_where_composition_shifts_past_the_threshold(self) -> None:
        frames = [{"road": 1.0, "sky": 0.0}] * 5 + [{"road": 0.0, "sky": 1.0}] * 5
        assert cut_points(frames, threshold=0.5, stride=1, minimum_frames=1) == [5]

    def test_the_floor_on_shot_length_suppresses_a_second_cut(self) -> None:
        frames = [{"a": 1.0}] * 3 + [{"b": 1.0}] * 3 + [{"a": 1.0}] * 3
        assert cut_points(frames, threshold=0.5, stride=1, minimum_frames=1) == [3, 6]
        # Neither cut can be taken under a five-frame floor: the first leaves a
        # three-frame head, the second a three-frame tail. The clip stays whole.
        assert cut_points(frames, threshold=0.5, stride=1, minimum_frames=5) == []

    def test_the_floor_binds_on_the_last_shot_too(self) -> None:
        # A change eight frames in, with only two frames left after it.
        frames = [{"a": 1.0}] * 8 + [{"b": 1.0}] * 2
        assert cut_points(frames, threshold=0.5, stride=1, minimum_frames=1) == [8]
        assert cut_points(frames, threshold=0.5, stride=1, minimum_frames=3) == []

    def test_segments_cover_the_whole_clip(self) -> None:
        assert segments(10, [4]) == [(0, 4), (4, 10)]
        assert segments(10, []) == [(0, 10)]

    def test_a_cut_outside_the_clip_is_ignored_rather_than_folded_in(self) -> None:
        assert segments(10, [0, 10, 25]) == [(0, 10)]
