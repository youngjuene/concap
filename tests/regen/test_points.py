"""§4: matching points to objects, once, with the overlap rule and the leftovers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dpo.regen.document import objects_of
from dpo.regen.points import (
    UNCLASSIFIED,
    PointError,
    match_points,
    matched_labels,
    parse_points,
    summary_of,
)


def at(*coordinates: tuple[float, float]) -> Any:
    return [{"x": x, "y": y} for x, y in coordinates]


class TestParsing:
    def test_creation_order_is_the_order_sent(self) -> None:
        points = parse_points(at((0.1, 0.1), (0.9, 0.9)))
        assert [point.order for point in points] == [0, 1]

    @pytest.mark.parametrize("bad", [at((1.5, 0.5)), at((-0.1, 0.5)), [{"x": "a", "y": 0.1}], "points"])
    def test_coordinates_outside_the_image_are_refused(self, bad: Any) -> None:
        with pytest.raises(PointError):
            parse_points(bad)


class TestMatching:
    def test_a_point_inside_one_mask_takes_its_label(self, document: dict[str, Any], media_dir: Path) -> None:
        objects = objects_of(document, media_dir, "A")
        # (0.40, 0.80) is in the left half — building — and outside the small square.
        matches = match_points(parse_points(at((0.40, 0.80))), objects)
        assert (matches[0].object_id, matches[0].label) == ("building", "Building")

    def test_the_smaller_mask_wins_where_two_overlap(self, document: dict[str, Any], media_dir: Path) -> None:
        # (0.20, 0.20) is inside both the left-half building and the 20x20
        # person square. §4: the person occludes the building there.
        objects = objects_of(document, media_dir, "A")
        matches = match_points(parse_points(at((0.20, 0.20))), objects)
        assert matches[0].label == "Person"

    def test_a_point_outside_every_mask_is_kept_as_unclassified(
        self, document: dict[str, Any], media_dir: Path
    ) -> None:
        objects = objects_of(document, media_dir, "A")
        matches = match_points(parse_points(at((0.95, 0.95))), objects)
        assert matches[0].object_id is None
        record = matches[0].record()
        assert record["label"] == UNCLASSIFIED
        # The coordinates survive: they are evidence about the segmentation.
        assert (record["x"], record["y"]) == (0.95, 0.95)


class TestSummary:
    def test_the_count_and_proportion_are_over_the_points_placed(
        self, document: dict[str, Any], media_dir: Path
    ) -> None:
        objects = objects_of(document, media_dir, "A")
        matches = match_points(parse_points(at((0.20, 0.20), (0.95, 0.95))), objects)
        assert summary_of(matches) == {"points": 2, "unclassified": 1, "unclassified_proportion": 0.5}

    def test_no_points_is_not_a_division_by_zero(self) -> None:
        assert summary_of(())["unclassified_proportion"] == 0.0


class TestLabelsForRegeneration:
    def test_unclassified_points_carry_no_label_into_the_prompt(
        self, document: dict[str, Any], media_dir: Path
    ) -> None:
        objects = objects_of(document, media_dir, "A")
        matches = match_points(parse_points(at((0.95, 0.95), (0.20, 0.20))), objects)
        assert matched_labels(matches) == ("Person",)

    def test_three_clicks_on_one_object_are_one_label_in_first_seen_order(
        self, document: dict[str, Any], media_dir: Path
    ) -> None:
        objects = objects_of(document, media_dir, "A")
        points = parse_points(at((0.40, 0.80), (0.20, 0.20), (0.45, 0.85)))
        assert matched_labels(match_points(points, objects)) == ("Building", "Person")
