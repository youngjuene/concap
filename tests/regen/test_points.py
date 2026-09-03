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

STRIP = 5  # frames in the fixture's strip; a point names one of them


def at(*coordinates: tuple[float, float], frame: int = 0) -> Any:
    return [{"x": x, "y": y, "frame": frame} for x, y in coordinates]


def parsed(*coordinates: tuple[float, float], frame: int = 0) -> Any:
    return parse_points(at(*coordinates, frame=frame), STRIP)


class TestParsing:
    def test_creation_order_is_the_order_sent(self) -> None:
        points = parsed((0.1, 0.1), (0.9, 0.9))
        assert [point.order for point in points] == [0, 1]

    @pytest.mark.parametrize("bad", [at((1.5, 0.5)), at((-0.1, 0.5)), [{"x": "a", "y": 0.1}], "points"])
    def test_coordinates_outside_the_image_are_refused(self, bad: Any) -> None:
        with pytest.raises(PointError):
            parse_points(bad, STRIP)


class TestMatching:
    def test_a_point_inside_one_mask_takes_its_label(self, document: dict[str, Any], media_dir: Path) -> None:
        objects = objects_of(document, media_dir, "A")
        # (0.40, 0.80) is in the left half — building — and outside the small square.
        matches = match_points(parsed((0.40, 0.80)), objects)
        assert (matches[0].object_id, matches[0].label) == ("building", "Building")

    def test_the_smaller_mask_wins_where_two_overlap(self, document: dict[str, Any], media_dir: Path) -> None:
        # (0.20, 0.20) is inside both the left-half building and the 20x20
        # person square. §4: the person occludes the building there.
        objects = objects_of(document, media_dir, "A")
        matches = match_points(parsed((0.20, 0.20)), objects)
        assert matches[0].label == "Person"

    def test_a_point_outside_every_mask_is_kept_as_unclassified(
        self, document: dict[str, Any], media_dir: Path
    ) -> None:
        objects = objects_of(document, media_dir, "A")
        matches = match_points(parsed((0.95, 0.95)), objects)
        assert matches[0].object_id is None
        record = matches[0].record()
        assert record["label"] == UNCLASSIFIED
        # The coordinates survive: they are evidence about the segmentation.
        assert (record["x"], record["y"]) == (0.95, 0.95)


class TestFrames:
    """A point is matched against the frame it was placed on, never another."""

    def test_a_point_names_the_frame_it_was_placed_on(self) -> None:
        assert [point.frame for point in parsed((0.2, 0.2), frame=3)] == [3]

    def test_a_frame_outside_the_strip_is_refused(self) -> None:
        with pytest.raises(PointError, match="frame"):
            parse_points(at((0.2, 0.2), frame=STRIP), STRIP)

    def test_a_point_missing_its_frame_belongs_to_the_first(self) -> None:
        assert parse_points([{"x": 0.2, "y": 0.2}], STRIP)[0].frame == 0

    def test_a_point_is_matched_against_its_own_frames_masks(
        self, document: dict[str, Any], media_dir: Path
    ) -> None:
        objects = objects_of(document, media_dir, "A")
        # The fixture cuts every frame alike, so the routing is what is under
        # test: a point on frame 4 must reach frame 4's masks, not frame 0's.
        assert match_points(parsed((0.20, 0.20), frame=4), objects)[0].label == "Person"
        assert match_points(parsed((0.20, 0.20), frame=4), objects)[0].point.frame == 4

    def test_a_frame_with_no_masks_leaves_its_points_unclassified(
        self, document: dict[str, Any], media_dir: Path
    ) -> None:
        objects = dict(objects_of(document, media_dir, "A"))
        objects.pop(2)
        matches = match_points(parsed((0.20, 0.20), frame=2), objects)
        assert matches[0].record()["label"] == UNCLASSIFIED

    def test_the_frame_travels_into_the_record(self, document: dict[str, Any], media_dir: Path) -> None:
        objects = objects_of(document, media_dir, "A")
        assert match_points(parsed((0.20, 0.20), frame=1), objects)[0].record()["frame"] == 1


class TestSummary:
    def test_the_count_and_proportion_are_over_the_points_placed(
        self, document: dict[str, Any], media_dir: Path
    ) -> None:
        objects = objects_of(document, media_dir, "A")
        matches = match_points(parsed((0.20, 0.20), (0.95, 0.95)), objects)
        assert summary_of(matches) == {"points": 2, "unclassified": 1, "unclassified_proportion": 0.5}

    def test_no_points_is_not_a_division_by_zero(self) -> None:
        assert summary_of(())["unclassified_proportion"] == 0.0


class TestLabelsForRegeneration:
    def test_unclassified_points_carry_no_label_into_the_prompt(
        self, document: dict[str, Any], media_dir: Path
    ) -> None:
        objects = objects_of(document, media_dir, "A")
        matches = match_points(parsed((0.95, 0.95), (0.20, 0.20)), objects)
        assert matched_labels(matches) == ("Person",)

    def test_three_clicks_on_one_object_are_one_label_in_first_seen_order(
        self, document: dict[str, Any], media_dir: Path
    ) -> None:
        objects = objects_of(document, media_dir, "A")
        points = parsed((0.40, 0.80), (0.20, 0.20), (0.45, 0.85))
        assert matched_labels(match_points(points, objects)) == ("Building", "Person")
