"""§4: turning the points a participant placed into objects they named.

A participant clicks on the five-second still to say what characterises the
scene. Those clicks are coordinates; the study needs labels. This module is the
one place that crosses between them, and it does so once, on the coordinates
submitted with the page — not live, as they are placed.

*Once* is a requirement, not an optimisation. Matching on every click would put
the mask tree's answer on screen while the participant is still deciding, and a
point that lights up as "Building" invites them to keep clicking until it says
something, which measures the mask tree rather than their perception.

**Overlap.** Masks overlap by construction: a person stands in front of a
building, and both masks contain that pixel. The smallest containing mask wins.
A smaller mask is the more specific claim about a pixel, and it is also what
the participant saw — the person occludes the building there, so the building's
mask covering that pixel is an artefact of how the masks were cut, not
something visible in the frame. Ties, which mean two masks of exactly equal
area, go to document order so the result is deterministic.

**Points that match nothing** are kept, tagged ``unclassified``, with their
coordinates intact (§4). They are not a failure to be cleaned up: a participant
pointing at something the segmentation has no object for is evidence about the
segmentation, and dropping those points would silently improve every downstream
count. §6 excludes them from the regeneration inputs — they carry no label to
put in a prompt — but the record keeps them, and the log carries their count
and proportion so the exclusion is visible in the data rather than implied.

Section numbers cite ``docs/v3-regen/spec-behavior.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

UNCLASSIFIED = "unclassified"
# A mask PNG is a binary mask written as 8-bit grey; anything above mid-grey is
# inside. The same threshold the overlay renderer uses on the same trees.
INSIDE = 127


class PointError(ValueError):
    """A point, or a mask, that cannot be matched."""


@dataclass(frozen=True)
class Point:
    """One click, normalised to the image (§4)."""

    order: int
    x: float
    y: float


@dataclass(frozen=True)
class MaskObject:
    """One segmented object of one segment's still."""

    id: str
    label: str
    path: Path


@dataclass(frozen=True)
class Match:
    """What one point resolved to.

    ``object_id`` and ``label`` are ``None`` for a point inside no mask; the
    record still carries ``unclassified`` so a reader never has to infer the
    meaning of a null.
    """

    point: Point
    object_id: str | None
    label: str | None

    @property
    def classified(self) -> bool:
        return self.object_id is not None

    def record(self) -> dict[str, Any]:
        return {
            "order": self.point.order,
            "x": self.point.x,
            "y": self.point.y,
            "object_id": self.object_id,
            "label": self.label if self.classified else UNCLASSIFIED,
        }


def parse_points(raw: object) -> tuple[Point, ...]:
    """Read the page's points, in the creation order it sends them in."""
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        raise PointError("points must be a list")
    points = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            raise PointError(f"points[{index}] must be an object")
        try:
            x, y = float(entry["x"]), float(entry["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PointError(f"points[{index}] needs numeric x and y: {exc}") from exc
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise PointError(f"points[{index}] must be normalised to [0, 1]; got ({x}, {y})")
        points.append(Point(order=index, x=x, y=y))
    return tuple(points)


@lru_cache(maxsize=64)
def _mask(path: str) -> tuple[Any, int]:
    """One decoded mask as a boolean array, and its area in pixels.

    Cached by path: every point of one submission tests every mask of one
    segment, and a segment's masks are the same files for every participant.
    """
    import numpy as np
    from PIL import Image

    try:
        with Image.open(path) as handle:
            inside = np.asarray(handle.convert("L")) > INSIDE
    except OSError as exc:
        raise PointError(f"{path}: cannot be read as a mask image: {exc}") from exc
    return inside, int(inside.sum())


def _contains(path: Path, x: float, y: float) -> tuple[bool, int]:
    """Whether the mask covers the normalised point, and the mask's area."""
    inside, area = _mask(str(path))
    height, width = inside.shape
    column = min(width - 1, max(0, round(x * (width - 1))))
    top = min(height - 1, max(0, round(y * (height - 1))))
    return bool(inside[top, column]), area


def match_points(points: Sequence[Point], objects: Sequence[MaskObject]) -> tuple[Match, ...]:
    """Match every point against every mask, smallest container winning.

    Runs once per submission (§4). Objects are tested in document order, and a
    strictly smaller area is needed to displace an incumbent, so equal-area
    masks resolve to the first one declared.
    """
    matches = []
    for point in points:
        best: MaskObject | None = None
        best_area = 0
        for candidate in objects:
            inside, area = _contains(candidate.path, point.x, point.y)
            if inside and (best is None or area < best_area):
                best, best_area = candidate, area
        matches.append(
            Match(
                point=point,
                object_id=None if best is None else best.id,
                label=None if best is None else best.label,
            )
        )
    return tuple(matches)


def summary_of(matches: Sequence[Match]) -> dict[str, Any]:
    """§4's counts: how many points landed outside every mask, and what share.

    The proportion is over the points placed, so a submission of the minimum
    count with every point unclassified reads as 1.0 rather than as a small
    number that has to be divided by something else to be understood.
    """
    total = len(matches)
    unclassified = sum(1 for match in matches if not match.classified)
    return {
        "points": total,
        "unclassified": unclassified,
        "unclassified_proportion": (unclassified / total) if total else 0.0,
    }


def matched_labels(matches: Sequence[Match]) -> tuple[str, ...]:
    """The labels §6 conditions on: classified points only, first seen first.

    Deduplicated, because three clicks on one building are one label in a
    prompt, and ordered by first mention so the prompt reflects what the
    participant attended to first rather than the mask tree's own order.
    """
    labels: list[str] = []
    for match in matches:
        if match.label is not None and match.label not in labels:
            labels.append(match.label)
    return tuple(labels)
