"""§4: turning participant clicks into operational mask-selection records.

A participant places a click on a representative frame. The submitted
coordinates are matched to declared mask codes here, once on submission rather
than while the participant is choosing.

*Once* is a requirement, not an optimisation. Matching on every click would put
the mask tree's answer on screen while the participant is still deciding, and a
point that lights up as "Building" invites them to keep clicking until it says
something, which measures the mask tree rather than their perception.

**One frame each.** §4 shows a strip of frames from the segment and the
participant marks whichever one they scrolled to, so a point carries the frame
it was placed on and is matched against that frame's masks. Objects move; a
mask cut two seconds earlier would put a click on empty road where a person was
standing, and the record would name an object nobody pointed at.

**Overlap.** Select the containing mask with the smallest total foreground
area, breaking equal-area ties by document order. This is the existing
operational selection rule, not a ground-truth judgment about object identity,
visibility, or what the participant perceived. Measurements retain each
original mask's complete area; overlapping pixels are neither subtracted nor
reassigned to a winner map.

**Points that match nothing** are kept, tagged ``unclassified``, with their
coordinates intact (§4). Their occupancy is missing, not zero. §6 excludes
them from regeneration inputs because there is no declared mask label to use;
the records retain their count and proportion so this distinction is explicit.

Section numbers cite ``docs/v3-regen/spec-behavior.md``.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

UNCLASSIFIED = "unclassified"
# A mask PNG is a binary mask written as 8-bit grey; anything above mid-grey is
# inside. The same threshold the overlay renderer uses on the same trees.
INSIDE = 127
MATCHING_RULE = "smallest-containing-mask-v1"


class PointError(ValueError):
    """A point, or a mask, that cannot be matched."""


@dataclass(frozen=True)
class Point:
    """One click, normalised to the image, on one frame of §4's strip."""

    order: int
    frame: int
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
            "frame": self.point.frame,
            "x": self.point.x,
            "y": self.point.y,
            "object_id": self.object_id,
            "label": self.label if self.classified else UNCLASSIFIED,
        }


@dataclass(frozen=True)
class PointMeasurement:
    """Keep the legacy match separate from reproducible whole-mask measurements."""

    match: Match
    metrics: dict[str, Any]


@dataclass(frozen=True)
class _MaskData:
    inside: Any
    area: int
    width: int
    height: int
    sha256: str


_Hit = tuple[MaskObject, _MaskData]


def parse_points(raw: object, frames: int) -> tuple[Point, ...]:
    """Read the page's points, in the creation order it sends them in.

    ``frames`` is how many frames the strip has, so a point can only name one
    that exists: a frame index out of range would otherwise match against no
    masks and read as an honest ``unclassified``.
    """
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
        frame = entry.get("frame", 0)
        if isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < frames:
            raise PointError(f"points[{index}].frame must be a frame of the strip (0..{frames - 1})")
        points.append(Point(order=index, frame=frame, x=x, y=y))
    return tuple(points)


@lru_cache(maxsize=64)
def _decoded_mask(content: bytes) -> _MaskData:
    """Cache the exact decoded bytes, never a mutable path/mtime identity."""
    import numpy as np
    from PIL import Image

    try:
        with Image.open(io.BytesIO(content)) as handle:
            inside = np.asarray(handle.convert("L")) > INSIDE
    except (OSError, SyntaxError, ValueError) as exc:
        raise PointError(f"Cannot decode mask image: {exc}") from exc
    inside.setflags(write=False)
    height, width = inside.shape
    return _MaskData(inside, int(inside.sum()), int(width), int(height), hashlib.sha256(content).hexdigest())


def _read_once(path: Path, contents: dict[Path, bytes]) -> bytes:
    key = path.resolve()
    if key not in contents:
        try:
            contents[key] = key.read_bytes()
        except OSError as exc:
            raise PointError(f"{path}: cannot read image: {exc}") from exc
    return contents[key]


def _contains(mask: _MaskData, x: float, y: float) -> bool:
    column = min(mask.width - 1, max(0, round(x * (mask.width - 1))))
    top = min(mask.height - 1, max(0, round(y * (mask.height - 1))))
    return bool(mask.inside[top, column])


def _candidate_record(hit: _Hit, width: int, height: int) -> dict[str, Any]:
    obj, mask = hit
    aligned = (mask.width, mask.height) == (width, height)
    return {
        "object_id": obj.id,
        "label": obj.label,
        "mask_area_px": mask.area,
        "mask_area_ratio": mask.area / (width * height) if aligned else None,
        "mask_sha256": mask.sha256,
        "mask_width": mask.width,
        "mask_height": mask.height,
        "status": "matched" if aligned else "resolution_mismatch",
    }


def _match_pass(
    points: Sequence[Point], objects: Mapping[int, Sequence[MaskObject]], contents: dict[Path, bytes]
) -> Iterator[tuple[Match, _Hit | None, list[_Hit]]]:
    for point in points:
        best: _Hit | None = None
        hits: list[_Hit] = []
        for candidate in objects.get(point.frame, ()):
            mask = _decoded_mask(_read_once(candidate.path, contents))
            if _contains(mask, point.x, point.y):
                hits.append((candidate, mask))
                if best is None or mask.area < best[1].area:
                    best = candidate, mask
        match = Match(point, best[0].id if best else None, best[0].label if best else None)
        yield match, best, hits


def match_points(points: Sequence[Point], objects: Mapping[int, Sequence[MaskObject]]) -> tuple[Match, ...]:
    """Match every point against its own frame's masks, smallest container winning.

    Runs once per submission (§4). ``objects`` is keyed by the frame's position
    in the strip: a point is matched against the masks cut from the frame it was
    placed on, never from another, or the click would be read against a picture
    the participant was not looking at.

    Objects are tested in document order, and a strictly smaller area is needed
    to displace an incumbent, so equal-area masks resolve to the first declared.
    """
    return tuple(match for match, _, _ in _match_pass(points, objects, {}))


def measure_points(
    points: Sequence[Point],
    objects: Mapping[int, Sequence[MaskObject]],
    frames: Mapping[int, Path],
) -> tuple[PointMeasurement, ...]:
    """Measure the selected original mask against its original frame, once.

    Candidate masks contain the click; their complete foreground areas retain
    overlap and are never subtracted or summed. Resolution differences keep
    the legacy match and raw pixel counts but leave occupancy undefined.
    """
    from PIL import Image

    contents: dict[Path, bytes] = {}
    frame_data: dict[Path, tuple[int, int, str]] = {}
    measurements = []
    for match, best, hits in _match_pass(points, objects, contents):
        if match.point.frame not in frames:
            raise PointError(f"Frame {match.point.frame} has no source image")
        path = frames[match.point.frame].resolve()
        if path not in frame_data:
            content = _read_once(path, contents)
            try:
                with Image.open(io.BytesIO(content)) as handle:
                    handle.load()
                    width, height = handle.size
            except (OSError, SyntaxError, ValueError) as exc:
                raise PointError(f"{path}: cannot decode frame image: {exc}") from exc
            frame_data[path] = width, height, hashlib.sha256(content).hexdigest()
        width, height, frame_hash = frame_data[path]
        total = width * height

        selected = _candidate_record(best, width, height) if best else {}
        metrics = {
            "selected_mask_id": match.object_id,
            "label": match.label if match.classified else UNCLASSIFIED,
            "selected_mask_area_px": selected.get("mask_area_px"),
            "selected_mask_area_ratio": selected.get("mask_area_ratio"),
            "frame_id": str(match.point.frame),
            "frame_width": width,
            "frame_height": height,
            "frame_area_px": total,
            "frame_sha256": frame_hash,
            "mask_sha256": selected.get("mask_sha256"),
            "mask_width": selected.get("mask_width"),
            "mask_height": selected.get("mask_height"),
            "matching_rule": MATCHING_RULE,
            "threshold": INSIDE,
            "candidate_masks": [_candidate_record(hit, width, height) for hit in hits],
            "status": selected.get("status", "unclassified"),
        }
        measurements.append(PointMeasurement(match, metrics))
    return tuple(measurements)


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
