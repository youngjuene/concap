"""From the Sa2VA mask tree to a console document: §3 and §4, over real frames.

Two independent prompt paths run in preprocessing (§3.3). Visual label prompts
drive shot segmentation, and audio label prompts drive source localization. The
label text itself bridges the two modalities, so no fixed correspondence table
between visual and audio vocabularies exists anywhere in the system, and none
appears here.

The pipeline, in the order it runs:

1. Every frame's visual masks give a composition vector ``c(t)`` over the fixed
   visual classes by pixel share. A cut goes where composition shifts by more
   than ``θ``, subject to the floor on shot length (§3.1).
2. Audio label masks are compared pairwise across the whole clip, and labels
   whose masks agree above the IoU threshold merge into one sound source (§4).
   Grouping is per clip, not per shot: a group is a physical thing in the
   scene, and a pedestrian who is one source in the first shot is not two in
   the second.
3. Per shot and per group, the union mask gives ``r_g`` and ``p_g`` (§5.1).
   Area is computed once per group, from the union, never summed across the
   labels that share it.

What this module cannot supply, and will not fake: ``c_g`` and ``e_g``.
``w_g = c_g · e_g`` separates detection from acoustic prominence (§5.1), and
the mask tree measures neither — a mask grounds a *possible visible source* of
a tagged sound, not its loudness or its onset. ``tidy_data.csv`` carries no
per-label confidence, and nothing in this repository computes band-limited
energy. So both fields are written null and named in ``review_required``, and
the document validator refuses the session until a researcher has supplied
them. ``--provisional-salience`` fills them from tag multiplicity for a dry
run, and stamps the manifest with what it did, because a provisional number
that does not announce itself is worse than a missing one.

Frames are read at a reduced resolution. Area ratios and IoU are both ratios,
so the reduction costs a little precision and saves reading 1280×720 for every
label of every frame; the factor is recorded in the manifest's provenance
because ``r_g`` depends on it.

Section numbers cite ``spec-system.md``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dpo.caption.ontology import TagRecord
from dpo.console.config import Calibration
from dpo.console.quantities import (
    QuantityError,
    composition,
    cut_points,
    mean_visual_share,
    presence_rate,
    segments,
)

MANIFEST_SCHEMA = "dpo.caption-console-masks/v1"
VISUAL_CATEGORIES = (
    "Road",
    "Sidewalk",
    "Building",
    "Vegetation",
    "Terrain",
    "Sky",
    "Person",
    "Road transport",
)
DOWNSAMPLE = 4


class MaskReadError(ValueError):
    """The mask tree is not shaped the way the pipeline needs."""


def _label_frames(clip_dir: Path, label: str) -> dict[int, Path]:
    """Frame index to mask path, for either release layout."""
    label_dir = clip_dir / label
    if label_dir.is_dir():
        return {int(path.stem): path for path in label_dir.glob("*.png") if path.stem.isdigit()}
    found: dict[int, Path] = {}
    if clip_dir.is_dir():
        for path in clip_dir.glob("*.png"):
            stem, separator, suffix = path.stem.rpartition("_")
            if separator and stem == label and suffix.isdigit():
                found[int(suffix)] = path
    return found


def _read(path: Path, downsample: int) -> np.ndarray:
    with Image.open(path) as raw:
        image = raw.convert("L")
        if downsample > 1:
            image = image.resize((max(1, image.width // downsample), max(1, image.height // downsample)))
        return np.asarray(image) > 0


@dataclass(frozen=True)
class ClipMasks:
    """One clip's mask tree, read lazily by frame."""

    clip_dir_visual: Path
    clip_dir_audio: Path
    downsample: int

    def visual_frames(self) -> list[int]:
        indices: set[int] = set()
        for label in VISUAL_CATEGORIES:
            indices.update(_label_frames(self.clip_dir_visual, label))
        return sorted(indices)

    def audio_labels(self) -> list[str]:
        directory = self.clip_dir_audio
        if not directory.is_dir():
            return []
        named = sorted(path.name for path in directory.iterdir() if path.is_dir())
        if named:
            return named
        stems = {path.stem.rpartition("_")[0] for path in directory.glob("*.png")}
        return sorted(stem for stem in stems if stem)

    def frame(self, branch: Path, label: str, index: int) -> np.ndarray | None:
        path = _label_frames(branch, label).get(index)
        return None if path is None else _read(path, self.downsample)


# ---- §3.1 shot segmentation -------------------------------------------------


def clip_segments(masks: ClipMasks, calibration: Calibration, fps: float) -> list[tuple[int, int]]:
    """Frame ranges for the clip's shots, cut on visual composition change."""
    frames = masks.visual_frames()
    if not frames:
        raise MaskReadError(f"no visual masks under {masks.clip_dir_visual}")
    shares_by_frame: list[dict[str, float]] = []
    paths = {label: _label_frames(masks.clip_dir_visual, label) for label in VISUAL_CATEGORIES}
    for index in frames:
        shares: dict[str, float] = {}
        for label in VISUAL_CATEGORIES:
            path = paths[label].get(index)
            if path is None:
                shares[label] = 0.0
                continue
            mask = _read(path, masks.downsample)
            shares[label] = float(mask.mean())
        shares_by_frame.append(composition(shares))
    stride = max(1, round(calibration.stride_ms * fps / 1000.0))
    minimum = max(1, round(calibration.minimum_shot_ms * fps / 1000.0))
    cuts = cut_points(
        shares_by_frame,
        threshold=calibration.cut_threshold,
        stride=stride,
        minimum_frames=minimum,
    )
    return segments(len(frames), cuts)


# ---- §4 source grouping -----------------------------------------------------


def group_audio_labels(
    masks: ClipMasks,
    labels: Sequence[str],
    threshold: float,
    families: Mapping[str, str] | None = None,
) -> list[tuple[str, ...]]:
    """Merge labels whose masks agree above ``threshold``, over the whole clip.

    Intersection and union are accumulated in pixels across every frame, so the
    score is a windowed IoU rather than a mean of per-frame IoUs: a label
    present in two frames of six cannot score highly on the strength of those
    two alone.

    ``families`` maps each label to its top-level AudioSet family, and two
    labels from different families never merge however their masks overlap.
    §4's justification for merging is that prompting with two labels "recovers
    the same pedestrian pixels" — labels that share a *physical source*. On a
    real clip the unconstrained merge fused *Bird* with *Traffic noise* and
    *Vehicle horn* because the segmenter returned one blob for all three
    prompts, which is a fact about the grounding, not about the street: an
    animal and a vehicle are not one thing. That merge also produced a source
    whose prose ran to seventy-one characters, which no caption inside the
    budget could name, so every caption mentioning it fell to the template.
    Without ``families`` the merge is unconstrained, and the manifest says so.
    """
    if len(labels) < 2:
        return [(label,) for label in labels]
    paths = {label: _label_frames(masks.clip_dir_audio, label) for label in labels}
    frames = sorted({index for table in paths.values() for index in table})
    intersection = np.zeros((len(labels), len(labels)), dtype=np.int64)
    union = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for index in frames:
        loaded: list[np.ndarray | None] = []
        for label in labels:
            path = paths[label].get(index)
            loaded.append(None if path is None else _read(path, masks.downsample))
        for left in range(len(labels)):
            for right in range(left + 1, len(labels)):
                first, second = loaded[left], loaded[right]
                # A frame one label is absent from still counts toward the
                # union, which is what keeps a label present in two frames of
                # six from scoring as though the other four never happened.
                if first is None:
                    if second is not None:
                        union[left, right] += int(second.sum())
                    continue
                if second is None:
                    union[left, right] += int(first.sum())
                    continue
                if first.shape != second.shape:
                    raise MaskReadError(f"masks for {labels[left]!r} and {labels[right]!r} differ in size")
                intersection[left, right] += int(np.logical_and(first, second).sum())
                union[left, right] += int(np.logical_or(first, second).sum())

    parent = list(range(len(labels)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            if families is not None and families.get(labels[left]) != families.get(labels[right]):
                continue
            total = union[left, right]
            if total and intersection[left, right] / total >= threshold:
                root, other = find(left), find(right)
                if root != other:
                    parent[other] = root
    grouped: dict[int, list[str]] = {}
    for index, label in enumerate(labels):
        grouped.setdefault(find(index), []).append(label)
    return [tuple(members) for members in grouped.values()]


def group_quantities(masks: ClipMasks, members: Sequence[str], frames: Sequence[int]) -> tuple[float, float]:
    """``(r_g, p_g)`` for one group over one shot's frames, from the union mask."""
    paths = {label: _label_frames(masks.clip_dir_audio, label) for label in members}
    areas: list[float] = []
    for index in frames:
        union: np.ndarray | None = None
        for label in members:
            path = paths[label].get(index)
            if path is None:
                continue
            mask = _read(path, masks.downsample)
            union = mask if union is None else np.logical_or(union, mask)
        areas.append(0.0 if union is None else float(union.mean()))
    return mean_visual_share(areas), presence_rate(areas)


# ---- the manifest -----------------------------------------------------------


def _identifier(members: Sequence[str]) -> str:
    joined = "_".join(members)
    slug = "".join(character.lower() if character.isalnum() else "_" for character in joined)
    return "_".join(part for part in slug.split("_") if part)[:64] or "source"


def _prose(members: Sequence[str]) -> str:
    return " and ".join(member.lower() for member in members)


def derive_clip(
    mask_root: str | Path,
    clip_id: str,
    *,
    calibration: Calibration | None = None,
    fps: float = 60.0,
    downsample: int = DOWNSAMPLE,
    tags: TagRecord | None = None,
    provisional_salience: bool = False,
) -> dict[str, Any]:
    """One clip's shots and grouped sources, ready to paste into a document.

    ``tags`` — the clip's audio labels with their families resolved — makes the
    grouping family-aware and the provisional salience possible. Without it
    the grouping is by mask agreement alone.
    """
    settings = calibration or Calibration()
    root = Path(mask_root)
    masks = ClipMasks(root / "visual" / clip_id, root / "audio" / clip_id, downsample)
    frames = masks.visual_frames()
    ranges = clip_segments(masks, settings, fps)
    labels = masks.audio_labels()
    families = dict(tags["parents"]) if tags is not None else None
    groups = group_audio_labels(masks, labels, settings.iou_threshold, families)
    counts = dict(tags["counts"]) if tags is not None else {}
    highest = max(counts.values(), default=0)

    shots: list[dict[str, Any]] = []
    for index, (low, high) in enumerate(ranges):
        window = frames[low:high]
        sources: list[dict[str, Any]] = []
        for members in groups:
            share, presence = group_quantities(masks, members, window)
            # Confidence is a max within the group and energy is summed within
            # it (§4, §5.1) — when either is measured. Neither is here.
            if provisional_salience and highest:
                summed = sum(counts.get(member, 0) for member in members)
                confidence: float | None = 1.0
                energy: float | None = round(summed / (highest * len(groups)), 6)
            else:
                confidence = None
                energy = None
            sources.append(
                {
                    "id": _identifier(members),
                    "labels": list(members),
                    "prose": _prose(members),
                    "share": round(share, 6),
                    "presence": round(presence, 6),
                    "confidence": confidence,
                    "energy": energy,
                    "review_required": ["prose", "confidence", "energy"],
                }
            )
        shots.append(
            {
                "shot_id": f"s{index + 1}",
                "start_ms": round(low * 1000.0 / fps),
                "end_ms": round(high * 1000.0 / fps),
                "raw_caption": "",
                "default_caption": "",
                "sources": sources,
                "frames": [low, high],
            }
        )
    return {
        "clip_id": clip_id,
        "shots": shots,
        "opening": {"grain": "itemized", "alpha": 0.5, "admitted": None},
        "grouping": [list(members) for members in groups],
        "grouping_constraint": "family" if families is not None else "none",
        "families": families or {},
    }


def derive_manifest(
    mask_root: str | Path,
    clips: Iterable[str],
    *,
    calibration: Calibration | None = None,
    fps: float = 60.0,
    downsample: int = DOWNSAMPLE,
    tags: Mapping[str, TagRecord] | None = None,
    provisional_salience: bool = False,
) -> dict[str, Any]:
    """Every clip's preprocessing, with the provenance of every derived number."""
    settings = calibration or Calibration()
    per_clip = {}
    for clip_id in clips:
        per_clip[clip_id] = derive_clip(
            mask_root,
            clip_id,
            calibration=settings,
            fps=fps,
            downsample=downsample,
            tags=(tags or {}).get(clip_id),
            provisional_salience=provisional_salience,
        )
    return {
        "schema": MANIFEST_SCHEMA,
        "source": {
            "mask_root": str(Path(mask_root).resolve()),
            "fps": fps,
            "downsample": downsample,
            "cut_threshold": settings.cut_threshold,
            "minimum_shot_ms": settings.minimum_shot_ms,
            "stride_ms": settings.stride_ms,
            "iou_threshold": settings.iou_threshold,
            "grouping_constraint": "family" if tags else "none",
        },
        "limitations": [
            "shot boundaries are cut on visual composition and are meant to be edited by hand",
            (
                "labels merge on mask agreement within one AudioSet family; without --tidy-data and"
                " --ontology the family constraint is off and cross-family merges are possible"
            ),
            "audio masks ground a possible visible source and measure neither loudness nor onset",
            (
                "c_g and e_g are null: no per-label confidence and no band-limited energy exist"
                " in the inputs, so w_g = c_g * e_g cannot be computed"
                + (" (provisional values were written from tag multiplicity)" if provisional_salience else "")
            ),
            "raw and default-policy captions are authored, not derived",
        ],
        "provisional_salience": provisional_salience,
        "clips": per_clip,
    }


def write_manifest(manifest: Mapping[str, Any], out: str | Path) -> Path:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


__all__ = [
    "DOWNSAMPLE",
    "MANIFEST_SCHEMA",
    "VISUAL_CATEGORIES",
    "ClipMasks",
    "MaskReadError",
    "QuantityError",
    "clip_segments",
    "derive_clip",
    "derive_manifest",
    "group_audio_labels",
    "group_quantities",
    "write_manifest",
]
