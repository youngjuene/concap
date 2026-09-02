"""Link Sa2VA segmentation evidence to the caption-session parameter model.

The participant interface owns three *choices*: admission, balance, and
detail.  Sa2VA does not make those choices for a participant; it supplies
evidence from which the session configuration can be calibrated:

* a tag in ``tidy_data.csv`` makes an audio source an admission candidate,
  even when its visual grounding mask is empty (an off-screen sound is still a
  sound);
* mask area and coordinates supply the Eye side of an audio balance and both
  Near/Far sides of the visual control;
* repeated audio tags supply a conservative Ear-side salience prior;
* mask persistence supplies a reviewable detail-role and wording suggestion.

The output deliberately keeps raw evidence beside every derived value.  Audio
semantic masks locate a *possible visible source* of a tagged sound; they do
not measure loudness or acoustic onset.  Callers must therefore treat the Ear
weight and temporal wording as calibration suggestions, not ground truth.

Both Sa2VA layouts are accepted:

* propagated: ``{branch}/{clip}/{label}/{frame:05d}.png``;
* 1 fps release: ``{branch}/{clip}/{label}_{second}.png``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from dpo.caption.ontology import OntologyError, TagRecord, load_tags

MASK_LINK_SCHEMA = "dpo.caption-mask-link/v1"
AUDIO_VISUAL_CATEGORIES = (
    "Road",
    "Sidewalk",
    "Building",
    "Vegetation",
    "Terrain",
    "Sky",
    "Person",
    "Road transport",
)

VISUAL_ROLES = {
    "Road": "fixed",
    "Sidewalk": "fixed",
    "Building": "backdrop",
    "Vegetation": "backdrop",
    "Terrain": "backdrop",
    "Sky": "backdrop",
    "Person": "passing",
    "Road transport": "passing",
}

BACKGROUND_FAMILIES = frozenset({"Channel, environment and background", "Natural sounds"})


class MaskParameterError(ValueError):
    """The mask tree and its semantic metadata cannot be linked safely."""


def _numeric_stem(path: Path) -> int | None:
    return int(path.stem) if path.stem.isdigit() else None


def _mask_paths(clip_dir: Path, label: str) -> tuple[str, list[tuple[int, Path]]]:
    label_dir = clip_dir / label
    if label_dir.is_dir():
        numbered: list[tuple[int, Path]] = []
        for path in label_dir.glob("*.png"):
            unit = _numeric_stem(path)
            if unit is not None:
                numbered.append((unit, path))
        return "propagated-frames", sorted(numbered, key=lambda item: item[0])
    per_second: list[tuple[int, Path]] = []
    for path in clip_dir.glob("*.png") if clip_dir.is_dir() else ():
        stem_label, separator, suffix = path.stem.rpartition("_")
        if separator and stem_label == label and suffix.isdigit():
            per_second.append((int(suffix), path))
    return "perframe-seconds", sorted(per_second, key=lambda item: item[0])


def _round(value: float) -> float:
    return round(value, 6)


def _runs(nonempty: Sequence[bool]) -> tuple[int, int]:
    count = longest = current = 0
    for present in nonempty:
        if present:
            current += 1
            longest = max(longest, current)
            if current == 1:
                count += 1
        else:
            current = 0
    return count, longest


def _mask_summary(clip_dir: Path, label: str, fps: float) -> dict[str, Any]:
    layout, paths = _mask_paths(clip_dir, label)
    foreground: list[bool] = []
    areas: list[float] = []
    centers: list[tuple[float, float]] = []
    bottoms: list[float] = []
    union: tuple[float, float, float, float] | None = None
    dimensions: tuple[int, int] | None = None
    for _, path in paths:
        with Image.open(path) as raw:
            image = raw.convert("L")
            width, height = image.size
            if dimensions is None:
                dimensions = (width, height)
            elif dimensions != (width, height):
                raise MaskParameterError(
                    f"{clip_dir.name}/{label}: mask dimensions change from {dimensions} to {(width, height)}"
                )
            histogram = image.histogram()
            area = (width * height - histogram[0]) / (width * height)
            bbox = image.getbbox()
        present = bbox is not None and area > 0.0
        foreground.append(present)
        areas.append(area)
        if not present or bbox is None:
            continue
        left, top, right, bottom = bbox
        normalized = (left / width, top / height, right / width, bottom / height)
        union = (
            normalized
            if union is None
            else (
                min(union[0], normalized[0]),
                min(union[1], normalized[1]),
                max(union[2], normalized[2]),
                max(union[3], normalized[3]),
            )
        )
        centers.append(((left + right) / (2 * width), (top + bottom) / (2 * height)))
        bottoms.append(bottom / height)
    nonempty_areas = [area for area in areas if area > 0.0]
    run_count, longest_run = _runs(foreground)
    observed = len(paths)
    nonempty = len(nonempty_areas)
    milliseconds_per_unit = 1000.0 / fps if layout == "propagated-frames" else 1000.0
    return {
        "layout": layout,
        "path": str((clip_dir / label) if layout == "propagated-frames" else clip_dir),
        "frames_observed": observed,
        "frames_nonempty": nonempty,
        "presence_ratio": _round(nonempty / observed) if observed else 0.0,
        "mean_area": _round(sum(areas) / observed) if observed else 0.0,
        "mean_area_nonempty": _round(sum(nonempty_areas) / nonempty) if nonempty else 0.0,
        "union_bbox": None if union is None else [_round(value) for value in union],
        "mean_bbox_center": (
            None
            if not centers
            else [
                _round(sum(center[0] for center in centers) / len(centers)),
                _round(sum(center[1] for center in centers) / len(centers)),
            ]
        ),
        "mean_bbox_bottom": _round(sum(bottoms) / len(bottoms)) if bottoms else None,
        "visible_runs": run_count,
        "longest_visible_run": longest_run,
        "frame_size": None if dimensions is None else list(dimensions),
        "first_unit": paths[0][0] if paths else None,
        "last_unit": paths[-1][0] if paths else None,
        "first_timestamp_ms": round(paths[0][0] * milliseconds_per_unit) if paths else None,
        "last_timestamp_ms": round(paths[-1][0] * milliseconds_per_unit) if paths else None,
    }


def _scale(values: Sequence[float]) -> list[float]:
    maximum = max(values, default=0.0)
    return [0.0 for _ in values] if maximum <= 0.0 else [_round(value / maximum) for value in values]


def _visibility_phrase(mask: Mapping[str, Any]) -> str:
    presence = float(mask["presence_ratio"])
    area = float(mask["mean_area_nonempty"])
    center = mask["mean_bbox_center"]
    if not mask["frames_nonempty"]:
        return "out of frame"
    if area >= 0.25:
        return "fills the frame"
    if isinstance(center, list) and (center[0] <= 0.2 or center[0] >= 0.8):
        return "at the edge of the frame"
    if presence >= 0.66:
        return "in frame"
    return "partly in frame"


def _temporal_phrase(mask: Mapping[str, Any]) -> str:
    observed = int(mask["frames_observed"])
    nonempty = int(mask["frames_nonempty"])
    if not nonempty:
        return "not visually grounded"
    if observed and nonempty / observed >= 0.85:
        return "visible throughout"
    if int(mask["visible_runs"]) == 1 and int(mask["longest_visible_run"]) <= max(1, observed // 3):
        return "appears briefly"
    if int(mask["visible_runs"]) > 1:
        return "comes and goes"
    return "stays in view"


def _audio_role_evidence(parent: str, tag_count: int, mask: Mapping[str, Any]) -> dict[str, Any]:
    """What is known about an audio source's soundscape role, and nothing more.

    The role heads — Underneath, Stands out, Only here — describe how a sound
    sits in the soundscape. No mask measures that. An earlier version of this
    module guessed one from the top-level family, the tag multiplicity, and
    the mask's presence ratio; on real footage it called a siren "underneath"
    because the segmenter found something siren-shaped in most frames, which
    inverts the one thing about a siren a caption must get right.

    So the evidence is reported and the verdict is withheld: ``role`` is null
    in the session source, ``review_required`` names it, and the document
    validator refuses a session until a researcher has annotated it.
    """
    return {
        "role": None,
        "background_family": parent in BACKGROUND_FAMILIES,
        "tag_count": tag_count,
        "grounding_presence": mask["presence_ratio"],
        "basis": (
            "top-level family and tag multiplicity, reported as evidence only: "
            "visual grounding does not measure how a sound sits in the soundscape"
        ),
    }


def _audio_entries(clip_dir: Path, tags: TagRecord | None, fps: float) -> list[dict[str, Any]]:
    if tags is None:
        return []
    labels = list(tags["counts"])
    masks = [_mask_summary(clip_dir, label, fps) for label in labels]
    eye_raw = [float(mask["presence_ratio"]) * math.sqrt(float(mask["mean_area_nonempty"])) for mask in masks]
    ear_raw = [float(tags["counts"][label]) for label in labels]
    eyes, ears = _scale(eye_raw), _scale(ear_raw)
    entries: list[dict[str, Any]] = []
    for label, mask, eye, ear in zip(labels, masks, eyes, ears, strict=True):
        parent = tags["parents"][label]
        count = tags["counts"][label]
        visibility = _visibility_phrase(mask)
        temporal = _temporal_phrase(mask)
        detail = _audio_role_evidence(parent, count, mask)
        weights = [eye, ear]
        entries.append(
            {
                "id": _identifier(label),
                "label": label,
                "token": label.upper(),
                "top_level_parent_name": parent,
                "tag_count": count,
                "mask": mask,
                "parameters": {
                    "admission": {
                        "candidate": True,
                        "basis": "tidy_data.final_labels",
                    },
                    "balance": {
                        "eye": eye,
                        "ear": ear,
                        "weights": weights,
                        "basis": {
                            "eye": (
                                "mask presence multiplied by square-root foreground area, normalized per clip"
                            ),
                            "ear": "final_labels multiplicity, normalized per clip",
                        },
                    },
                    "detail": detail,
                    "phrases": {
                        "visibility": visibility,
                        "mask_temporal": temporal,
                        "acoustic_temporal_requires_review": True,
                    },
                },
                "session_source": {
                    "id": _identifier(label),
                    "token": label.upper(),
                    "prose": label.lower(),
                    "phrases": [visibility, temporal],
                    "weights": weights,
                    # Null, not a guess: see ``_audio_role_evidence``. The
                    # document validator refuses the session until it is set.
                    "role": None,
                    "review_required": ["prose", "phrases[1]", "role", "weights[1]"],
                },
            }
        )
    return entries


def _visual_entries(clip_dir: Path, fps: float) -> list[dict[str, Any]]:
    labels = list(AUDIO_VISUAL_CATEGORIES)
    masks = [_mask_summary(clip_dir, label, fps) for label in labels]
    near_raw: list[float] = []
    far_raw: list[float] = []
    for mask in masks:
        presence = float(mask["presence_ratio"])
        area = math.sqrt(float(mask["mean_area_nonempty"]))
        bottom = mask["mean_bbox_bottom"]
        vertical = float(bottom) if isinstance(bottom, int | float) else 0.0
        near_raw.append(presence * (0.7 * vertical + 0.3 * area))
        far_raw.append(presence * (0.7 * (1.0 - vertical) + 0.3 * area))
    nears, fars = _scale(near_raw), _scale(far_raw)
    entries: list[dict[str, Any]] = []
    for label, mask, near, far in zip(labels, masks, nears, fars, strict=True):
        size = _visibility_phrase(mask)
        motion = _temporal_phrase(mask)
        role = VISUAL_ROLES[label]
        weights = [near, far]
        entries.append(
            {
                "id": _identifier(label),
                "label": label,
                "token": label.upper(),
                "mask": mask,
                "parameters": {
                    "admission": {
                        "candidate": bool(mask["frames_nonempty"]),
                        "basis": "non-empty predefined-category mask",
                    },
                    "balance": {
                        "near": near,
                        "far": far,
                        "weights": weights,
                        "basis": "normalized mask bottom coordinate, foreground area, and persistence",
                    },
                    "detail": {
                        "role": role,
                        "basis": "predefined visual category",
                    },
                    "phrases": {
                        "size": size,
                        "motion": motion,
                    },
                },
                "session_source": {
                    "id": _identifier(label),
                    "token": label.upper(),
                    "prose": label.lower(),
                    "phrases": [size, motion],
                    "weights": weights,
                    "role": role,
                    "review_required": ["prose", "phrases[1]"],
                },
            }
        )
    return entries


def _identifier(label: str) -> str:
    slug = "".join(character.lower() if character.isalnum() else "_" for character in label)
    return "_".join(part for part in slug.split("_") if part)


def _mask_clip_ids(mask_root: Path) -> set[str]:
    found: set[str] = set()
    for branch in ("audio", "visual"):
        branch_dir = mask_root / branch
        if branch_dir.is_dir():
            found.update(path.name for path in branch_dir.iterdir() if path.is_dir())
    return found


def derive_mask_links(
    mask_root: str | Path,
    tidy_data: str | Path,
    ontology: str | Path,
    *,
    fps: float = 60.0,
    clips: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build one deterministic, inspectable parameter-link manifest."""
    if not math.isfinite(fps) or fps <= 0.0:
        raise MaskParameterError("fps must be a positive finite number")
    root = Path(mask_root)
    tidy_path = Path(tidy_data)
    ontology_path = Path(ontology)
    if not root.is_dir():
        raise MaskParameterError(f"mask root does not exist: {root}")
    try:
        tags = load_tags(tidy_path, ontology_path)
    except OntologyError as exc:
        raise MaskParameterError(str(exc)) from exc
    selected = list(dict.fromkeys(clips)) if clips is not None else sorted(set(tags) | _mask_clip_ids(root))
    unknown = [clip_id for clip_id in selected if clip_id not in tags and clip_id not in _mask_clip_ids(root)]
    if unknown:
        raise MaskParameterError(f"unknown clip {unknown[0]!r}: absent from tidy data and mask root")
    linked: dict[str, Any] = {}
    for clip_id in selected:
        linked[clip_id] = {
            "tag_family_counts": dict(tags[clip_id]["family_counts"]) if clip_id in tags else {},
            "audio": _audio_entries(root / "audio" / clip_id, tags.get(clip_id), fps),
            "visual": _visual_entries(root / "visual" / clip_id, fps),
        }
    return {
        "schema": MASK_LINK_SCHEMA,
        "source": {
            "mask_root": str(root.resolve()),
            "tidy_data": str(tidy_path.resolve()),
            "ontology": str(ontology_path.resolve()),
            "fps": fps,
        },
        "limitations": [
            "top_level_parent_name is an aggregate family/count check; final_labels names the mask prompts",
            "audio masks ground possible visible sources and do not measure loudness or acoustic onset",
            (
                "Ear weights and temporal/detail suggestions require researcher calibration "
                "before a live session"
            ),
            "mask coordinates remain preprocessing evidence and are never exposed on the participant stage",
        ],
        "clips": linked,
    }
