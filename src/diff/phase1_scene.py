"""Phase-1 scene parts: what the Sa2VA masks say about each clip, for the binding model.

Researcher's request of 2026-10-23: the second caption's direction comes from a
Bayesian model (:mod:`diff.phase1_binding`) over the response scales, the scene's
visual mask proportions, and whether the sound the participant noticed has a
visible source in its Sa2VA audio mask, with sounds grouped by AudioSet label.
This module computes the participant-independent half once per setup, offline,
from a Sa2VA run laid out as ``masks/visual/<clip>/<Category>/<frame>.png`` and
``masks/audio/<clip>/<AudioSet label>/<frame>.png`` (60 fps, the clip's own frame
numbers, the stills' resolution):

- **Composition:** the share of the frame each visual category holds, every
  ``stride``-th frame. Category masks overlap, so each pixel goes to the smallest
  mask containing it (regen's click rule) and pixels in no mask count as
  ``none``; the shares then sum to 1. The categories are grouped into five
  elements (:data:`ELEMENTS`).
- **What sets the caption type:** the clip's distinctive element, the one whose
  share most exceeds its mean over the setup's clips ("built" leads in 10 of the
  12 c2 clips, so the largest share alone would not tell scenes apart), or the
  largest element when no lift reaches :data:`MIN_LIFT`; and its openness from
  its sky share (:func:`openness`).
- **Where each A1s sound's source can be seen:** the option's AudioSet label
  (``audioset_label``, the audio caption's category match) gives its groups, the
  ancestors one level under AudioSet's top categories, and they give the visual
  categories the source can be seen in (:data:`PLAUSIBLE`). Sa2VA returns audio
  masks for sounds with no visible source too (microphone wind on terrain, a bird
  on a person, measured 2026-10-23), so a mask alone is not a seen source.
- **Visibility is measured from the masks, never judged** (researcher,
  2026-10-24, replacing the design review's per-option judgement): the share of
  the label's audio mask that lies on those categories (:func:`on_labels`),
  averaged over the sampled frames, a frame whose audio mask is empty counting 0
  (a frame with no mask file is skipped). Unlabelled pixels count against it, so
  a source Sa2VA's visual masks miss is not seen (singapore_067's distant
  talkers, outside its Person mask). The source is on screen from
  :data:`ON_SCREEN`: on average for the caption, per frame for a click. The
  binding model's prior is the share's Beta posterior mean.
- **What "visibility" means here: candidate-source availability** (design
  review, 2026-10-25). Sa2VA never hears the audio: it draws an audio mask from
  the frames and the label's text alone ("segment all possible sources of
  {label} sound in the scene"). So the share says whether a plausible source of
  the label's kind is in view, not whether the actual source is, and nothing
  about whether anything on screen moves with the sound. The names in the code
  (``visibility``, ``on_screen``) are unchanged.
- **Click masks:** each sound's audio mask on each P2 frame, widened by
  :data:`CLICK_TOLERANCE` and copied next to the clip's media, so that the
  participant's point can be tested against it at P5. Several are under 0.5% of
  the frame, smaller than a point's own imprecision.

The scene part is the clip's ``scene`` (``diff.phase1-scene/v3``), part of the
content the researcher's approval binds to.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any, Final

SCENE_SCHEMA: Final = "diff.phase1-scene/v3"
STRIDE: Final = 10
INSIDE: Final = 127  # the mask threshold regen's point matching uses
# The share of an audio mask on its source's categories from which the source is on screen.
ON_SCREEN: Final = 0.5
CLICK_MASKS: Final = "audio-masks"
CLICK_TOLERANCE: Final = 0.015  # the click masks' widening, as a share of the frame's width
# The visual categories grouped into the scene's elements, in the order ties resolve.
ELEMENTS: Final[dict[str, tuple[str, ...]]] = {
    "nature": ("vegetation", "terrain"),
    "open": ("sky",),
    "traffic": ("road", "road_transport"),
    "built": ("building", "sidewalk"),
    "people": ("person",),
}
OPEN_ELEMENT: Final = "open"
MIN_LIFT: Final = 1.5  # a weaker lift is no distinctive element: the largest one stands instead
# Sky shares for an open or an enclosed scene; between them the caption says nothing of the space.
# Set in the gaps of the 12 c2 clips' sky shares (0.063 | 0.084 and 0.128 | 0.183).
OPEN_SKY: Final = 0.15
ENCLOSED_SKY: Final = 0.07
OPENNESS: Final = ("open", "neutral", "enclosed")
# Where each AudioSet group's source can be seen, among the visual categories. A group with
# no category has no visible source: its sounds are heard from off screen or from the recording.
PLAUSIBLE: Final[dict[str, tuple[str, ...]]] = {
    "Vehicle": ("road_transport", "road"),
    "Engine": ("road_transport", "road"),
    "Alarm": ("road_transport",),
    "Human voice": ("person",),
    "Human locomotion": ("person", "sidewalk"),
    "Wild animals": ("vegetation", "sky"),
    "Domestic animals, pets": ("person", "sidewalk", "road"),
    "Music": ("person", "building"),
    "Musical instrument": ("person",),
    "Bell": ("building",),
    "Wind": (),
    "Miscellaneous sources": (),
    "Noise": (),
    "Acoustic environment": (),
}


class SceneError(ValueError):
    """Material a scene part cannot be built from, named where the researcher can act on it."""


def category_id(name: str) -> str:
    """A Sa2VA category directory ("Road transport") as the setup's object id ("road_transport")."""
    return name.strip().lower().replace(" ", "_")


def ontology_groups(path: Path) -> Callable[[str], tuple[str, ...]]:
    """Label → its AudioSet groups: on every path from a top category, the node one level under it."""
    entries = json.loads(path.read_text(encoding="utf-8"))
    ids = {entry["name"]: entry["id"] for entry in entries}
    names = {entry["id"]: entry["name"] for entry in entries}
    parents: dict[str, set[str]] = {}
    for entry in entries:
        for child in entry["child_ids"]:
            parents.setdefault(child, set()).add(entry["id"])

    def paths(node: str) -> list[list[str]]:
        above = parents.get(node)
        if not above:
            return [[node]]
        return [path + [node] for parent in sorted(above) for path in paths(parent)]

    def groups(label: str) -> tuple[str, ...]:
        if label not in ids:
            raise SceneError(f"{label!r} is not in the ontology")
        return tuple(sorted({names[path[1] if len(path) > 1 else path[0]] for path in paths(ids[label])}))

    return groups


def _mask(path: Path) -> Any:
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > INSIDE


def _visual_dirs(run: Path, clip: str) -> dict[str, Path]:
    folder = run / "visual" / clip
    if not folder.is_dir():
        raise SceneError(f"{clip}: no visual masks at {folder}")
    return {category_id(entry.name): entry for entry in sorted(folder.iterdir()) if entry.is_dir()}


def sampled_frames(visual: Mapping[str, Path], stride: int) -> list[str]:
    """Every ``stride``-th frame file name that every category has."""
    names = [{path.name for path in folder.glob("*.png")} for folder in visual.values()]
    common = sorted(set.intersection(*names)) if names else []
    if not common:
        raise SceneError(f"no frame has every category mask in {sorted(visual)}")
    return common[::stride]


def winners(visual: Mapping[str, Path], frame: str) -> Any:
    """Each pixel's category index in ``sorted(visual)``, the smallest containing mask winning;
    ``len(visual)`` where no mask holds it."""
    import numpy as np

    categories = sorted(visual)
    masks = np.stack([_mask(visual[category] / frame) for category in categories])
    area = masks.reshape(len(categories), -1).sum(axis=1).astype(float)
    winner = np.where(masks, area[:, None, None], np.inf).argmin(axis=0).astype(np.int8)
    winner[~masks.any(axis=0)] = len(categories)
    return winner


def composition(names: Sequence[str], frames: Sequence[Any]) -> dict[str, float]:
    """Each category's share of the frame, from the frames' winner maps."""
    import numpy as np

    totals = np.zeros(len(names) + 1)
    for winner in frames:
        totals += np.bincount(winner.ravel(), minlength=len(names) + 1) / winner.size
    shares = totals / len(frames)
    return {name: round(float(share), 4) for name, share in zip([*names, "none"], shares, strict=True)}


def elements_of(shares: Mapping[str, float]) -> dict[str, float]:
    return {name: round(sum(shares.get(c, 0.0) for c in members), 4) for name, members in ELEMENTS.items()}


def plausible_categories(groups: Sequence[str]) -> tuple[str, ...]:
    unmapped = [group for group in groups if group not in PLAUSIBLE]
    if unmapped:
        raise SceneError(f"AudioSet groups {unmapped} have no entry in phase1_scene.PLAUSIBLE")
    return tuple(sorted({category for group in groups for category in PLAUSIBLE[group]}))


def on_labels(mask: Any, winner: Any, names: Sequence[str], plausible: Sequence[str]) -> float:
    """The share of an audio mask's pixels whose category (the winner map's) its source can be
    seen in; 0 for an empty mask, or a source none of whose categories the run segmented."""
    import numpy as np

    size = int(mask.sum())
    inside = [index for index, name in enumerate(names) if name in plausible]
    if not size or not inside:
        return 0.0
    return int(np.isin(winner[mask], inside).sum()) / size


def dilated(mask: Any, radius: int) -> Any:
    """The mask widened by a disc of ``radius`` pixels (row by row, with running sums)."""
    import numpy as np

    if radius <= 0:
        return mask
    height, width = mask.shape
    sums = np.pad(np.cumsum(np.pad(mask, radius).astype(np.int32), axis=1), ((0, 0), (1, 0)))
    columns = np.arange(width) + radius
    out = np.zeros_like(mask)
    for dy in range(-radius, radius + 1):
        half = int((radius * radius - dy * dy) ** 0.5)
        rows = sums[radius + dy : radius + dy + height]
        out |= (rows[:, columns + half + 1] - rows[:, columns - half]) > 0
    return out


def visibility(
    audio: Path,
    frames: Mapping[str, Any],
    names: Sequence[str],
    categories: Sequence[str],
    seconds: int,
) -> dict[str, Any]:
    """The audio mask's mean share on its source's categories, whether that puts the source on
    screen, and the share's Beta posterior, whose mean is the binding model's prior.

    The Beta(1, 1) prior takes one observation per second, since neighbouring frames are not
    independent.
    """
    shares = [
        on_labels(_mask(audio / frame), winner, names, categories)
        for frame, winner in frames.items()
        if (audio / frame).is_file()
    ]
    if not shares:
        raise SceneError(f"no audio mask among the sampled frames in {audio}")
    share = mean(shares)
    alpha, beta = 1 + seconds * share, 1 + seconds * (1 - share)
    return {
        "frames": len(shares),
        "share": round(share, 4),
        "on_screen": share >= ON_SCREEN,
        "alpha": round(alpha, 4),
        "beta": round(beta, 4),
        "mean": round(alpha / (alpha + beta), 4),
    }


def _click_frames(
    clip: Mapping[str, Any],
    value: str,
    audio: Path,
    winner_of: Callable[[str], Any],
    names: Sequence[str],
    categories: Sequence[str],
    *,
    root: Path,
) -> dict[str, dict[str, Any]]:
    """The sound's audio mask on each P2 frame, widened and written beside the clip's media.

    ``shown``: whether the frame puts the source on screen, from the mask as Sa2VA gave it;
    ``area`` is the widened mask's share, the chance a point lands in it.
    """
    from PIL import Image

    frames: dict[str, dict[str, Any]] = {}
    for index, frame in enumerate(clip["frames"]):
        still = Path(frame["still"])
        source = audio / still.name
        if not source.is_file():
            frames[str(index)] = {"mask": None, "shown": False, "area": 0.0}
            continue
        target = still.parent.parent / CLICK_MASKS / still.stem / f"{value}.png"
        (root / target).parent.mkdir(parents=True, exist_ok=True)
        mask = _mask(source)
        wide = dilated(mask, round(CLICK_TOLERANCE * mask.shape[1]))
        Image.fromarray(wide.astype("uint8") * 255).save(root / target)
        frames[str(index)] = {
            "mask": str(target),
            "shown": on_labels(mask, winner_of(still.name), names, categories) >= ON_SCREEN,
            "area": round(float(wide.mean()), 5),
        }
    return frames


def build_scene(
    clip: Mapping[str, Any],
    *,
    run: Path,
    groups_of: Callable[[str], tuple[str, ...]],
    root: Path,
    stride: int = STRIDE,
) -> dict[str, Any]:
    """One clip's scene part, without the setup-relative fields (:func:`add_scenes` adds them)."""
    key = str(clip["key"])
    visual = _visual_dirs(run, key)
    names = sorted(visual)
    maps: dict[str, Any] = {}

    def winner_of(frame: str) -> Any:
        if frame not in maps:
            maps[frame] = winners(visual, frame)
        return maps[frame]

    sampled = {frame: winner_of(frame) for frame in sampled_frames(visual, stride)}
    shares = composition(names, list(sampled.values()))
    labels = {entry["label"] for entry in clip["audioset_labels"]}
    seconds = max(1, round(int(clip["duration_ms"]) / 1000))
    sounds = {}
    for option in clip["options"]["A1s"]:
        label = option.get("audioset_label")
        if label not in labels:
            raise SceneError(
                f"{key}.A1s {option['value']}: audioset_label {label!r} is not one of the clip's labels"
            )
        groups = groups_of(label)
        categories = plausible_categories(groups)
        audio = run / "audio" / key / label
        if not audio.is_dir():
            raise SceneError(f"{key}: no audio masks for {label!r} at {audio}")
        sounds[option["value"]] = {
            "label": option["label"],
            "audioset_label": label,
            "groups": list(groups),
            "plausible": list(categories),
            "visibility": visibility(audio, sampled, names, categories, seconds),
            "click_frames": _click_frames(
                clip, option["value"], audio, winner_of, names, categories, root=root
            ),
        }
    return {
        "schema": SCENE_SCHEMA,
        "source": {
            "run": str(run),
            "stride": stride,
            "frames": len(sampled),
            "on_screen": ON_SCREEN,
            "click_tolerance": CLICK_TOLERANCE,
        },
        "composition": shares,
        "elements": elements_of(shares),
        "sounds": sounds,
    }


def openness(sky: float) -> str:
    """How the acousmatic caption hears the space: open, enclosed, or neutral between them."""
    return "open" if sky >= OPEN_SKY else "enclosed" if sky <= ENCLOSED_SKY else "neutral"


def add_scenes(
    setup: Mapping[str, Any], *, run: Path, ontology: Path, root: Path, stride: int = STRIDE
) -> dict[str, Any]:
    """A copy of the setup with every clip's scene part, its distinctive element and openness.

    An option's ``source_visible``, the researcher's judgement before visibility was measured, is
    dropped.
    """
    groups_of = ontology_groups(ontology)
    updated: dict[str, Any] = copy.deepcopy(dict(setup))
    for clip in updated["clips"]:
        for option in clip["options"]["A1s"]:
            option.pop("source_visible", None)
    scenes = [
        build_scene(clip, run=run, groups_of=groups_of, root=root, stride=stride) for clip in updated["clips"]
    ]
    means = {name: mean(scene["elements"][name] for scene in scenes) for name in ELEMENTS}
    for clip, scene in zip(updated["clips"], scenes, strict=True):
        lift = {
            name: round(scene["elements"][name] / means[name], 3) if means[name] else 0.0 for name in ELEMENTS
        }
        scene["lift"] = lift
        lifted = max(ELEMENTS, key=lambda name: lift[name])
        if lift[lifted] >= MIN_LIFT:
            scene["distinctive"], scene["distinctive_by"] = lifted, "lift"
        else:
            scene["distinctive"] = max(ELEMENTS, key=lambda name: scene["elements"][name])
            scene["distinctive_by"] = "largest"
        scene["openness"] = openness(scene["elements"][OPEN_ELEMENT])
        scene["setup_reference"] = {
            "element_means": {k: round(v, 4) for k, v in means.items()},
            "min_lift": MIN_LIFT,
            "sky": {"open": OPEN_SKY, "enclosed": ENCLOSED_SKY},
        }
        clip["scene"] = scene
    return updated


def scene_problems(clip: Mapping[str, Any], root: Path) -> list[str]:
    """Every reason the clip's scene part cannot drive its second caption."""
    where = f"{clip.get('key')}.scene"
    scene = clip.get("scene")
    if not isinstance(scene, Mapping) or scene.get("schema") != SCENE_SCHEMA:
        return [f"{where}: missing; run `python -m diff phase1 scene`"]
    problems = []
    if scene.get("distinctive") not in ELEMENTS or scene.get("openness") not in OPENNESS:
        problems.append(f"{where}: no distinctive element or openness")
    sounds = scene.get("sounds", {})
    for option in clip["options"]["A1s"]:
        sound = sounds.get(option["value"])
        if sound is None or sound.get("audioset_label") != option.get("audioset_label"):
            problems.append(f"{where}: A1s {option['value']} has no scene entry for its audioset_label")
            continue
        if "source_visible" in option:
            problems.append(
                f"{clip.get('key')}.A1s {option['value']}: source_visible is measured from the masks now; "
                "run `python -m diff phase1 scene` again"
            )
        frames = sound.get("click_frames", {})
        if set(frames) != {str(index) for index in range(len(clip["frames"]))}:
            problems.append(f"{where}: A1s {option['value']} lacks a click mask entry per P2 frame")
        for index, frame in frames.items():
            if frame.get("mask") and not (root / frame["mask"]).is_file():
                problems.append(f"{where}: A1s {option['value']} frame {index}: missing {frame['mask']}")
    return problems
