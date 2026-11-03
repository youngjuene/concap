"""Phase 2's scene parts and caption mode: the long videos' Sa2VA masks per cue window, which set each
cue's live caption mode.

Researcher's decisions of 2026-10-28 (gate 5, ``docs/v4-diff/plan.md``), as changed on 2026-11-03:

- **The mode comes from the masks alone.** Each cue's mode is the phase-1 model's recommendation with a
  flat tendency: ``logit P(b) = logit q``, where ``q`` is the measured visibility of the cue's first
  sound (the starting captions name the most prominent sound first). Scene-aware when
  ``P(b) ≥ phase1_binding.THRESHOLD``, acousmatic otherwise (:func:`cue_steer`). Until 2026-11-03 the
  tendency was learnt from each participant's stage-A attention evidence (D1 (c)); it recovered the
  simulated tendency at r = 0.17 and gained nothing over the masks
  (``artifacts/diff/phase2-scene/d1-sim``). What each participant perceives is now asked outright
  after the watch (:mod:`diff.source_check`), and the watch has no mode control.
- **D2, which mask a sound uses.** The audio masks of 2026-10-22 were prompted with the English
  phrases (``en_afnext``); the participant reads ``ko_final``, edited and partly re-heard. Each sound
  and phrase gets an AudioSet label (``scripts/diff_phase2_keys.py``, Claude, from the classes
  :data:`~diff.phase1_scene.PLAUSIBLE` covers), and a sound uses the mask of a phrase in its own
  window with the same label (``label``); with none it has no mask (``none``), so its source is never
  on screen. Matching on a shared group instead paired a horn with a tyre screech, so it is not
  used. The keys are the researcher's to approve (``data/phase2/scene-keys.csv``).
- **D3, the thresholds.** Phase 1's :data:`~diff.phase1_scene.ON_SCREEN`,
  :data:`~diff.phase1_scene.PLAUSIBLE` and :data:`~diff.phase1_scene.MIN_LIFT`, unchanged, and the
  same frame stride.

A cue's scene part is phase 1's, per 10 s window: the visual categories' shares over the window's
frames (masks from the same windows as the audio masks, so frame numbers line up), the elements,
the distinctive element by lift over the 90 windows' mean, and each sound's visibility. Until the
researcher approves the keys, the scene parts and the scene-aware live prompts
(``rewrite_prompts.LIVE_SCENE_CELL_PROMPTS``), participant sessions are refused.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from diff.phase1_binding import MODES, THRESHOLD, p_bound, tendency
from diff.phase1_scene import (
    ON_SCREEN,
    STRIDE,
    SceneError,
    category_id,
    composition,
    distinguish,
    elements_of,
    ontology_groups,
    plausible_categories,
    visibility,
    winners,
)
from diff.phase1_tweak import ELEMENTS_KO, RECORDING_ARTEFACTS
from diff.rewrite_prompts import NOT_SEEN

SCENE_SCHEMA: Final = "diff.phase2-scene/v1"
STEER_POLICY: Final = "diff.phase2-mode/masks-v2"  # v2 (2026-11-03): masks alone, no participant profile
PHASE2_SCENE_PROVISIONAL: Final = True
SCENE_FILE: Final = Path("data/phase2/phase2-scene.json")
KEYS_FILE: Final = Path("data/phase2/scene-keys.csv")
MATCHES: Final = ("label", "none")
WINDOWS: Final = "visual_jobs.csv"  # the run's manifest, beside its masks folder


def read_keys(path: Path) -> dict[tuple[str, int], list[dict[str, str]]]:
    """The keys table by (video, window), each window's sounds in caption order."""
    keys: dict[tuple[str, int], list[dict[str, str]]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["mask_match"] not in MATCHES:
                raise SceneError(f"{path}: {row['video']} w{row['window']}: mask_match {row['mask_match']!r}")
            if (row["mask_match"] == "none") != (not row["mask_dir"]):
                raise SceneError(
                    f"{path}: {row['video']} w{row['window']} {row['sound']!r}: mask_dir disagrees"
                )
            keys.setdefault((row["video"], int(row["window"])), []).append(row)
    for rows in keys.values():
        rows.sort(key=lambda row: int(row["sound_index"]))
    return keys


def read_windows(path: Path) -> dict[tuple[str, int], tuple[int, int]]:
    """Each (video, window)'s frames as the Sa2VA run decoded them, ``[start, end)``.

    The videos' frame rates vary a little (59.94 to 60), and the run numbered frames by decoding at
    each video's own rate, so a cue's milliseconds do not give its frames.
    """
    if not path.is_file():
        raise SceneError(f"no window manifest at {path} (sa2va_captions.py --branch visual writes it)")
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            (row["video"], int(row["window"])): (int(row["frame_start"]), int(row["frame_end_exclusive"]))
            for row in csv.DictReader(handle)
        }


def cue_sounds(text: str) -> list[str]:
    """A starting caption's sounds, as the keys table splits them."""
    return [part.strip().rstrip(".").strip() for part in text.replace("\n", ",").split(",") if part.strip()]


def _window_frames(visual: Mapping[str, Path], lo: int, hi: int, stride: int) -> list[str]:
    names = [{p.name for p in folder.glob("*.png") if lo <= int(p.stem) < hi} for folder in visual.values()]
    common = sorted(set.intersection(*names)) if names else []
    if not common:
        raise SceneError(f"no frame in [{lo}, {hi}) has every category mask in {sorted(visual)}")
    return common[::stride]


def build_cue(
    source: Any,
    keys: Sequence[Mapping[str, str]],
    *,
    frames: tuple[int, int],
    run: Path,
    groups_of: Callable[[str], tuple[str, ...]],
    stride: int = STRIDE,
) -> dict[str, Any]:
    """One cue's scene part, without the set-relative fields (:func:`build_scenes` adds them)."""
    video = str(source.unit_key)
    lo, hi = frames
    folder = run / "visual" / video
    if not folder.is_dir():
        raise SceneError(f"{video}: no visual masks at {folder}")
    visual = {category_id(entry.name): entry for entry in sorted(folder.iterdir()) if entry.is_dir()}
    names = sorted(visual)
    sampled = {frame: winners(visual, frame) for frame in _window_frames(visual, lo, hi, stride)}
    shares = composition(names, list(sampled.values()))
    seconds = max(1, round((source.end_ms - source.start_ms) / 1000))
    if [row["sound"] for row in keys] != cue_sounds(source.text):
        raise SceneError(f"{video} cue {source.cue_index}: the keys' sounds are not the caption's")
    sounds = []
    for row in keys:
        label = row["audioset_label"]
        groups = groups_of(label)
        categories = plausible_categories(groups)
        if row["mask_dir"]:
            audio = run / "audio" / video / row["mask_dir"]
            if not audio.is_dir():
                raise SceneError(f"{video}: no audio masks at {audio}")
            seen = visibility(audio, sampled, names, categories, seconds)
        else:  # no mask: no source in view, the prior's mean at a share of 0
            seen = {
                "frames": 0,
                "share": 0.0,
                "on_screen": False,
                "alpha": 1.0,
                "beta": float(1 + seconds),
                "mean": round(1 / (2 + seconds), 4),
            }
        sounds.append(
            {
                "sound": row["sound"],
                "audioset_label": label,
                "groups": list(groups),
                "plausible": list(categories),
                "mask_match": row["mask_match"],
                "mask_dir": row["mask_dir"] or None,
                "visibility": seen,
            }
        )
    return {
        "video": video,
        "cue_index": source.cue_index,
        "frames": [lo, hi],
        "composition": shares,
        "elements": elements_of(shares),
        "sounds": sounds,
    }


def build_scenes(
    sources: Sequence[Any], *, keys: Path, run: Path, ontology: Path, stride: int = STRIDE
) -> dict[str, Any]:
    """Every phase-2 cue's scene part, keyed by its source ID, with the distinctive element by lift."""
    table = read_keys(keys)
    windows = read_windows(run.parent / WINDOWS)
    groups_of = ontology_groups(ontology)
    cues = [source for source in sources if source.phase == 2]
    scenes = []
    for source in cues:
        rows = table.get((str(source.unit_key), source.cue_index))
        if rows is None:
            raise SceneError(f"{source.unit_key} cue {source.cue_index}: no keys in {keys}")
        frames = windows.get((str(source.unit_key), source.cue_index))
        if frames is None:
            raise SceneError(f"{source.unit_key} cue {source.cue_index}: not in {run.parent / WINDOWS}")
        scenes.append(build_cue(source, rows, frames=frames, run=run, groups_of=groups_of, stride=stride))
    means = distinguish(scenes)
    return {
        "schema": SCENE_SCHEMA,
        "source": {
            "run": str(run),
            "keys": str(keys),
            "stride": stride,
            "on_screen": ON_SCREEN,
            "element_means": {name: round(value, 4) for name, value in means.items()},
        },
        "cues": {source.source_id: scene for source, scene in zip(cues, scenes, strict=True)},
    }


def read_scenes(path: Path) -> dict[str, Any]:
    scenes: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if scenes.get("schema") != SCENE_SCHEMA:
        raise SceneError(f"{path}: not a {SCENE_SCHEMA} file; run `python -m diff phase2 scene`")
    return scenes


def scene_problems(scenes: Mapping[str, Any] | None, sources: Sequence[Any]) -> list[str]:
    """Every reason the scene file cannot steer these cues' live captions."""
    if scenes is None:
        return ["phase 2 has no scene parts: pass --phase2-scene (run `python -m diff phase2 scene`)"]
    problems = []
    cues = scenes.get("cues", {})
    for source in sources:
        if source.phase != 2:
            continue
        where = f"{source.unit_key} cue {source.cue_index}"
        scene = cues.get(source.source_id)
        if scene is None:
            problems.append(f"{where}: no scene part")
            continue
        if [sound["sound"] for sound in scene["sounds"]] != cue_sounds(source.text):
            problems.append(f"{where}: the scene part's sounds are not the starting caption's")
        if scene.get("distinctive") not in ELEMENTS_KO:
            problems.append(f"{where}: no distinctive element")
        if scene["sounds"] and scene["sounds"][0]["audioset_label"] in RECORDING_ARTEFACTS:
            # The cue's mode is read off its first sound (cue_steer), and a sound of the recording is
            # never placed, so it would set the mode for a sound the prompt leaves out.
            problems.append(f"{where}: the first sound is a sound of the recording itself")
    return problems


def scene_slots(scene: Mapping[str, Any]) -> dict[str, str]:
    """The scene-aware prompt's slots: the setting, and the sounds whose source is or is not in view.

    Sounds of the recording itself are in neither list, as in phase 1.
    """
    placed = [sound for sound in scene["sounds"] if sound["audioset_label"] not in RECORDING_ARTEFACTS]
    visible = [sound["sound"] for sound in placed if sound["visibility"]["on_screen"]]
    unseen = [sound["sound"] for sound in placed if not sound["visibility"]["on_screen"]]
    return {
        "seen": ELEMENTS_KO[scene["distinctive"]],
        "visible_sounds": ", ".join(dict.fromkeys(visible)) or NOT_SEEN,
        "unseen_sounds": ", ".join(dict.fromkeys(unseen)) or NOT_SEEN,
    }


def cue_steer(scene: Mapping[str, Any] | None) -> dict[str, Any]:
    """The cue's mode, from the first sound's visibility, with the scene-aware slots; acousmatic with
    no scene part (a smoke run without masks)."""
    if scene is None:
        return {"policy": STEER_POLICY, "mode": MODES[1], "reason": "no scene part", "slots": {}}
    lead = scene["sounds"][0]
    q = float(lead["visibility"]["mean"])
    probability = p_bound(tendency([]), q, 0.0)
    mode = MODES[0] if probability >= THRESHOLD else MODES[1]
    return {
        "policy": STEER_POLICY,
        "mode": mode,
        "lead": lead["sound"],
        "visibility": q,
        "p_bound": round(probability, 4),
        "slots": scene_slots(scene) if mode == MODES[0] else {},
    }
