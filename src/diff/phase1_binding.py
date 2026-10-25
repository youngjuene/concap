"""Phase 1's second-caption mode: a Bayesian estimate of whether the noticed sound is bound to the scene.

Researcher's request of 2026-10-23: the second caption is rewritten scene-aware or
acousmatic, one mode per clip, by a Bayesian algorithm over the response scales,
the scene's visual mask proportions and the match between the Sa2VA audio mask
and the sound the participant noticed (its AudioSet label, grouped). The model
(``diff.phase1-binding/v3``) is provisional: its parameters are the
researcher's to approve, and participant sessions are refused until then.

**Revised 2026-10-24** after the design review, before any Trida run (v1 → v2):
- v1 carried E1/E3 into the tendency clip by clip, so a participant's rating
  level accumulated: in simulation, rating them 4.5 rather than 3 took the
  recommendation from 23% to 89% scene-aware by the twelfth clip. The
  modes are now decided once, when stage A ends (every first step precedes
  every second viewing), with E1/E3 centred on the participant's own mean and
  the tendency learnt from the other clips' points only.
- v1's visibility was the share of frames Sa2VA put the sound in a plausible
  category, which moved by more than 0.1 for 10 of the 27 seeable sounds
  under another reasonable rule. v2 made it the researcher's judgement per
  option (``source_visible``), drafted from that share.
- v1 assumed half of bound viewers place V1 on the source. V1 asks what
  first caught the eye, which salience drives, so :data:`DRAWN` is 0.2.

**v3, the same day** (researcher): visibility follows the masks strictly, with no
judgement: the share of the noticed sound's audio mask (its AudioSet label, the
audio caption's category match) on the visual categories its source can be seen
in, averaged over frames (:mod:`diff.phase1_scene`). Its Beta posterior mean is
``q``, and the source is on screen from ``phase1_scene.ON_SCREEN``.

**Latent state.** ``b``: whether this participant hears the sound they noticed
(P3 A1s) as belonging to the scene on screen. ``π``: their tendency to bind
sounds to the scene, uniform on a grid of :data:`GRID_SIZE` points, updated by
the attention evidence of their other clips. A clip's prior is
``logit P(b) = logit π + logit q``, where ``q`` is the measured visibility of
the noticed sound's source (the scene part's posterior mean, 0.08–0.92 for a
10 s clip): a sound whose source is never seen is rarely bound to what is seen.

**Evidence**, as likelihood ratios of bound to detached:

- *Attention* (P2 V1 and the visual mask proportions): whether the point falls
  inside the noticed sound's click mask on the frame it was placed on (the
  Sa2VA audio mask widened by the click tolerance, :mod:`diff.phase1_scene`).
  A detached viewer's point lands there by chance, at the mask's share ``a`` of
  the frame, so a region's size is the attention it takes for free (floored at
  :data:`AREA_FLOOR`). A bound viewer's eye is drawn to the source with
  probability :data:`DRAWN`, and lands by chance otherwise. A hit gives
  ``(DRAWN + (1 − DRAWN)·a) / a``, so a hit on a small source counts more than
  one on a large one, and a miss gives ``1 − DRAWN``. When the source is not
  on screen on that frame, or not on screen on average, the point says nothing
  (ratio 1).
- *Congruence* (P4 E1 and E3, whether the sounds belong to and fit the place on
  screen, 0–6; the sheet's axis-1 check items): ``log ratio = K·(C − C̄)/3``
  for their mean ``C`` and the participant's mean ``C̄`` over all their clips,
  so how someone uses the scale moves nothing, and a clip rated 3 points
  above their own level gives ratio 3.

**Decision.** P(b) sums over the grid. Its recommendation is scene-aware when
P(b) ≥ 0.5 and acousmatic otherwise, so that the caption matches how the sound
is heard (compatibility, in attention restoration theory). Each participant
follows a balanced plan, drawn from the order seed at entry: half the clips
take the recommended mode and half the other one. Following or countering is
then randomised within participant and independent of every rating, so each
mode is shown with probability 1/2 whatever P(b) is. The model's inputs enter
only through the recommendation.

**Moderators** (researcher, 2026-10-25, to pre-register). P(b) is fixed before
the outcome, but on the 12 c2 clips it mostly follows ``q``, so the analysis
takes it as two moderators of the mode's effect on P4→P9, both recorded:
- ``moderator_scene``: ``logit q``, the noticed sound's measured visibility;
- ``moderator_participant``: ``logit P(b) − logit P₀(b)``, what the
  participant's answers add, where ``P₀`` is the model with no answers (a flat
  tendency and no evidence from the clip). A flat tendency pulls P(b) toward
  1/2, so ``logit q`` alone would leave part of ``q`` in it; this is 0 exactly
  when the answers say nothing.
Their interactions with the mode separate the compatibility effect of binding
from that of visibility.

The restoration subscales (F, BF, E, PRS) and FA2/FA3 are recorded as each
clip's baseline, and never steer: they are the outcome's first measure.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any, Final

BINDING_POLICY: Final = "diff.phase1-binding/v3"
BINDING_PROVISIONAL: Final = True
ASSIGNMENT_POLICY: Final = "diff.phase1-assignment/balanced-v1"
GRID_SIZE: Final = 100
GRID: Final = tuple((index + 0.5) / GRID_SIZE for index in range(GRID_SIZE))
DRAWN: Final = 0.2
AREA_FLOOR: Final = 0.05
CONGRUENCE_ITEMS: Final = ("E1", "E3")
CONGRUENCE_K: Final = math.log(3)
CONGRUENCE_SPAN: Final = 3  # scale points above the participant's own mean that give ratio exp(K)
THRESHOLD: Final = 0.5
MODES: Final = ("scene", "acousmatic")
FOLLOW, COUNTER = "follow", "counter"
INSIDE: Final = 127
# The first measures of the outcome, by subscale (P4) and P5's familiarity and preference.
BASELINE: Final[dict[str, tuple[str, ...]]] = {
    "fascination": ("F1", "F2", "F3"),
    "being_away_from": ("BF1", "BF2", "BF3"),
    "extent_coherence": ("E1", "E2", "E3"),
    "being_away": ("P1", "P2"),
    "familiarity": ("FA2",),
    "preference": ("FA3",),
}


def assignment_plan(seed: int, count: int) -> list[str]:
    """Follow or counter per clip, in stage-A order: half each, the odd one out countering."""
    plan = [FOLLOW] * (count // 2) + [COUNTER] * (count - count // 2)
    random.Random(f"{ASSIGNMENT_POLICY}:{seed}").shuffle(plan)
    return plan


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def _prior_bound(pi: float, visibility: float) -> float:
    return _sigmoid(_logit(pi) + _logit(visibility))


def tendency(history: Sequence[tuple[float, float]]) -> list[float]:
    """π's grid weights given other clips, each as (visibility, log ratio of its attention evidence)."""
    weights = [1 / GRID_SIZE] * GRID_SIZE
    for visibility, log_ratio in history:
        ratio = math.exp(log_ratio)
        weights = [
            w * ((1 - (s := _prior_bound(pi, visibility))) + s * ratio)
            for w, pi in zip(weights, GRID, strict=True)
        ]
        total = sum(weights)
        weights = [w / total for w in weights]
    return weights


def summary(weights: Sequence[float]) -> dict[str, float]:
    """The grid posterior's mean and its central 90% interval."""
    cumulative, low, high = 0.0, GRID[0], GRID[-1]
    for pi, w in zip(GRID, weights, strict=True):
        before = cumulative
        cumulative += w
        if before < 0.05 <= cumulative:
            low = pi
        if before < 0.95 <= cumulative:
            high = pi
    return {
        "mean": round(sum(pi * w for pi, w in zip(GRID, weights, strict=True)), 4),
        "low": low,
        "high": high,
    }


def p_bound(weights: Sequence[float], visibility: float, log_ratio: float) -> float:
    """P(b = bound | this clip's evidence and the other clips')."""
    ratio = math.exp(log_ratio)
    bound = detached = 0.0
    for w, pi in zip(weights, GRID, strict=True):
        s = _prior_bound(pi, visibility)
        bound += w * s * ratio
        detached += w * (1 - s)
    return bound / (bound + detached)


def _inside(path: Path, x: float, y: float) -> bool:
    """Regen's point test (``dpo.regen.points``): the pixel nearest the normalised point."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        mask = np.asarray(image.convert("L")) > INSIDE
    height, width = mask.shape
    return bool(mask[min(height - 1, round(y * (height - 1))), min(width - 1, round(x * (width - 1)))])


def attention(sound: Mapping[str, Any], match: Mapping[str, Any] | None, root: Path) -> dict[str, Any]:
    """A point (V1, or V2 for the manipulation check) against the noticed sound's click mask on its frame.

    A source not on screen, on that frame or on average, says nothing.
    """
    if match is None:
        return {"status": "no point", "log_ratio": 0.0}
    frame = sound["click_frames"].get(str(match["frame"]), {})
    record: dict[str, Any] = {"frame": int(match["frame"]), "x": match["x"], "y": match["y"]}
    if not sound["visibility"]["on_screen"] or not frame.get("mask") or not frame.get("shown"):
        return {**record, "status": "source not seen", "log_ratio": 0.0}
    area = max(float(frame["area"]), AREA_FLOOR)
    hit = _inside(root / frame["mask"], float(match["x"]), float(match["y"]))
    ratio = (DRAWN + (1 - DRAWN) * area) / area if hit else 1 - DRAWN
    return {
        **record,
        "status": "hit" if hit else "miss",
        "area": area,
        "log_ratio": round(math.log(ratio), 6),
    }


def congruence_mean(answers: Mapping[str, Any]) -> float:
    return mean(float(answers[item]) for item in CONGRUENCE_ITEMS)


def congruence(answers: Mapping[str, Any], centre: float) -> dict[str, Any]:
    """E1/E3's mean against the participant's own mean over their clips, and its log ratio."""
    value = congruence_mean(answers)
    return {
        "items": list(CONGRUENCE_ITEMS),
        "mean": value,
        "centre": round(centre, 4),
        "log_ratio": round(CONGRUENCE_K * (value - centre) / CONGRUENCE_SPAN, 6),
    }


def baseline(answers: Mapping[str, Any]) -> dict[str, float]:
    return {name: round(mean(float(answers[item]) for item in items), 4) for name, items in BASELINE.items()}


def noticed_option(clip: Mapping[str, Any], value: str) -> dict[str, Any]:
    return dict(next(option for option in clip["options"]["A1s"] if option["value"] == value))


def decide_all(
    clips: Sequence[Mapping[str, Any]],
    answers: Sequence[Mapping[str, Any]],
    matches: Sequence[Mapping[str, Any] | None],
    plan: Sequence[str],
    *,
    root: Path,
) -> list[dict[str, Any]]:
    """The model's record for each of a participant's clips, once their whole stage A is in.

    The three sequences run in stage-A order: each clip, its P2–P5 answers merged and its
    P2 match record. ``plan`` is the balanced plan, one entry per clip.
    """
    if not len(clips) == len(answers) == len(matches) == len(plan):
        raise ValueError("one clip, answer set, match and plan entry each")
    if any(assignment not in (FOLLOW, COUNTER) for assignment in plan):
        raise ValueError(f"assignment must be {FOLLOW!r} or {COUNTER!r}")
    options = [noticed_option(clip, str(given["A1s"])) for clip, given in zip(clips, answers, strict=True)]
    sounds = [clip["scene"]["sounds"][option["value"]] for clip, option in zip(clips, options, strict=True)]
    visibility = [float(sound["visibility"]["mean"]) for sound in sounds]
    seen = [attention(sound, match, root) for sound, match in zip(sounds, matches, strict=True)]
    centre = mean(congruence_mean(given) for given in answers)
    records = []
    for index, given in enumerate(answers):
        others = [(visibility[j], seen[j]["log_ratio"]) for j in range(len(clips)) if j != index]
        weights = tendency(others)
        fit = congruence(given, centre)
        log_ratio = round(seen[index]["log_ratio"] + fit["log_ratio"], 6)
        probability = p_bound(weights, visibility[index], log_ratio)
        no_answers = p_bound(tendency(()), visibility[index], 0.0)
        recommended = MODES[0] if probability >= THRESHOLD else MODES[1]
        mode = recommended if plan[index] == FOLLOW else next(m for m in MODES if m != recommended)
        records.append(
            {
                "policy": BINDING_POLICY,
                "provisional": BINDING_PROVISIONAL,
                "noticed": {
                    "value": options[index]["value"],
                    "audioset_label": sounds[index]["audioset_label"],
                    "groups": sounds[index]["groups"],
                    "on_screen": sounds[index]["visibility"]["on_screen"],
                },
                "visibility": visibility[index],
                "measured_visibility": sounds[index]["visibility"]["share"],
                "attention": seen[index],
                "congruence": fit,
                "log_ratio": log_ratio,
                "tendency": {**summary(weights), "clips": len(others)},
                "p_bound": round(probability, 4),
                "moderator_scene": round(_logit(visibility[index]), 6),
                "moderator_participant": round(_logit(probability) - _logit(no_answers), 6),
                "recommended": recommended,
                "assignment": plan[index],
                "mode": mode,
                "baseline": baseline(given),
            }
        )
    return records
