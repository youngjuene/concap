"""Where a live phase-2 session starts: the calibrated sliders and the tuner system message.

Researcher's decisions:
- 2026-10-11: copy regen's sheet-handoff rule for the sliders. No explicit
  detail preference is collected, so both start at the neutral midpoint.
- 2026-10-12: the system message is our own "audio caption tuner" role, built
  once per session at the phase 1 → phase 2 handoff from the participant's
  phase-1 ratings and answers, and fixed through the watch. It replaces the
  guidance sentence the live prompt used to carry.

The profile comes from phase-1 records:
- per short clip, the sound family behind the most-attended sound (P3 ``A1s``,
  as regen's rule);
- the caption ratings S1–S3 (0–6: sounds represented, fit with the scene,
  effect on understanding).

Regen's visual labels are kept in the record but never in the prompt, because
free-text labels (often English) invite Latin letters into Korean captions.
The slots are filled into ``rewrite_prompts.TUNER_SYSTEM``, whose wording the
researcher writes; until then it is provisional and participant sessions are
refused (see ``python -m diff live serve``).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Final

from diff.caption_sources import CATEGORIES_KO
from diff.rewrite_prompts import TUNER_PROVISIONAL, TUNER_SYSTEM, render_tuner_system
from diff.study_protocol import sha256_of

CALIBRATION_POLICY: Final = "diff.live-calibration/v3"  # v3 (2026-10-28): the scene profile
CALIBRATION_SOURCE: Final = "regen sheet handoff for the sliders; phase-1 ratings and answers for the tuner"
START_TEXTURE: Final = 0.5
START_CONTEXT: Final = 0.5
RATING_ITEMS: Final = ("S1", "S2", "S3")
NOT_OBSERVED: Final = "없음"


@dataclass(frozen=True)
class Calibration:
    texture: float
    context: float
    system_message: str
    provisional: bool
    record: Mapping[str, Any] = field(default_factory=dict)


def heard_families_text(heard: Mapping[str, int]) -> str:
    named = [
        (CATEGORIES_KO[family], count) for family, count in sorted(heard.items()) if family in CATEGORIES_KO
    ]
    return ", ".join(f"{name} {count}번" for name, count in named) or NOT_OBSERVED


def rating_means(ratings: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    """The mean of each item over the phase-1 clips; ``None`` when no clip rated it."""
    means: dict[str, float | None] = {}
    for item in RATING_ITEMS:
        values = [
            float(row[item])
            for row in ratings
            if isinstance(row.get(item), (int, float)) and not isinstance(row.get(item), bool)
        ]
        means[item] = round(mean(values), 2) if values else None
    return means


def profile_slots(heard: Mapping[str, int], means: Mapping[str, float | None]) -> dict[str, str]:
    """The tuner's named slots, all Korean or numeric text."""

    def shown(value: float | None) -> str:
        return NOT_OBSERVED if value is None else f"{value:g}"

    return {
        "heard_families": heard_families_text(heard),
        "s1": shown(means.get("S1")),
        "s2": shown(means.get("S2")),
        "s3": shown(means.get("S3")),
    }


def calibrate(
    observations: Sequence[Mapping[str, Any]],
    ratings: Sequence[Mapping[str, Any]] = (),
    scene_profile: Mapping[str, Any] | None = None,
) -> Calibration:
    """Midpoint sliders and the session's tuner system message from phase-1 answers and ratings, with
    the participant's scene profile (``phase2_scene.scene_profile``; ``None`` without phase 1), which
    sets each cue's mode through the watch."""
    heard: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    for observation in observations:
        heard.update(family for family, value in dict(observation.get("heard", {})).items() if value)
        labels.update(set(observation.get("labels", [])))
    means = rating_means(ratings)
    slots = profile_slots(heard, means)
    system_message = render_tuner_system(slots)
    record = {
        "policy": CALIBRATION_POLICY,
        "source": CALIBRATION_SOURCE,
        "sample_count": len(observations),
        "rated_clips": len(ratings),
        "start": {"texture": START_TEXTURE, "context": START_CONTEXT},
        "heard_observations": dict(sorted(heard.items())),
        "visual_observations": dict(sorted(labels.items())),
        "unmapped_families": sorted(family for family in heard if family not in CATEGORIES_KO),
        "caption_rating_means": means,
        "tuner": {
            "template": TUNER_SYSTEM.template_id,
            "version": TUNER_SYSTEM.version,
            "sha256": TUNER_SYSTEM.sha256,
            "provisional": TUNER_PROVISIONAL,
        },
        "slots": slots,
        "system_message": system_message,
        "scene_profile": dict(scene_profile) if scene_profile is not None else None,
    }
    return Calibration(
        START_TEXTURE,
        START_CONTEXT,
        system_message,
        TUNER_PROVISIONAL,
        {**record, "sha256": sha256_of(record)},
    )
