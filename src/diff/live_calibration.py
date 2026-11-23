"""Where a live phase-2 session starts: each video's slider stops and the tuner system message.

Researcher's decisions:
- 2026-10-11: copy regen's sheet-handoff rule for the sliders. No explicit
  detail preference is collected, so both start at the neutral midpoint.
- 2026-10-12: the system message is our own "audio caption tuner" role, built
  once per session at the phase 1 → phase 2 handoff from the participant's
  phase-1 ratings and answers, and fixed through the watch. It replaces the
  guidance sentence the live prompt used to carry.
- **2026-11-23** (``docs/v4-diff/chain-spec.md``): the sliders are the method of adjustment with
  varied starts. Each slider starts at stop 0 on one video, 1 on one and 2 on one; the order is
  counterbalanced across participants and has nothing to do with their phase 1
  (:func:`start_orders`), so the pull of the starting stop is measured and removed in the
  analysis instead of being read as preference. The tuner is one fixed sentence for everyone; the
  heard families and the rating means below are still recorded, but reach no prompt.

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

import itertools
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Final

from diff.caption_sources import CATEGORIES_KO
from diff.rewrite_prompts import TUNER_PROVISIONAL, TUNER_SYSTEM, render_tuner_system
from diff.study_protocol import sha256_of

# v4 (2026-11-03): no scene profile. v5 (2026-11-23): counterbalanced starts, a fixed tuner.
CALIBRATION_POLICY: Final = "diff.live-calibration/v5"
CALIBRATION_SOURCE: Final = "counterbalanced start orders by participant number; a fixed tuner sentence"
START_TEXTURE: Final = 0.5  # the midpoint, for a session with no participant number (smoke)
START_CONTEXT: Final = 0.5
# The six orders of stops 0, 1, 2 over the three videos, as the page's slider positions.
ORDERS: Final = tuple(tuple(stop / 2 for stop in order) for order in itertools.permutations(range(3)))
NUMBER: Final = re.compile(r"P(\d+)")
RATING_ITEMS: Final = ("S1", "S2", "S3")
NOT_OBSERVED: Final = "없음"


@dataclass(frozen=True)
class Calibration:
    texture: float
    context: float
    system_message: str
    provisional: bool
    record: Mapping[str, Any] = field(default_factory=dict)
    # Each video's starting (texture, context), in watch order; the first is (texture, context).
    starts: tuple[tuple[float, float], ...] = ()

    def start(self, video_index: int) -> tuple[float, float]:
        """Where video ``video_index``'s sliders start (the session's first start when unset)."""
        if video_index < len(self.starts):
            return self.starts[video_index]
        return (self.texture, self.context)


def start_orders(participant: str | None, videos: int = 3) -> tuple[tuple[float, float], ...] | None:
    """The participant's (texture, context) start per video, or None without a participant number.

    Participant ``P{k+1}`` takes texture order ``k mod 6`` and context order ``(k + k div 6) mod 6``:
    every block of six participants starts each slider at each stop on each video equally often,
    and 36 participants meet all 36 pairs of orders once, so the two sliders' starts are unrelated.
    """
    found = NUMBER.fullmatch(participant or "")
    if found is None or videos != 3:
        return None
    k = int(found.group(1)) - 1
    texture, context = ORDERS[k % 6], ORDERS[(k + k // 6) % 6]
    return tuple(zip(texture, context, strict=True))


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
    participant: str | None = None,
) -> Calibration:
    """Each video's starting stops (:func:`start_orders`; the midpoint without a participant number)
    and the session's tuner system message; the phase-1 profile is recorded, not used."""
    heard: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    for observation in observations:
        heard.update(family for family, value in dict(observation.get("heard", {})).items() if value)
        labels.update(set(observation.get("labels", [])))
    means = rating_means(ratings)
    slots = profile_slots(heard, means)
    system_message = render_tuner_system({})
    starts = start_orders(participant) or ((START_TEXTURE, START_CONTEXT),) * 3
    record = {
        "policy": CALIBRATION_POLICY,
        "source": CALIBRATION_SOURCE,
        "sample_count": len(observations),
        "rated_clips": len(ratings),
        "start": {"texture": starts[0][0], "context": starts[0][1]},
        "starts": [{"texture": texture, "context": context} for texture, context in starts],
        "counterbalanced": start_orders(participant) is not None,
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
    }
    return Calibration(
        starts[0][0],
        starts[0][1],
        system_message,
        TUNER_PROVISIONAL,
        {**record, "sha256": sha256_of(record)},
        starts,
    )
