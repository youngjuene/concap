"""Where a live phase-2 session starts: the calibrated sliders and the participant guidance.

Researcher's decision of 2026-10-11: copy regen's rule for now, and redesign it
carefully later as a new policy version. Regen's sheet handoff
(``dpo.regen.sheet_protocol``, ``compile_profile({}, observations, "ko")``)
collects no explicit detail preference, so both sliders start at the neutral
midpoint, and the only participant-specific input is the phase-1 observations:
per short clip, the sound family behind the most-attended sound (P3 ``A1s``)
and the visual labels of the marked point (P2).

Regen writes this context into an English instruction for Gemma: weigh the
viewer's previously noticed categories only when equally relevant sources
compete, as limited observations, never over the current evidence or the
requested detail. Dream's prompt is Korean, so the same rule is written as
one Korean guidance sentence over the heard families.

One deliberate deviation: regen's visual labels are free text (often English)
and Dream echoes Latin letters into captions, which fails the script check. So
the labels are kept in the calibration record but not put in the prompt.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from diff.caption_sources import CATEGORIES_KO
from diff.study_protocol import sha256_of

CALIBRATION_POLICY: Final = "diff.live-calibration/v1"
CALIBRATION_SOURCE: Final = "regen sheet handoff: compile_profile({}, observations) — midpoint sliders"
START_TEXTURE: Final = 0.5
START_CONTEXT: Final = 0.5


@dataclass(frozen=True)
class Calibration:
    texture: float
    context: float
    guidance: str
    record: Mapping[str, Any] = field(default_factory=dict)


def guidance_sentence(heard: Mapping[str, int]) -> str:
    """The Korean form of regen's 'previously noticed categories' rule; empty with no observations."""
    named = [
        (CATEGORIES_KO[family], count) for family, count in sorted(heard.items()) if family in CATEGORIES_KO
    ]
    if not named:
        return ""
    listed = ", ".join(f"{name} {count}번" for name, count in named)
    return (
        f"참고로, 이 시청자는 앞 단계에서 {listed}에 가장 주의를 기울였습니다. "
        "비슷하게 중요한 소리가 함께 있을 때만 이를 참고하고, 원래 자막의 내용과 정해진 수준을 먼저 따르세요."
    )


def calibrate(observations: Sequence[Mapping[str, Any]]) -> Calibration:
    """Regen's sheet-handoff rule: midpoint sliders, guidance from the heard sound families."""
    heard: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    for observation in observations:
        heard.update(family for family, value in dict(observation.get("heard", {})).items() if value)
        labels.update(set(observation.get("labels", [])))
    guidance = guidance_sentence(heard)
    record = {
        "policy": CALIBRATION_POLICY,
        "source": CALIBRATION_SOURCE,
        "sample_count": len(observations),
        "start": {"texture": START_TEXTURE, "context": START_CONTEXT},
        "heard_observations": dict(sorted(heard.items())),
        "visual_observations": dict(sorted(labels.items())),
        "unmapped_families": sorted(family for family in heard if family not in CATEGORIES_KO),
        "guidance": guidance,
    }
    return Calibration(START_TEXTURE, START_CONTEXT, guidance, {**record, "sha256": sha256_of(record)})
