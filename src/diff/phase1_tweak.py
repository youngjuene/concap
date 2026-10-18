"""Phase 1's second caption: a participant's first-step answers as the tweak prompt's slots.

Researcher's decisions of 2026-10-18: each clip is viewed twice. The second
viewing shows a caption Trida writes by tweaking the clip's raw first caption,
steered by that participant's answers to the clip's first step (P2–P5). The
answers are processed into the slots of ``rewrite_prompts.TWEAK_P1``, which
names each steering direction and its degree in percent.

The slots, for one participant and clip:
- ``caption``: the raw first caption the participant saw;
- ``sounds``: the labels of the clip's mentioned R2 options, the sounds
  ``orig_text`` names. The prompt names them, so they are R2's key (the
  researcher's "key from the prompt");
- ``attended_sound``: the A1s answer's label (P3);
- ``attended_object``: the object the V1 point fell on (P2), in Korean, or
  "없음" when it fell on no annotated object;
- four degrees, ``round(mean / 6 × 100)`` of the 0–6 ratings behind each
  direction (:data:`DIRECTIONS`). The degree *reflects* the answers: a higher
  rating steers more strongly that way. Whether it should compensate instead,
  ``(6 − mean) / 6``, is the researcher's call (:data:`DEGREE_RULE`).

FA1 (visited) and FA3 (preference) are kept in the records and not used.
"""

from __future__ import annotations

from collections.abc import Mapping
from statistics import mean
from typing import Any, Final

from diff.caption_sources import CaptionSource
from diff.job_plan import Job, tweak_job
from diff.output_checks import check_tweak, output_from_record
from diff.rewrite_prompts import TWEAK_P1, TWEAK_PROVISIONAL, RenderedPrompt, render_tweak
from diff.study_protocol import TWEAK_MAX_CHARS

TWEAK_POLICY: Final = "diff.phase1-tweak/v1"
# The step-1 ratings behind each direction's degree (P4 restorativeness, P5 familiarity).
DIRECTIONS: Final[dict[str, tuple[str, ...]]] = {
    "fascination": ("F1", "F2", "F3"),
    "being_away": ("BF1", "BF2", "BF3", "P1", "P2"),
    "coherence": ("E1", "E2", "E3"),
    "familiarity": ("FA2",),
}
DEGREE_RULE: Final = "reflect"
SCALE_MAX: Final = 6
STEP1_PAGES: Final = ("P2", "P3", "P4", "P5")
# The mask objects of regen's c2 frames, in Korean, so no Latin letters reach the prompt.
OBJECTS_KO: Final[dict[str, str]] = {
    "building": "건물",
    "person": "사람",
    "road": "도로",
    "road_transport": "차량",
    "sidewalk": "인도",
    "sky": "하늘",
    "terrain": "땅",
    "vegetation": "나무와 풀",
}
NOT_SEEN: Final = "없음"


class TweakError(ValueError):
    """A clip's first step is not complete enough to write its second caption."""


def degree(values: list[float]) -> int:
    """A direction's degree in percent from its 0–6 ratings."""
    shown = mean(values) / SCALE_MAX
    return round((shown if DEGREE_RULE == "reflect" else 1 - shown) * 100)


def tweak_slots(clip: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """The prompt's slots from one clip's setup and its first-step records (by page)."""
    missing = [page for page in STEP1_PAGES if page not in records]
    if missing:
        raise TweakError(f"{clip['key']}: the first step has no {missing}")
    answers: dict[str, Any] = {}
    for page in STEP1_PAGES:
        answers.update(records[page]["answers"])
    sounds = {option["value"]: option["label"] for option in clip["options"]["A1s"]}
    mentioned = [option["label"] for option in clip["options"]["R2"] if option.get("mentioned")]
    objects = [match.get("object_id") for match in records["P2"].get("matches", [])]
    first = next((OBJECTS_KO.get(str(item), NOT_SEEN) for item in objects if item), NOT_SEEN)
    slots = {
        "caption": clip["caption"]["text"],
        "sounds": ", ".join(mentioned),
        "attended_sound": sounds.get(answers.get("A1s"), NOT_SEEN),
        "attended_object": first,
    }
    for direction, items in DIRECTIONS.items():
        slots[direction] = str(degree([float(answers[item]) for item in items]))
    return slots


def tweak_source(clip: Mapping[str, Any]) -> CaptionSource:
    """The clip's raw first caption as a one-cue source, with no evidence."""
    caption = clip["caption"]
    return CaptionSource(
        phase=1,
        unit_key=str(clip["key"]),
        cue_index=0,
        start_ms=int(caption["start_ms"]),
        end_ms=int(caption["end_ms"]),
        text=str(caption["text"]),
        events=(),
        provenance_text=str(caption.get("provenance", "")),
        provenance_evidence="none",
        status="reviewed",
    )


def tweak_request(
    clip: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]], purpose: str
) -> tuple[CaptionSource, Job, dict[str, Any]]:
    """The source, the job and the record of what steered it, for one participant and clip."""
    return tweak_job_for(clip, tweak_slots(clip, records), purpose)


def tweak_job_for(
    clip: Mapping[str, Any], slots: Mapping[str, str], purpose: str
) -> tuple[CaptionSource, Job, dict[str, Any]]:
    """The source, the job and its steering record from slots already computed (and stored)."""
    prompt: RenderedPrompt = render_tweak(slots)
    source = tweak_source(clip)
    job = tweak_job(source, prompt, purpose)
    record = {
        "policy": TWEAK_POLICY,
        "template": {
            "id": TWEAK_P1.template_id,
            "version": TWEAK_P1.version,
            "sha256": TWEAK_P1.sha256,
            "provisional": TWEAK_PROVISIONAL,
        },
        "degree_rule": DEGREE_RULE,
        "slots": dict(slots),
        "intended_r2_key": intended_r2_key(clip),
        "job_id": job.job_id,
    }
    return source, job, record


def intended_r2_key(clip: Mapping[str, Any]) -> list[str]:
    """The R2 options the prompt names (the mentioned ones): R2's key, intended, not checked."""
    return [option["value"] for option in clip["options"]["R2"] if option.get("mentioned")]


def check_record(source: CaptionSource, job: Job, result: Mapping[str, Any]) -> dict[str, Any]:
    """The tweak check's record for one completed call (``LiveService(tweak_check=...)``)."""
    return check_tweak(source, output_from_record(result["output"]), max_chars=TWEAK_MAX_CHARS).record()
