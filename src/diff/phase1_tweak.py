"""Phase 1's second caption: a participant's first-step answers as the tweak prompt's mode and slots.

Researcher's decisions: each clip is viewed twice (2026-10-18), and the second
viewing shows a caption Trida writes by tweaking the clip's raw first caption.
Since 2026-10-23 the tweak takes one mode per clip, scene-aware or acousmatic,
chosen by the binding model (:mod:`diff.phase1_binding`) from that participant's
P2–P4 answers and the clip's Sa2VA scene part (:mod:`diff.phase1_scene`), and
its wording varies with the scene. Since the design review of 2026-10-24 every
clip's mode is decided when stage A ends (:func:`tweak_steer_all`). The mode's
prompt is ``rewrite_prompts.TWEAK_PROMPTS[mode]``.

The slots, for one participant and clip:
- ``mode``: ``scene`` or ``acousmatic``;
- ``caption``: the raw first caption the participant saw;
- ``sounds``: the labels of the clip's mentioned R2 options, the sounds
  ``orig_text`` names. The prompt names them, so they are R2's key (the
  researcher's "key from the prompt");
- ``attended_sound``: the A1s answer's label (P3);
- scene-aware only: ``scene_elements``, the clip's distinctive element, then
  its largest other element; ``seen``, the scene's setting: the distinctive
  element (researcher, 2026-10-24: one setting per clip, so that every second
  caption can be drafted and reviewed before the study; it was the object the
  V1 point fell on, which gave 248 scene-aware prompts instead of 29);
  ``visible_sounds``, the A1s sounds whose source the masks put on screen
  (``phase1_scene.ON_SCREEN``), and the noticed one when the V1 point hit its
  click mask; ``offscreen``, a sentence placing the sounds not on screen "here,
  though unseen", or nothing when there are none. Sounds of the
  recording itself (:data:`RECORDING_ARTEFACTS`, microphone wind) are never
  placed: v2 spread them "in this scene" (hypothesis team, 2026-10-24);
- acousmatic only: ``space``, a sentence on how the sound spreads, from the
  scene's openness, and nothing when it is neither open nor enclosed.

FA1 (visited) is kept in the records and not used; the restoration subscales
and FA2/FA3 are recorded as the clip's baseline (``model.baseline``).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from diff.caption_sources import CaptionSource
from diff.job_plan import Job, tweak_job
from diff.output_checks import check_tweak, displayed, output_from_record
from diff.phase1_binding import FOLLOW, decide_all
from diff.rewrite_prompts import TWEAK_PROMPTS, TWEAK_PROVISIONAL, RenderedPrompt, render_tweak
from diff.study_protocol import TWEAK_MAX_CHARS

# v5 (2026-10-24): the scene-aware setting is the clip's distinctive element, not the V1 object.
TWEAK_POLICY: Final = "diff.phase1-tweak/v5"
STEP1_PAGES: Final = ("P2", "P3", "P4", "P5")
# The scene's elements (phase1_scene.ELEMENTS) in Korean.
ELEMENTS_KO: Final[dict[str, str]] = {
    "nature": "나무와 풀",
    "open": "하늘",
    "traffic": "도로와 차량",
    "built": "건물",
    "people": "사람들",
}
# How a sound spreads, by the scene's openness, as the acousmatic prompt's last request (each
# value starts with its own space); a scene neither open nor enclosed gets none.
SPACE_KO: Final[dict[str, str]] = {
    "open": " 소리가 넓게 트인 곳에서 멀리 퍼져 나가는 느낌도 살려 주세요.",
    "enclosed": " 소리가 둘러싸인 곳에서 가깝게 들리는 느낌도 살려 주세요.",
    "neutral": "",
}
# The scene-aware prompt's sentence for the sounds not on screen, one wording for everyone; it
# avoids "화면 밖", which would echo E1's "화면 속 장소" (it starts with its own space).
OFFSCREEN_KO: Final = (
    " 화면에 음원이 보이지 않는 소리({sounds})는 "
    "화면에는 보이지 않지만 이곳 어딘가에서 들려오는 소리로 쓰세요."
)
# AudioSet labels for sounds of the recording, not of the place: never placed in the scene.
RECORDING_ARTEFACTS: Final = frozenset({"Wind noise (microphone)"})
NOT_SEEN: Final = "없음"


class TweakError(ValueError):
    """A clip's first step is not complete enough to write its second caption."""


def step_answers(records: Mapping[str, Mapping[str, Any]], clip_key: str) -> dict[str, Any]:
    missing = [page for page in STEP1_PAGES if page not in records]
    if missing:
        raise TweakError(f"{clip_key}: the first step has no {missing}")
    answers: dict[str, Any] = {}
    for page in STEP1_PAGES:
        answers.update(records[page]["answers"])
    return answers


def scene_slots(clip: Mapping[str, Any], mode: str, model: Mapping[str, Any]) -> dict[str, str]:
    """The mode's own slots, from the clip's scene part and whether the V1 point hit the noticed sound.

    A noticed sound whose click mask the point hit counts as visible, whatever its judgement:
    the participant found its source on screen.
    """
    scene = clip["scene"]
    if mode == "acousmatic":
        return {"space": SPACE_KO[scene["openness"]]}
    distinctive = scene["distinctive"]
    others = sorted(
        (name for name in scene["elements"] if name != distinctive), key=lambda n: -scene["elements"][n]
    )
    hit = model["noticed"]["value"] if model["attention"]["status"] == "hit" else None
    visible = [
        option["label"]
        for option in clip["options"]["A1s"]
        if option["value"] == hit or on_screen(clip, option)
    ]
    unseen = [
        option["label"]
        for option in clip["options"]["A1s"]
        if option["value"] != hit
        and not on_screen(clip, option)
        and option["audioset_label"] not in RECORDING_ARTEFACTS
    ]
    return {
        "scene_elements": ", ".join(ELEMENTS_KO[name] for name in (distinctive, others[0])),
        "seen": ELEMENTS_KO[distinctive],
        "visible_sounds": ", ".join(dict.fromkeys(visible)) or NOT_SEEN,
        "offscreen": OFFSCREEN_KO.format(sounds=", ".join(dict.fromkeys(unseen))) if unseen else "",
    }


def caption_variant(clip: Mapping[str, Any], mode: str) -> dict[str, Any]:
    """The caption's variant for analysis: the mode and the scene characteristic it draws on."""
    scene = clip["scene"]
    if mode == "acousmatic":
        return {"type": f"acousmatic/{scene['openness']}"}
    return {"type": f"scene/{scene['distinctive']}"}


def tweak_steer_all(
    clips: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Mapping[str, Any]]],
    plan: Sequence[str],
    *,
    root: Path,
) -> list[tuple[dict[str, str], dict[str, Any]]]:
    """Each clip's prompt slots and binding-model record, from the participant's whole stage A.

    ``clips`` and ``records`` (each clip's first-step records, by page) run in stage-A order.
    """
    answers = [step_answers(step, str(clip["key"])) for clip, step in zip(clips, records, strict=True)]
    matches = [(step["P2"].get("matches") or [None])[0] for step in records]
    models = decide_all(clips, answers, matches, plan, root=root)
    steered = []
    for clip, given, model in zip(clips, answers, models, strict=True):
        slots = tweak_slots(clip, model["mode"], given.get("A1s"), model)
        model["caption"] = caption_variant(clip, model["mode"])
        steered.append((slots, model))
    return steered


def tweak_slots(
    clip: Mapping[str, Any], mode: str, attended: str | None, model: Mapping[str, Any]
) -> dict[str, str]:
    """One clip's prompt slots in ``mode``, from the attended sound (A1s) and the binding model's record."""
    sounds = {option["value"]: option["label"] for option in clip["options"]["A1s"]}
    mentioned = [option["label"] for option in clip["options"]["R2"] if option.get("mentioned")]
    return {
        "mode": mode,
        "caption": clip["caption"]["text"],
        "sounds": ", ".join(mentioned),
        "attended_sound": sounds.get(str(attended), NOT_SEEN),
        **scene_slots(clip, mode, model),
    }


def on_screen(clip: Mapping[str, Any], option: Mapping[str, Any]) -> bool:
    """Whether the masks put an A1s option's source on screen (its scene part's measured share)."""
    return bool(clip["scene"]["sounds"][option["value"]]["visibility"]["on_screen"])


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


def template_record(mode: str) -> dict[str, Any]:
    template = TWEAK_PROMPTS[mode]
    return {
        "id": template.template_id,
        "version": template.version,
        "mode": mode,
        "sha256": template.sha256,
        "provisional": TWEAK_PROVISIONAL,
    }


def tweak_request(
    clip: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    purpose: str,
    *,
    root: Path,
    assignment: str = FOLLOW,
) -> tuple[CaptionSource, Job, dict[str, Any]]:
    """The source, the job and its steering record for one clip decided alone (no other clips)."""
    ((slots, _),) = tweak_steer_all([clip], [records], [assignment], root=root)
    return tweak_job_for(clip, slots, purpose)


def tweak_job_for(
    clip: Mapping[str, Any], slots: Mapping[str, str], purpose: str
) -> tuple[CaptionSource, Job, dict[str, Any]]:
    """The source, the job and its steering record from slots already computed (and stored)."""
    prompt: RenderedPrompt = render_tweak(slots)
    source = tweak_source(clip)
    job = tweak_job(source, prompt, purpose)
    record = {
        "policy": TWEAK_POLICY,
        "template": template_record(slots["mode"]),
        "slots": dict(slots),
        "intended_r2_key": intended_r2_key(clip),
        "job_id": job.job_id,
    }
    return source, job, record


def named_in(text: str, label: str) -> bool:
    """Whether a caption seems to name the sound an option's label stands for: any word of the label
    but "소리", by its first two syllables. Rough: it misses "새가" for "새소리"."""
    return any(word[:2] in text for word in re.findall(r"[가-힣]{2,}", label) if word != "소리")


def intended_r2_key(clip: Mapping[str, Any]) -> list[str]:
    """The R2 options the prompt names (the mentioned ones): R2's key, intended, not checked."""
    return [option["value"] for option in clip["options"]["R2"] if option.get("mentioned")]


def check_record(source: CaptionSource, job: Job, result: Mapping[str, Any]) -> dict[str, Any]:
    """The tweak check's record for one completed call (``LiveService(tweak_check=...)``), as shown."""
    output = displayed(output_from_record(result["output"]))
    return check_tweak(source, output, max_chars=TWEAK_MAX_CHARS).record()
