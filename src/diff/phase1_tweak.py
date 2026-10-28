"""Phase 1's second caption: a participant's first-step answers as the tweak prompt's mode and slots.

Researcher's decisions: each clip is viewed twice (2026-10-18), and the second
viewing shows a caption Trida writes by tweaking the clip's raw first caption.
Since 2026-10-23 the tweak takes one mode per clip, scene-aware or acousmatic,
chosen by the binding model (:mod:`diff.phase1_binding`) from that participant's
P2–P4 answers and the clip's Sa2VA scene part (:mod:`diff.phase1_scene`). Since
the design review of 2026-10-24 every clip's mode is decided when stage A ends
(:func:`tweak_steer_all`). The mode's prompt is ``rewrite_prompts.TWEAK_PROMPTS[mode]``.

**Responsive, generated live** (researcher, 2026-10-26): the rewrite steers the
audio caption from what the scene holds to what this participant perceived and
preferred, in the attention-restorative direction; it is not a scene-descriptive
rewrite. Two answers steer the text, in both modes alike: the noticed sound (A1s),
which the caption centres, and the restorative appraisal (P4), whose emphasis
*reflects* the quality the participant perceived most in this clip
(:func:`emphases`). Each caption is written when stage A ends and shown once it
passes the gate (:mod:`diff.phase1_usable`); the researcher no longer approves
every caption before the study (that design, 2026-10-24, is retired). The writer is
Claude through the API since 2026-10-27 (:mod:`diff.claude_writer`), after the local
models failed the delivery benches: Trida delivered too few sentence rewrites, and
Qwen3.5-9B, the writer for a day, drifted into other scripts and dropped sounds. A caption
the gate refuses is written again at the next rung of ``rewrite_prompts.TWEAK_LADDER``
(a greedy call repeats itself; a first version varied Trida's τ, which reproduced the
same text in 61 of 86 retries).
Phase 1 tests the mode (follow or counter); the emphasis is applied the same way under
either mode, and testing it is phase 2's role, under the same rule.

The slots, for one participant and clip:
- ``mode``: ``scene`` or ``acousmatic``;
- ``caption``: the raw first caption the participant saw;
- ``sounds``: the labels of the clip's mentioned R2 options, the sounds
  ``orig_text`` names. The prompt names them, so they are R2's key (the
  researcher's "key from the prompt");
- ``attended_sound``: the A1s answer's label (P3);
- ``emphasis``: the sentence of :data:`EMPHASIS_KO` for the clip's reflected
  subscale;
- scene-aware only: ``scene_elements``, the clip's distinctive element, then
  its largest other element; ``seen``, the scene's setting: the distinctive
  element (researcher, 2026-10-24: one setting per clip; it was the object the
  V1 point fell on); ``visible_sounds``, the A1s sounds whose source the masks
  put on screen (``phase1_scene.ON_SCREEN``); ``offscreen``, a sentence placing
  the sounds not on screen "here, though unseen", or nothing when there are
  none. Sounds of the recording itself (:data:`RECORDING_ARTEFACTS`, microphone
  wind) are never placed: v2 spread them "in this scene" (hypothesis team,
  2026-10-24). Until v7 a V1 point on the noticed sound's click mask also made it
  visible, but a point can hit only a source already on screen
  (``phase1_binding.attention``), so the clause never changed the list;
- acousmatic: no slot of its own. Until v6 it had ``space``, a sentence on how
  the sound spreads, from the scene's openness (its sky share); the researcher
  dropped it on 2026-10-26 so that the acousmatic caption draws on the audio
  caption alone, with nothing taken from the picture.

FA1 (visited) is kept in the records and not used; FA2/FA3 and the place's being
away (P1–P2) are recorded with the subscales as the clip's baseline
(``model.baseline``).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any, Final

from diff.caption_sources import CaptionSource
from diff.job_plan import Job, tweak_job
from diff.phase1_binding import BASELINE, FOLLOW, assignment_plan, decide_all
from diff.rewrite_prompts import (
    NOT_SEEN,
    TWEAK_LADDER,
    TWEAK_PROMPTS,
    TWEAK_PROVISIONAL,
    TWEAK_WORDINGS,
    RenderedPrompt,
    render_tweak,
)

# v5 (2026-10-24): the scene-aware setting is the clip's distinctive element, not the V1 object.
# v6 (2026-10-26): the acousmatic prompt has no space slot, so the scene's openness reaches no caption.
# v7 (2026-10-26): the restorative emphasis steers both modes; captions are written live, on a τ ladder.
# v8 (2026-10-26): Qwen3.5-9B writes them, and the ladder's rungs are wordings.
# v9 (2026-10-27): Claude writes them through the API, on the same ladder.
# v10 (2026-10-27): the first wording is tried twice, since Claude's retries vary.
TWEAK_POLICY: Final = "diff.phase1-tweak/v10"
EMPHASIS_POLICY: Final = "diff.phase1-emphasis/reflect-v1"
STEP1_PAGES: Final = ("P2", "P3", "P4", "P5")
# The scene's elements (phase1_scene.ELEMENTS) in Korean.
ELEMENTS_KO: Final[dict[str, str]] = {
    "nature": "나무와 풀",
    "open": "하늘",
    "traffic": "도로와 차량",
    "built": "건물",
    "people": "사람들",
}
# The scene-aware prompt's sentence for the sounds not on screen, one wording for everyone; it
# avoids "화면 밖", which would echo E1's "화면 속 장소" (it starts with its own space).
OFFSCREEN_KO: Final = (
    " 화면에 음원이 보이지 않는 소리({sounds})는 "
    "화면에는 보이지 않지만 이곳 어딘가에서 들려오는 소리로 쓰세요."
)
# AudioSet labels for sounds of the recording, not of the place: never placed in the scene.
RECORDING_ARTEFACTS: Final = frozenset({"Wind noise (microphone)"})
# The restorative appraisal that steers the emphasis: the soundscape subscales of P4 (Payne, 2013),
# in the questionnaire's order, which breaks a tie left after the raw means. The place's being away
# (P1–P2, Hartig et al., 1997) rates the picture, not the sounds the caption describes.
RESTORATIVE: Final = ("fascination", "being_away_from", "extent_coherence")
# Each subscale's emphasis, one wording for everyone and both modes. None uses the P4/P9 items' own
# words (``phase1_usable.ITEM_STEMS``), which a caption must not echo before P9 rates them.
EMPHASIS_KO: Final[dict[str, str]] = {
    "fascination": "소리에서 귀를 사로잡는 점이 살아나도록 쓰세요.",
    "being_away_from": "소리를 들으며 잠시 다른 곳에 와 있는 듯한 느낌이 살아나도록 쓰세요.",
    "extent_coherence": "소리들이 따로 놀지 않고 서로 이어져 들리는 느낌이 살아나도록 쓰세요.",
}
# For named_in: a label's words that describe its sound rather than name it, by their ending (a
# verb before a noun, "달리는"; an adverb, "멀리서", "크게"; an object, "도로를") or as listed.
DESCRIBING: Final = re.compile(r"(?:는|서|게|를|을)$")
NOT_NAMING: Final = frozenset({"소리", "소음", "계속", "많은", "다른", "가까이"})
# Sounds a caption may name in other words (speech, shouting); a describing word that starts with
# one ("이야기하는", "외치는") names its sound. Vehicles stay out: "차량" is also a place in the
# captions ("도로와 차량", "차량들 사이로"), so it would hide a dropped "자동차 소리".
SOUND_WORDS: Final = (
    ("말소리", "목소리", "이야기", "대화", "웅성", "말하", "말한", "말해"),
    # "외쳐" as conjugated ("크게 외쳐지며"), 2026-10-26; "외친다" and the noun "외침", 2026-10-27.
    ("외치", "외쳐", "외친", "외침", "고함", "소리치"),
)
PARTICLES: Final = "(?:이|가|은|는|을|를|의|도|만|와|과|로)?"
# A one-syllable name ("새", "차") counts only as a noun: with a plural, "소리" (spaced or not) or a
# particle after it ("새가", "새들", "새 소리"), never alone ("쉴 새 없이", "새 건물") or before
# "로" ("새로", "차로"). Until 2026-10-27 it also counted alone, so "쉴 새 없이" named the bird.
NOUN_PARTICLES: Final = "(?:이|가|은|는|을|를|의|도|만|와|과)"


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


def emphases(answers: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Each clip's reflected subscale, from the participant's whole stage A (in stage-A order).

    A subscale's mean on a clip is centred on the participant's own mean of it over their clips,
    so how someone uses the scale, or rates one subscale above another everywhere, moves nothing.
    The clip's emphasis is the subscale it stands highest on; a tie goes to the higher raw mean,
    then to the order of :data:`RESTORATIVE`.
    """
    means = [
        {name: mean(float(given[item]) for item in BASELINE[name]) for name in RESTORATIVE}
        for given in answers
    ]
    centre = {name: mean(clip[name] for clip in means) for name in RESTORATIVE}
    records = []
    for clip in means:
        centred = {name: round(clip[name] - centre[name], 6) for name in RESTORATIVE}
        rank = sorted(RESTORATIVE, key=lambda n: (-centred[n], -clip[n], RESTORATIVE.index(n)))
        records.append(
            {
                "policy": EMPHASIS_POLICY,
                "rule": "reflect",
                "chosen": rank[0],
                "tied": centred[rank[0]] == centred[rank[1]],
                "subscales": {
                    name: {
                        "mean": round(clip[name], 4),
                        "centre": round(centre[name], 4),
                        "centred": centred[name],
                    }
                    for name in RESTORATIVE
                },
            }
        )
    return records


def scene_slots(clip: Mapping[str, Any], mode: str) -> dict[str, str]:
    """The mode's own slots, from the clip's scene part."""
    if mode == "acousmatic":
        return {}
    scene = clip["scene"]
    distinctive = scene["distinctive"]
    others = sorted(
        (name for name in scene["elements"] if name != distinctive), key=lambda n: -scene["elements"][n]
    )
    visible = [option["label"] for option in clip["options"]["A1s"] if on_screen(clip, option)]
    unseen = [
        option["label"]
        for option in clip["options"]["A1s"]
        if not on_screen(clip, option) and option["audioset_label"] not in RECORDING_ARTEFACTS
    ]
    return {
        "scene_elements": ", ".join(ELEMENTS_KO[name] for name in (distinctive, others[0])),
        "seen": ELEMENTS_KO[distinctive],
        "visible_sounds": ", ".join(dict.fromkeys(visible)) or NOT_SEEN,
        "offscreen": OFFSCREEN_KO.format(sounds=", ".join(dict.fromkeys(unseen))) if unseen else "",
    }


def caption_variant(clip: Mapping[str, Any], mode: str) -> dict[str, Any]:
    """The caption's variant for analysis: the mode, and for scene-aware the setting it draws on."""
    if mode == "acousmatic":
        return {"type": "acousmatic"}
    return {"type": f"scene/{clip['scene']['distinctive']}"}


def stratified_plan(
    clips: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Mapping[str, Any]]],
    seed: int,
    *,
    root: Path,
) -> list[str]:
    """The participant's follow/counter plan, drawn from their recommendations once stage A is in."""
    answers = [step_answers(step, str(clip["key"])) for clip, step in zip(clips, records, strict=True)]
    matches = [(step["P2"].get("matches") or [None])[0] for step in records]
    models = decide_all(clips, answers, matches, [FOLLOW] * len(clips), root=root)
    return assignment_plan(seed, [model["recommended"] for model in models])


def tweak_steer_all(
    clips: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Mapping[str, Any]]],
    plan: Sequence[str],
    *,
    root: Path,
) -> list[tuple[dict[str, str], dict[str, Any]]]:
    """Each clip's prompt slots and steering record, from the participant's whole stage A.

    ``clips`` and ``records`` (each clip's first-step records, by page) run in stage-A order. The
    record is the binding model's, with the clip's ``emphasis`` and ``caption`` variant added.
    """
    answers = [step_answers(step, str(clip["key"])) for clip, step in zip(clips, records, strict=True)]
    matches = [(step["P2"].get("matches") or [None])[0] for step in records]
    models = decide_all(clips, answers, matches, plan, root=root)
    steered = []
    for clip, given, model, emphasis in zip(clips, answers, models, emphases(answers), strict=True):
        model["emphasis"] = emphasis
        slots = tweak_slots(clip, model["mode"], given.get("A1s"), emphasis["chosen"])
        model["caption"] = caption_variant(clip, model["mode"])
        steered.append((slots, model))
    return steered


def tweak_slots(clip: Mapping[str, Any], mode: str, attended: str | None, emphasis: str) -> dict[str, str]:
    """One clip's prompt slots in ``mode``, from the attended sound (A1s) and the reflected subscale."""
    sounds = {option["value"]: option["label"] for option in clip["options"]["A1s"]}
    if str(attended) not in sounds:
        # A1s is a required choice among the clip's options; the prompt must never name "no sound" as the
        # one the participant attended to.
        raise TweakError(
            f"{clip['key']}: the attended sound {attended!r} is not one of the clip's A1s options"
        )
    mentioned = [option["label"] for option in clip["options"]["R2"] if option.get("mentioned")]
    return {
        "mode": mode,
        "caption": clip["caption"]["text"],
        "sounds": ", ".join(mentioned),
        "attended_sound": sounds[str(attended)],
        "emphasis": EMPHASIS_KO[emphasis],
        **scene_slots(clip, mode),
    }


def on_screen(clip: Mapping[str, Any], option: Mapping[str, Any]) -> bool:
    """Whether the masks put an A1s option's source on screen (its scene part's measured share)."""
    return bool(clip["scene"]["sounds"][option["value"]]["visibility"]["on_screen"])


def reachable_slots(clip: Mapping[str, Any]) -> list[dict[str, str]]:
    """Every slot set a participant can reach on this clip, so every second caption it can show:
    one per mode, attended sound (A1s, a required choice) and emphasis."""
    found: dict[str, dict[str, str]] = {}
    for option in clip["options"]["A1s"]:
        for mode in TWEAK_PROMPTS:
            for emphasis in RESTORATIVE:
                slots = tweak_slots(clip, mode, option["value"], emphasis)
                found.setdefault(variant_id(slots), slots)
    return list(found.values())


def variant_id(slots: Mapping[str, str]) -> str:
    """A second caption's key: a hash of its prompt's slots, by which a reviewed caption is found."""
    return hashlib.sha256(json.dumps(slots, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:12]


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


def template_record(mode: str, wording: str = TWEAK_LADDER[0]) -> dict[str, Any]:
    template = TWEAK_WORDINGS[wording][mode]
    return {
        "id": template.template_id,
        "version": template.version,
        "mode": mode,
        "wording": wording,
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
    """The source, the first rung's job and its steering record for one clip decided alone."""
    ((slots, _),) = tweak_steer_all([clip], [records], [assignment], root=root)
    return tweak_job_for(clip, slots, purpose)


def tweak_job_for(
    clip: Mapping[str, Any], slots: Mapping[str, str], purpose: str, rung: int = 0
) -> tuple[CaptionSource, Job, dict[str, Any]]:
    """The source, the job at ``rung`` of ``rewrite_prompts.TWEAK_LADDER`` (its wording) and its
    steering record, from slots already computed (and stored). The record carries what the gate
    needs to judge the output."""
    if not 0 <= rung < len(TWEAK_LADDER):
        raise ValueError(f"the ladder has rungs 0 to {len(TWEAK_LADDER) - 1}, not {rung}")
    wording = TWEAK_LADDER[rung]
    prompt: RenderedPrompt = render_tweak(slots, wording)
    source = tweak_source(clip)
    job = tweak_job(source, prompt, purpose)
    record = {
        "policy": TWEAK_POLICY,
        "template": template_record(slots["mode"], wording),
        "slots": dict(slots),
        "intended_r2_key": intended_r2_key(clip),
        "job_id": job.job_id,
        "rung": rung,
        "rungs": len(TWEAK_LADDER),
        "wording": wording,
        "gate": gate_inputs(clip),
    }
    return source, job, record


def gate_inputs(clip: Mapping[str, Any]) -> dict[str, list[str]]:
    """The clip's sound names the gate judges a caption against: R2's options not in the audio, and
    every option's label (a place word inside one names the sound, not a place)."""
    r2 = clip["options"]["R2"]
    return {
        "distractors": [option["label"] for option in r2 if not option.get("mentioned")],
        "names": sorted({option["label"] for option in (*clip["options"]["A1s"], *r2)}),
    }


def sound_name(label: str) -> str:
    """The word of an option's label that names its sound: its last naming word, without particles
    ("경찰차 사이렌 소리" → 사이렌, "진행자인 남자의 말소리" → 말소리, "크게 외치는 소리" → 외치)."""
    names: list[str] = []
    for word in re.findall(r"[가-힣]+", label):
        if word in NOT_NAMING:
            continue
        if DESCRIBING.search(word):
            names += [w for group in SOUND_WORDS for w in group if len(w) > 1 and word.startswith(w)][:1]
            continue
        if len(word) > 1:
            word = re.sub(r"(?:의|이|가|인|에)$", "", word)
        if len(word) > 1:
            word = re.sub(r"들$", "", word)
        names.append(word)
    return names[-1] if names else ""


def named_in(text: str, label: str) -> bool:
    """Whether a caption seems to name the sound an option's label stands for: a word of the caption
    starts with the label's :func:`sound_name`, one of its :data:`SOUND_WORDS`, or the bird of
    "새소리" ("새가"); a one-syllable name only as a noun (:data:`NOUN_PARTICLES`). Rough: two options
    told apart only by a describing word (bangkok_209's near and background women's voices) are named
    together, a "차량" named as a place counts for a "차량" sound, and "자동차" does not count for
    it. A hint for the researcher's blind coding of R2 against the shown caption, never the key
    itself.

    Until 2026-10-25 any word of the label matched by its first two syllables anywhere in the
    caption, so "사이렌" matched "사이로" and "들리는" matched "들린다" (design review, I33)."""
    name = sound_name(label)
    if not name:
        return False
    group = next((g for g in SOUND_WORDS if name in g), (name,))
    stems = group + ((name[:-2],) if name.endswith("소리") and len(name) > 2 else ())
    words = re.findall(r"[가-힣]+", text)
    return any(
        any(word.startswith(stem) for word in words) if len(stem) > 1 else _as_noun(stem, text)
        for stem in stems
    )


def _as_noun(stem: str, text: str) -> bool:
    """Whether a one-syllable name stands in the caption as a noun (:data:`NOUN_PARTICLES`)."""
    pattern = rf"(?<![가-힣]){stem}(?:들|\s?소리|들\s?소리)?{NOUN_PARTICLES}?(?![가-힣])"
    return any(len(match.group(0)) > 1 for match in re.finditer(pattern, text))


def intended_r2_key(clip: Mapping[str, Any]) -> list[str]:
    """The R2 options the prompt names (the mentioned ones): R2's key, intended, not checked."""
    return [option["value"] for option in clip["options"]["R2"] if option.get("mentioned")]
