"""Phase 1's gate for a second caption: what must hold before a participant is shown it.

Researcher, 2026-10-26: second captions are written live, when stage A ends, and Trida is
made to deliver usable ones by engineering rather than by the researcher approving every
caption before the study. This gate adds to the server's technical check
(:func:`diff.output_checks.check_tweak`) the defects the 2026-10-24 bench and the
2026-10-26 drafts found that check letting through:

- **Decoding damage:** a loop (the technical check's ``repetition``, a flag there), a word
  cut or doubled by parallel decoding (:data:`BROKEN`, :data:`DAMAGE`), or a caption cut
  off mid-clause (:data:`CUT_OFF`). The first captions' endings are free (researcher,
  2026-10-18), so a caption may end on a noun; it may not end on a connective.
- **No rewrite:** a near-copy of the first caption (:data:`NEAR_COPY`), or a list of names: two or
  more comma-separated phrases and no predicate anywhere (:data:`PREDICATE`; researcher, 2026-10-26).
  A list carries neither the emphasis nor the mode. A noun ending after a clause stays allowed.
- **Steering lost:** the attended sound not named, the one thing the caption centres.
- **A false sound:** an R2 option not in the audio named, which would also make R2's
  intended key wrong.
- **Priming:** wording of the P4/P9 items (:data:`ITEM_STEMS`), which the participant
  rates after the second viewing, or the emphasis instruction's own words (:data:`ECHO_STEMS`): a
  caption should carry the quality, not state it.
- **The mode:** an acousmatic caption naming a place or object on screen; a scene-aware
  one naming a place the prompt did not give.

A failure sends the clip to the next rung of ``rewrite_prompts.TWEAK_LADDER``; when none is
left the clip's second step is skipped, as before. Flags are recorded, never acted on:
other sounds of the prompt left unnamed (the matcher is rough, and R2's key is intended,
not checked) and spatial relations (:data:`RELATIONS`), which the drafts invented.
The rules are provisional until the researcher approves them with the prompt wording.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

from diff.caption_sources import CaptionSource
from diff.output_checks import CheckResult, Finding, check_tweak, displayed, output_from_record, readable
from diff.phase1_tweak import ELEMENTS_KO, named_in
from diff.study_protocol import TWEAK_MAX_CHARS

# v2 (2026-10-26): a list of names is refused.
# v3 (2026-10-26): every item's own words, the emphasis's own words, and "차" as a vehicle.
# v4 (2026-10-26, on Qwen3.5-9B's drafts): "채," is no cut verb; "somewhere else" in other words.
# v5 (2026-10-27, on Claude's drafts): named_in reads "쉴 새 없이" as no bird, and "외침" as a shout.
USABLE_POLICY: Final = "diff.phase1-usable/v5"
# The mask objects of regen's c2 frames, in Korean: things a caption can name on screen.
OBJECTS_KO: Final = ("건물", "사람", "도로", "차량", "인도", "하늘", "땅", "나무와 풀")
# Words for a place or an object on screen: the objects, the scene elements and common street
# places, two syllables or more ("길" and "강" would match "길게" and "강하게"; "거리" is also the
# distance the acousmatic prompt asks for, so it is left out).
PLACE_WORDS: Final = tuple(
    sorted(
        {
            word
            for phrase in (*OBJECTS_KO, *ELEMENTS_KO.values())
            for word in re.findall(r"[가-힣]{2,}", phrase.replace("와 ", " "))
        }
        | {"골목", "광장", "가게", "상점", "시장", "공원", "정류장", "화면", "장면", "도시", "카페", "보도"}
    )
)
# Place words a scene-aware caption may use without the prompt giving them: the screen it is on.
DEICTIC: Final = frozenset({"화면", "장면"})
# Wording of the P4/P9 items a caption should not echo (until v3 only E1/E3's place, E1's belonging,
# E2's merging and BF's escape): F1 liking, F2 interest and drawn attention, F3 immersion; BF1 refuge
# from disturbance, BF2 freedom from routine and duty, BF3 usual sounds; E1 the place and belonging,
# E2 merging and consistency, E3 fitting; the sound environment of F, BF and E2; P1–P2's rest.
ITEM_STEMS: Final = (
    "화면 속 장소",
    "속해",
    "어우러",
    "벗어나",
    "소리 환경",
    "마음에 든",
    "흥미로",
    "주의가 이끌",
    "몰입",
    "방해",
    "피난처",
    "일상",
    "책임",
    "자유로",
    "평소 듣던",
    "일관된",
    "어울리",
    "쉬어가",
)
# The emphasis sentences' own words (``phase1_tweak.EMPHASIS_KO``), one pattern each: "somewhere
# else" stated as being there ("다른 곳에 와 있는", "다른 곳에 서 있다"), not a sound from elsewhere.
ECHO_STEMS: Final = ("귀를 사로잡", r"다른 (?:곳|공간|장소)에 (?:와|서|있)", "따로 놀지")
# A word cut to "소" (from "소리"), and doubled commas or full stops ("..." is an ellipsis).
BROKEN: Final = re.compile(r"(?<![가-힣])소(?![가-힣])|,,|(?<!\.)\.\.(?!\.)")
# A verb cut at a comma ("퍼,", "흐,"), a stray syllable, a doubled word (the bench's wording
# round, 2026-10-24). "채," left the list in v4: "소리가 깔린 채," is a whole phrase, which Qwen writes.
DAMAGE: Final = re.compile(
    r"(?:퍼|흐)[,.](?!\.)|(?<!버)스[,.](?!\.)|(?:^|\s)(?:들|울|공|적|체)[,.\s]|(?<!경)적음"
    r"|에서에서|의에서|([가-힣]{2,}(?: 소리가)?)\s\1(?![가-힣])"
)
# A caption that stops on a connective ending, an adverb ("희미하게."), a place it starts from
# ("멀리서.") or a comma: the clause it opens never comes.
CUT_OFF: Final = re.compile(r"(?:며|고|면서|는데|지만|으나|거나|(?<!가)게|서|[,，、])\s*[.…]*\s*$")
NEAR_COPY: Final = 0.9
# A predicate ending: "들린다", "울립니다", "들려요". A caption with none, in two or more comma-separated
# phrases, is a list of names ("경찰차 사이렌 소리, 기차 소리").
PREDICATE: Final = re.compile(r"(?:다|요)(?=[.!?…\s]|$)")
# Vehicle words a caption may use for one another ("차량" for "자동차"); ``named_in`` leaves them
# out because "차량" is also a place in the scene prompt, but a dropped vehicle is what it guards.
VEHICLES: Final = ("자동차", "차량", "차들", "버스", "트럭", "오토바이", "택시")
# "차" as a word of its own, bare or with a particle ("차가 지나간다"), not inside "차가운".
CAR: Final = re.compile(r"(?<![가-힣])차(?:가|는|들|의|를|도|와|로|만)?(?![가-힣])")
RELATIONS: Final = re.compile(r"너머|뒤에서|뒤로|건너편")


def _vehicle(text: str) -> bool:
    return any(word in text for word in VEHICLES) or bool(CAR.search(text))


def attended_named(text: str, label: str) -> bool:
    """Whether the caption names the attended sound: ``named_in``, with vehicles counted together."""
    return named_in(text, label) or (_vehicle(label) and _vehicle(text))


def place_words(text: str, names: Sequence[str]) -> list[str]:
    """Place or object words in a caption, outside the sounds' own names ("도로를 달리는 차량 소리")."""
    return [word for word in PLACE_WORDS if word in text and not any(word in name for name in names)]


def findings(
    shown: str, source: str, slots: Mapping[str, str], gate: Mapping[str, Sequence[str]]
) -> tuple[list[Finding], list[Finding]]:
    """The gate's failures and flags for one shown caption, its first caption and prompt slots."""
    failures: list[Finding] = []
    flags: list[Finding] = []
    damaged = [m.group(0) for m in BROKEN.finditer(shown)] + [m.group(0) for m in DAMAGE.finditer(shown)]
    if damaged:
        failures.append(Finding("damaged_words", ", ".join(damaged)))
    if CUT_OFF.search(shown):
        failures.append(Finding("cut_off", shown[-8:]))
    similarity = difflib.SequenceMatcher(None, re.sub(r"\s+", "", shown), re.sub(r"\s+", "", source)).ratio()
    if similarity >= NEAR_COPY:
        failures.append(Finding("near_copy", f"{similarity:.2f} of the first caption"))
    phrases = [part for part in re.split(r"[,，、]", shown) if part.strip()]
    if len(phrases) >= 2 and not PREDICATE.search(shown):
        failures.append(Finding("list_of_names", f"{len(phrases)} phrases, no predicate"))
    attended = slots["attended_sound"]
    if not attended_named(shown, attended):
        failures.append(Finding("attended_unnamed", attended))
    false = [name for name in gate["distractors"] if named_in(shown, name)]
    if false:
        failures.append(Finding("distractor_named", ", ".join(false)))
    echoed = [stem for stem in ITEM_STEMS if stem in shown]
    if echoed:
        failures.append(Finding("item_wording", ", ".join(echoed)))
    stated = [m.group(0) for stem in ECHO_STEMS if (m := re.search(stem, shown))]
    if stated:
        failures.append(Finding("emphasis_echo", ", ".join(stated)))
    places = place_words(shown, gate["names"])
    if slots["mode"] == "acousmatic" and places:
        failures.append(Finding("place_named", ", ".join(places)))
    if slots["mode"] == "scene":
        given = " ".join((slots["scene_elements"], slots["seen"], source))
        invented = [word for word in places if word not in given and word not in DEICTIC]
        if invented:
            failures.append(Finding("place_invented", ", ".join(invented)))
    unnamed = [name for name in slots["sounds"].split(", ") if name and not named_in(shown, name)]
    if unnamed:
        flags.append(Finding("sound_unnamed", ", ".join(unnamed)))
    relations = sorted({m.group(0) for m in RELATIONS.finditer(shown)})
    if relations:
        flags.append(Finding("relation_word", ", ".join(relations)))
    return failures, flags


def check_record(
    source: CaptionSource, steer: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    """The technical check and the gate for one completed call (``LiveService(tweak_check=...)``),
    on the caption as shown. ``steer`` is the request's record (``phase1_tweak.tweak_job_for``)."""
    output = displayed(output_from_record(result["output"]))
    checked = check_tweak(source, output, max_chars=TWEAK_MAX_CHARS)
    failures: list[Finding] = list(checked.failures)
    flags: list[Finding] = []
    for flag in checked.flags:
        (failures if flag.code == "repetition" else flags).append(flag)
    if readable(output.text):  # with nothing to read, the display rule's refusal is the reason
        failed, flagged = findings(output.text, source.text, steer["slots"], steer["gate"])
        failures += failed
        flags += flagged
    record = CheckResult(checked.profile, tuple(failures), tuple(flags), checked.measures).record()
    return {**record, "gate": USABLE_POLICY}
