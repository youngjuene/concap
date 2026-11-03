"""The perceived-source block: after the post-watch items, 21 of the long videos' 10 s cues are replayed
without a caption, and for each the participant says where its named sound seems to come from.

Researcher's decision of 2026-11-03 (``artifacts/diff/phase2-scene/design-sim/PLAN.md``, D6c). No text
is written for the block: the sound is the cue's lead sound as listened, the answers are fixed, so no
writer, length gap or wording can stand between the percept and the answer. The analysis is the slope
of the answer on the lead sound's logit q, tested by person and by item (min F').

Each participant gets their own stratified draw, 10 cues whose lead sound's source is on screen by the
masks and 11 whose is not, as the design simulation ran it. Cues are grouped by video so that the page
changes its video file at most twice; the order inside each video is random. A cue whose lead sound has
no mask (``mask_match`` is ``none``) has no measured q, so it is never drawn. The draw is seeded by the
session ID, so the audit can re-derive it.

The items come first (``diff.live_server``): this question would otherwise colour the ratings of the
captions just watched. The researcher approved the Korean wording on 2026-11-03.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from typing import Any, Final

SOURCE_POLICY: Final = "diff.source-check/v1"
SOURCE_CHECK_PROVISIONAL: Final = False  # the researcher approved the wording, 2026-11-03
TRIALS_ON_SCREEN: Final = 10
TRIALS_OFF_SCREEN: Final = 11
QUESTION: Final = "이 소리는 어디에서 나는 것처럼 들렸나요?"
ON_SCREEN_ANSWER: Final = "on_screen"
OFF_SCREEN_ANSWER: Final = "off_screen"
ANSWERS: Final = {
    ON_SCREEN_ANSWER: "화면에 보이는 것에서",
    OFF_SCREEN_ANSWER: "화면에 보이지 않는 곳에서",
}


def _pools(
    scenes: Mapping[str, Any], videos: Sequence[Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Every drawable cue as a trial, split by whether its lead sound's source is on screen."""
    on_screen: list[dict[str, Any]] = []
    off_screen: list[dict[str, Any]] = []
    for video in videos:
        for cue in video.stream.cues:
            scene = scenes["cues"].get(cue.source_id)
            if scene is None or not scene["sounds"] or scene["sounds"][0].get("mask_match") != "label":
                continue
            lead = scene["sounds"][0]
            visibility = lead["visibility"]
            (on_screen if visibility["on_screen"] else off_screen).append(
                {
                    "source_id": cue.source_id,
                    "video_id": video.video_id,
                    "cue": cue.cue_index,
                    "start_ms": cue.start_ms,
                    "end_ms": cue.end_ms,
                    "sound": lead["sound"],
                    "visibility": float(visibility["mean"]),
                    "on_screen": bool(visibility["on_screen"]),
                }
            )
    return on_screen, off_screen


def pool_problems(scenes: Mapping[str, Any] | None, videos: Sequence[Any]) -> list[str]:
    """Why these cues cannot fill the block; empty when a whole draw is possible."""
    if scenes is None:
        return ["the perceived-source block has no scene parts: pass --phase2-scene"]
    on_screen, off_screen = _pools(scenes, videos)
    problems = []
    if len(on_screen) < TRIALS_ON_SCREEN:
        problems.append(f"{len(on_screen)} on-screen cues, the block draws {TRIALS_ON_SCREEN}")
    if len(off_screen) < TRIALS_OFF_SCREEN:
        problems.append(f"{len(off_screen)} off-screen cues, the block draws {TRIALS_OFF_SCREEN}")
    return problems


def draw(session_id: str, scenes: Mapping[str, Any], videos: Sequence[Any]) -> dict[str, Any]:
    """This session's trials and answer order; the same session always draws the same."""
    rng = random.Random(hashlib.sha256(f"{SOURCE_POLICY}:{session_id}".encode()).digest())
    on_screen, off_screen = _pools(scenes, videos)
    chosen = rng.sample(on_screen, TRIALS_ON_SCREEN) + rng.sample(off_screen, TRIALS_OFF_SCREEN)
    order = [video.video_id for video in videos]
    rng.shuffle(order)
    rng.shuffle(chosen)
    chosen.sort(key=lambda trial: order.index(trial["video_id"]))  # stable: random inside each video
    answers = list(ANSWERS)
    rng.shuffle(answers)
    return {"policy": SOURCE_POLICY, "answer_order": answers, "trials": chosen}
