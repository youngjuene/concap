"""The binding model: closed forms, evidence ratios, the tendency and centring it uses, and the plan."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
from phase1_fixtures import approved_setup
from test_phase1_tweak import ON_THE_ROAD, first_step

from diff.phase1_binding import (
    AREA_FLOOR,
    COUNTER,
    DRAWN,
    FOLLOW,
    GRID_SIZE,
    assignment_plan,
    attention,
    balanced_plan,
    congruence,
    decide_all,
    p_bound,
    summary,
    tendency,
)

UNIFORM = [1 / GRID_SIZE] * GRID_SIZE


# Five clips recommended scene-aware and seven acousmatic, in stage-A order.
RECOMMENDED = ["scene", "acousmatic", "acousmatic", "scene", "acousmatic", "scene"] + [
    "acousmatic",
    "scene",
    "acousmatic",
    "acousmatic",
    "scene",
    "acousmatic",
]


def shown_scene(plan: list[str], recommended: list[str]) -> int:
    return sum((rec == "scene") == (step == FOLLOW) for step, rec in zip(plan, recommended, strict=True))


def test_the_stratified_plan_halves_each_recommendation_and_is_fixed_by_the_seed() -> None:
    plan = assignment_plan(7, RECOMMENDED)
    assert plan.count(FOLLOW) == plan.count(COUNTER) == 6
    assert plan == assignment_plan(7, RECOMMENDED) and plan != assignment_plan(8, RECOMMENDED)
    scene = [step for step, rec in zip(plan, RECOMMENDED, strict=True) if rec == "scene"]
    assert scene.count(FOLLOW) in (2, 3)  # the odd strata share one extra follow by a coin
    odd = assignment_plan(7, ["scene", "acousmatic", "acousmatic"])
    assert odd.count(FOLLOW) == 1 and odd.count(COUNTER) == 2
    with pytest.raises(ValueError):
        assignment_plan(7, ["scene", "maybe"])


def test_every_participant_sees_both_modes_and_each_clip_follows_half_the_time() -> None:
    plans = [assignment_plan(seed, RECOMMENDED) for seed in range(2000)]
    # 5–7 scene-aware captions of 12 for everyone; the balanced plan let this reach 3 or 9.
    assert {shown_scene(plan, RECOMMENDED) for plan in plans} <= {5, 6, 7}
    for index in range(len(RECOMMENDED)):
        share = sum(plan[index] == FOLLOW for plan in plans) / len(plans)
        assert abs(share - 0.5) < 0.04, (index, share)
    spread = {shown_scene(balanced_plan(seed, 12), RECOMMENDED) for seed in range(2000)}
    assert min(spread) <= 4 and max(spread) >= 8


def test_closed_forms_at_an_even_visibility() -> None:
    # With q = 0.5 the prior is π itself, so P(bound) = E[π]·r / (E[π]·r + 1 − E[π]).
    assert p_bound(UNIFORM, 0.5, 0.0) == pytest.approx(0.5)
    assert p_bound(UNIFORM, 0.5, math.log(3)) == pytest.approx(0.75)
    assert p_bound(UNIFORM, 0.9, 0.0) > 0.5 > p_bound(UNIFORM, 0.1, 0.0)
    assert summary(UNIFORM)["mean"] == pytest.approx(0.5)


def test_the_tendency_carries_other_clips_evidence() -> None:
    bound = summary(tendency([(0.5, math.log(10))] * 4))
    detached = summary(tendency([(0.5, -math.log(10))] * 4))
    assert detached["mean"] < 0.5 < bound["mean"]
    assert bound["low"] <= bound["mean"] <= bound["high"]
    # A clip with a strongly visible source says little about the participant: it is bound anyway.
    assert abs(summary(tendency([(0.95, math.log(3))]))["mean"] - 0.5) < abs(bound["mean"] - 0.5)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def setup(root: Path) -> dict[str, Any]:
    return approved_setup(root)


def test_a_point_counts_by_the_click_mask_s_share_of_the_frame(setup: dict[str, Any], root: Path) -> None:
    bus = setup["clips"][0]["scene"]["sounds"]["s1"]
    match = {"frame": 0, "x": ON_THE_ROAD[0], "y": ON_THE_ROAD[1]}
    hit = attention(bus, match, root)
    area = bus["click_frames"]["0"]["area"]
    assert hit["status"] == "hit" and hit["log_ratio"] == pytest.approx(
        math.log((DRAWN + (1 - DRAWN) * area) / area), abs=1e-6
    )
    small = {**bus, "click_frames": {"0": {**bus["click_frames"]["0"], "area": 0.01}}}
    assert attention(small, match, root)["log_ratio"] == pytest.approx(
        math.log((DRAWN + (1 - DRAWN) * AREA_FLOOR) / AREA_FLOOR), abs=1e-6
    )
    assert attention(bus, {**match, "y": 0.1}, root)["log_ratio"] == pytest.approx(
        math.log(1 - DRAWN), abs=1e-6
    )
    assert attention(bus, None, root) == {"status": "no point", "log_ratio": 0.0}
    # A source the masks do not put on screen says nothing, wherever its click mask lies.
    off = {**bus, "visibility": {**bus["visibility"], "on_screen": False}}
    assert attention(off, match, root)["status"] == "source not seen"
    horn = setup["clips"][1]["scene"]["sounds"]["s2"]  # lima's horn lies in the sky: never seen
    assert attention(horn, match, root)["status"] == "source not seen"


def test_congruence_counts_from_the_participant_s_own_level() -> None:
    assert congruence({"E1": 6, "E3": 6}, 3)["log_ratio"] == pytest.approx(math.log(3))
    assert congruence({"E1": 0, "E3": 0}, 3)["log_ratio"] == pytest.approx(-math.log(3))
    assert congruence({"E1": 6, "E3": 6}, 6)["log_ratio"] == 0.0
    assert congruence({"E1": 2, "E3": 4}, 3) == {
        "items": ["E1", "E3"],
        "mean": 3,
        "centre": 3,
        "log_ratio": 0.0,
    }


def stage_a(setup: dict[str, Any], fits: tuple[int, ...], sound: str = "s1") -> tuple[list[Any], ...]:
    clips = setup["clips"]
    steps = [first_step(sound=sound, fit=fit) for fit in fits]
    answers = [
        {k: v for page in ("P2", "P3", "P4", "P5") for k, v in step[page]["answers"].items()}
        for step in steps
    ]
    return clips, answers, [step["P2"]["matches"][0] for step in steps]


def test_a_rating_level_moves_nothing_and_a_clip_s_own_point_is_not_its_tendency(
    setup: dict[str, Any], root: Path
) -> None:
    plan = [FOLLOW] * 3
    high = decide_all(*stage_a(setup, (6, 6, 6)), plan, root=root)
    low = decide_all(*stage_a(setup, (1, 1, 1)), plan, root=root)
    assert [r["p_bound"] for r in high] == [r["p_bound"] for r in low]
    assert {r["congruence"]["log_ratio"] for r in high} == {0.0} and high[0]["congruence"]["centre"] == 6
    mixed = decide_all(*stage_a(setup, (6, 3, 0)), plan, root=root)
    assert mixed[0]["p_bound"] > high[0]["p_bound"] and high[2]["p_bound"] > mixed[2]["p_bound"]
    # Kyoto's own hit leaves its tendency as the other two clips' points set it.
    assert high[0]["tendency"]["clips"] == 2
    alone = decide_all(*(part[1:] for part in stage_a(setup, (6, 6, 6))), plan[1:], root=root)
    assert alone[1]["tendency"]["mean"] == pytest.approx(
        summary(tendency([(high[1]["visibility"], high[1]["attention"]["log_ratio"])]))["mean"]
    )


def test_the_moderators_split_p_bound_into_the_scene_s_part_and_the_answers(
    setup: dict[str, Any], root: Path
) -> None:
    def logit(p: float) -> float:
        return math.log(p / (1 - p))

    # No points and level ratings: the answers say nothing.
    clips, answers, matches = stage_a(setup, (3, 3, 3), sound="s2")
    silent = decide_all(clips, answers, [None] * 3, [FOLLOW] * 3, root=root)[2]
    assert silent["moderator_participant"] == 0.0
    # The other clips' points alone move it, through the tendency.
    assert decide_all(clips, answers, matches, [FOLLOW] * 3, root=root)[2]["moderator_participant"] != 0.0
    assert silent["moderator_scene"] == pytest.approx(logit(silent["visibility"]), abs=1e-6)
    # A flat tendency pulls P(b) toward 1/2, so logit q alone would not have been 0.
    assert logit(silent["p_bound"]) - silent["moderator_scene"] > 0.5
    # Points on the bus and a high rating add to P(b), and the moderator is all of that addition.
    bound = decide_all(*stage_a(setup, (6, 3, 0)), [FOLLOW] * 3, root=root)[0]
    alone = p_bound(UNIFORM, bound["visibility"], 0.0)
    assert bound["moderator_participant"] > 0
    assert logit(alone) + bound["moderator_participant"] == pytest.approx(logit(bound["p_bound"]), abs=1e-3)


def test_decide_follows_or_counters_the_recommendation(setup: dict[str, Any], root: Path) -> None:
    clips, answers, matches = stage_a(setup, (3, 3, 3), sound="s2")  # oslo's wind: seen nowhere
    followed = decide_all(clips, answers, matches, [FOLLOW] * 3, root=root)[2]
    assert followed["noticed"]["on_screen"] is False and followed["measured_visibility"] == 0.0
    assert followed["visibility"] == pytest.approx(1 / 12, abs=1e-4)  # the share's posterior mean
    assert followed["p_bound"] < 0.5 and (followed["recommended"], followed["mode"]) == (
        "acousmatic",
        "acousmatic",
    )
    assert decide_all(clips, answers, matches, [COUNTER] * 3, root=root)[2]["mode"] == "scene"
    with pytest.raises(ValueError, match="assignment"):
        decide_all(clips, answers, matches, ["maybe"] * 3, root=root)
    with pytest.raises(ValueError, match="one clip"):
        decide_all(clips, answers, matches, [FOLLOW], root=root)
