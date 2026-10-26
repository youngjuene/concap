"""Phase 1's second caption: the mode, emphasis and slots from first-step answers, the prompt, the
job on its ladder and its check."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from phase1_fixtures import approved_setup

from diff.job_plan import PlanError
from diff.output_checks import GeneratedOutput, check_tweak
from diff.phase1_tweak import (
    EMPHASIS_KO,
    RESTORATIVE,
    TWEAK_LADDER,
    TweakError,
    emphases,
    named_in,
    reachable_slots,
    tweak_job_for,
    tweak_request,
    tweak_source,
    tweak_steer_all,
    variant_id,
)
from diff.phase1_usable import ECHO_STEMS, ITEM_STEMS
from diff.rewrite_prompts import TWEAK_PROMPTS, TWEAK_PROVISIONAL, TWEAK_SLOTS, render_tweak
from diff.study_protocol import LIVE_CONDITION, TWEAK_MAX_CHARS

RATINGS = ("F1", "F2", "F3", "BF1", "BF2", "BF3", "E1", "E2", "E3", "P1", "P2")
ON_THE_ROAD = (0.5, 0.9)  # the road, and kyoto's bus


def first_step(
    *,
    sound: str = "s1",
    obj: str | None = "road",
    point: tuple[float, float] = ON_THE_ROAD,
    fit: int = 6,
    rest: int = 3,
) -> dict[str, dict[str, Any]]:
    ratings = {item: (fit if item in ("E1", "E3") else rest) for item in RATINGS}
    x, y = point
    return {
        "P2": {
            "answers": {"V1": [{"frame_id": "0", "x": x, "y": y}]},
            "matches": [{"order": 0, "frame": 0, "x": x, "y": y, "object_id": obj}],
        },
        "P3": {"answers": {"A1s": sound}},
        "P4": {"answers": ratings},
        "P5": {"answers": {"FA1": "no", "FA2": 0, "FA3": 6}},
    }


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def clip(root: Path) -> dict[str, Any]:
    return approved_setup(root)["clips"][0]  # kyoto_101


def steer(
    clip: dict[str, Any], records: dict[str, dict[str, Any]], assignment: str, root: Path
) -> tuple[dict[str, str], dict[str, Any]]:
    """One clip decided alone: no other clips inform its tendency or centre its E1/E3."""
    ((slots, model),) = tweak_steer_all([clip], [records], [assignment], root=root)
    return slots, model


def test_a_bound_sound_followed_is_written_scene_aware(clip: dict[str, Any], root: Path) -> None:
    slots, model = steer(clip, first_step(), "follow", root)
    assert (model["recommended"], model["mode"], model["attention"]["status"]) == ("scene", "scene", "hit")
    assert set(slots) == set(TWEAK_SLOTS["scene"])
    assert slots == {
        "mode": "scene",
        "caption": "버스가 공회전하는 소리",
        "sounds": "버스 소리",
        "attended_sound": "버스 소리",
        # Alone, the clip stands level with itself on every subscale: the raw means break the tie.
        "emphasis": EMPHASIS_KO["extent_coherence"],
        "scene_elements": "하늘, 도로와 차량",
        "seen": "하늘",  # the distinctive element, wherever the V1 point fell (on the road)
        "visible_sounds": "버스 소리, 남자의 말소리",  # the man is judged partly visible
        "offscreen": "",  # kyoto has no sound judged not visible
    }
    assert model["caption"] == {"type": "scene/open"}
    assert model["baseline"]["fascination"] == 3 and model["baseline"]["extent_coherence"] == 5
    assert (model["emphasis"]["chosen"], model["emphasis"]["tied"]) == ("extent_coherence", True)


def test_countering_it_writes_it_acousmatic_from_the_audio_caption_alone(
    clip: dict[str, Any], root: Path
) -> None:
    slots, model = steer(clip, first_step(), "counter", root)
    assert (model["recommended"], model["mode"], model["assignment"]) == ("scene", "acousmatic", "counter")
    assert slots == {
        "mode": "acousmatic",
        "caption": "버스가 공회전하는 소리",
        "sounds": "버스 소리",
        "attended_sound": "버스 소리",
        "emphasis": EMPHASIS_KO["extent_coherence"],  # the same under either mode
    }
    assert model["caption"] == {"type": "acousmatic"}
    # The scene's openness reaches no acousmatic caption (v6): its slots and key stay the same.
    enclosed = {**clip, "scene": {**clip["scene"], "openness": "enclosed"}}
    assert steer(enclosed, first_step(), "counter", root)[0] == slots


def test_the_setting_is_the_clip_s_whatever_the_point_and_a_missing_page_stops(
    clip: dict[str, Any], root: Path
) -> None:
    on_road, on_nothing = (
        steer(clip, first_step(), "follow", root),
        steer(clip, first_step(obj=None), "follow", root),
    )
    assert on_road == on_nothing
    records = first_step()
    del records["P4"]
    with pytest.raises(TweakError, match=r"no \['P4'\]"):
        steer(clip, records, "follow", root)


def test_every_caption_a_participant_can_reach_is_listed_once_per_mode_sound_and_emphasis(
    root: Path,
) -> None:
    for clip in approved_setup(root)["clips"]:
        listed = {variant_id(slots): slots for slots in reachable_slots(clip)}
        assert len(listed) == 2 * len(clip["options"]["A1s"]) * len(RESTORATIVE)
        for option in clip["options"]["A1s"]:
            for obj, point in (("road", ON_THE_ROAD), (None, (0.5, 0.1))):
                for assignment in ("follow", "counter"):
                    records = first_step(sound=option["value"], obj=obj, point=point)
                    slots, _ = steer(clip, records, assignment, root)
                    assert listed[variant_id(slots)] == slots


def test_a_source_judged_unseen_is_placed_here_unless_it_is_the_recording_s_own(root: Path) -> None:
    setup = approved_setup(root)
    lima, oslo = setup["clips"][1], setup["clips"][2]  # lima's horn and oslo's mic wind: never visible
    slots, model = steer(lima, first_step(sound="s1"), "follow", root)
    assert slots["visible_sounds"] == "차량이 오가는 소리" and model["attention"]["status"] == "hit"
    assert slots["offscreen"] == (
        " 화면에 음원이 보이지 않는 소리(경적 소리)는 "
        "화면에는 보이지 않지만 이곳 어딘가에서 들려오는 소리로 쓰세요."
    )
    assert steer(oslo, first_step(sound="s1"), "follow", root)[0]["offscreen"] == ""


def test_both_prompts_name_every_slot_in_korean_only(clip: dict[str, Any], root: Path) -> None:
    for assignment, mode in (("follow", "scene"), ("counter", "acousmatic")):
        slots, _ = steer(clip, first_step(), assignment, root)
        prompt = render_tweak(slots)
        assert prompt.template_sha256 == TWEAK_PROMPTS[mode].sha256
        for text in (
            "「버스가 공회전하는 소리」",
            "이 영상에서 들리는 소리: 버스 소리.",
            "주의를 기울인 소리: 버스 소리.",
            EMPHASIS_KO["extent_coherence"],
        ):
            assert text in prompt.text
        assert "{" not in prompt.text and not re.search(r"[A-Za-z]", prompt.text)
    assert "화면 속 대상이나 장소는 말하지 마세요" in TWEAK_PROMPTS["acousmatic"].text
    assert "그 소리를 내는 대상과만 이어 주세요." in TWEAK_PROMPTS["scene"].text
    assert "화면에 없는 대상을 지어내지 마세요" in TWEAK_PROMPTS["scene"].text
    assert (TWEAK_PROMPTS["scene"].version, TWEAK_PROMPTS["acousmatic"].version) == (4, 5)
    assert all(prompt.text.count("{emphasis}") == 1 for prompt in TWEAK_PROMPTS.values())
    assert "트인" not in TWEAK_PROMPTS["acousmatic"].text  # v4: no open or enclosed feel from the sky
    assert all("80자 이내로" in template.text for template in TWEAK_PROMPTS.values())
    assert "이 장면에 퍼져 있는 소리" not in TWEAK_PROMPTS["scene"].text and TWEAK_MAX_CHARS == 80
    assert TWEAK_PROVISIONAL is True
    with pytest.raises(ValueError, match="exactly the slots"):
        render_tweak({"mode": "scene", "caption": "x"})
    with pytest.raises(ValueError, match="mode must be one of"):
        render_tweak({"caption": "x"})


def ratings(fascination: float, away: float, coherence: float) -> dict[str, float]:
    return {
        **{item: fascination for item in ("F1", "F2", "F3")},
        **{item: away for item in ("BF1", "BF2", "BF3")},
        **{item: coherence for item in ("E1", "E2", "E3")},
    }


def test_the_emphasis_reflects_the_subscale_a_clip_stands_highest_on_for_this_participant() -> None:
    # This participant rates coherence high everywhere; centring on their own means removes that,
    # so each clip's emphasis is where it stands out for them, not their highest raw subscale.
    answers = [ratings(5, 3, 6), ratings(3, 5, 6), ratings(3, 3, 6), ratings(3, 3, 6)]
    chosen = [record["chosen"] for record in emphases(answers)]
    assert chosen == ["fascination", "being_away_from", "extent_coherence", "extent_coherence"]
    first = emphases(answers)[0]["subscales"]
    assert (first["fascination"]["centre"], first["fascination"]["centred"]) == (3.5, 1.5)
    assert first["extent_coherence"]["centred"] == 0
    # A tie of centred means goes to the higher raw mean, then to the questionnaire's order.
    flat = emphases([ratings(4, 4, 4), ratings(4, 4, 4)])
    assert [r["chosen"] for r in flat] == ["fascination"] * 2 and all(r["tied"] for r in flat)
    assert all(sentence for sentence in EMPHASIS_KO.values()) and set(EMPHASIS_KO) == set(RESTORATIVE)
    assert not any(stem in sentence for sentence in EMPHASIS_KO.values() for stem in ITEM_STEMS)
    # Each emphasis sentence carries one echo stem the gate refuses in a caption, and only its own.
    assert [
        [bool(re.search(stem, sentence)) for stem in ECHO_STEMS] for sentence in EMPHASIS_KO.values()
    ] == [
        [True, False, False],
        [False, True, False],
        [False, False, True],
    ]


def test_a_refused_caption_is_written_again_at_the_next_tau_with_what_the_gate_needs(
    clip: dict[str, Any], root: Path
) -> None:
    source, job, steer = tweak_request(clip, first_step(), "smoke", root=root)
    rungs = [tweak_job_for(clip, steer["slots"], "smoke", rung) for rung in range(len(TWEAK_LADDER))]
    assert [j.condition for _, j, _ in rungs] == list(TWEAK_LADDER) and rungs[0][1].job_id == job.job_id
    assert [j.generate["threshold"] for _, j, _ in rungs] == [0.9, 1.0, 0.7]
    assert len({j.job_id for _, j, _ in rungs}) == len(TWEAK_LADDER)  # each rung is its own call
    assert [(r["rung"], r["rungs"], r["condition"]) for _, _, r in rungs] == [
        (rung, len(TWEAK_LADDER), condition) for rung, condition in enumerate(TWEAK_LADDER)
    ]
    assert steer["gate"] == {"distractors": ["음악"], "names": ["남자의 말소리", "버스 소리", "음악"]}
    with pytest.raises(ValueError, match="rungs 0 to 2"):
        tweak_job_for(clip, steer["slots"], "smoke", len(TWEAK_LADDER))


def test_the_job_is_phase_one_at_the_live_condition_and_follows_the_mode(
    clip: dict[str, Any], root: Path
) -> None:
    source, job, record = tweak_request(clip, first_step(), "smoke", root=root)
    assert (job.phase, job.condition, job.cell) == (1, LIVE_CONDITION, None)
    assert job.spec()["generate"]["max_new_tokens"] == 128 and source.text == clip["caption"]["text"]
    assert record["intended_r2_key"] == ["m1"] and record["job_id"] == job.job_id
    assert record["template"]["mode"] == "scene"
    assert tweak_request(clip, first_step(), "smoke", root=root)[1].job_id == job.job_id
    countered = tweak_request(clip, first_step(), "smoke", root=root, assignment="counter")[1]
    assert countered.job_id != job.job_id
    with pytest.raises(PlanError, match="participant session or a smoke run"):
        tweak_request(clip, first_step(), "development", root=root)


def output(text: str, terminator: str | None = "endoftext") -> GeneratedOutput:
    return GeneratedOutput(text, text, terminator, len(text) if terminator else None, (), "", 0, (), 40)


def codes(result: Any, kind: str = "failures") -> set[str]:
    return {finding.code for finding in getattr(result, kind)}


def test_the_check_fails_only_what_cannot_be_shown(clip: dict[str, Any]) -> None:
    source = tweak_source(clip)
    three_lines = "버스가 낮게 공회전한다.\n남자가 천천히 말한다.\n바람이 분다."
    assert not check_tweak(source, output(three_lines), max_chars=TWEAK_MAX_CHARS).excluded
    for text, terminator, code in (
        ("", "endoftext", "empty_text"),
        ("버스 소리" * 40, "endoftext", "length_over_limit"),
        ("bus 소리가 들린다.", "endoftext", "non_korean_script"),
        ("버스 소리가 들�린다.", "endoftext", "undecodable_bytes"),
        ("버스 소리가 들린다.", None, "incomplete"),
    ):
        result = check_tweak(source, output(text, terminator), max_chars=TWEAK_MAX_CHARS)
        assert result.excluded and code in codes(result), (text, codes(result))
    copied = check_tweak(source, output(source.text), max_chars=TWEAK_MAX_CHARS)
    assert not copied.excluded and "identical_to_source" in codes(copied, "flags")
    assert copied.profile == "tweak-v3"


def test_a_caption_names_a_sound_by_its_words_roughly() -> None:
    assert named_in("경찰차 사이렌이 울린다.", "경찰차 사이렌 소리")
    assert named_in("사이렌 소리가 멀리서 들린다.", "경찰차 사이렌 소리")
    assert named_in("사람들의 말소리가 겹친다.", "사람들의 말소리")
    assert named_in("새가 지저귄다.", "새소리")
    assert named_in("진행자의 목소리가 들린다.", "진행자인 남자의 말소리")
    assert named_in("사람들이 크게 외치는 소리가 들린다.", "크게 외치는 소리")
    assert named_in("높은 목소리가 멀리서 크게 외쳐지며 울린다.", "크게 외치는 소리")  # conjugated
    assert named_in("차들이 오간다.", "차들이 오가는 소리")
    assert not named_in("엔진 소리만 들린다.", "음악 소리")
    # Neither a fragment of another word nor a describing word names a sound (design review, I33).
    assert not named_in("차량이 오가는 소리가 건물 사이를 지나간다.", "경찰차 사이렌 소리")
    assert not named_in("버스 엔진 소리가 사람들 사이를 지나간다.", "사람들이 이야기하는 소리")
    assert not named_in("엔진 소리가 가까이에서 들린다.", "가까이에서 말하는 여성의 목소리")
    assert not named_in("자동차, 음악이 흐른다.", "자동차 경적 소리")
    assert not named_in("사람들이 크게 외치는 소리가 들린다.", "많은 사람들이 이야기하는 소리")
    assert not named_in("새로운 소리가 들린다.", "새소리")
    assert not named_in("경적이 도로의 차량들 사이로 울린다.", "지나가는 자동차 소리")
