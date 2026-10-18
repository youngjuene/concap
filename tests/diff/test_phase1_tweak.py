"""Phase 1's second caption: the slots from first-step answers, the prompt, the job and its check."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from phase1_fixtures import approved_setup

from diff.job_plan import PlanError
from diff.output_checks import GeneratedOutput, check_tweak
from diff.phase1_tweak import NOT_SEEN, TweakError, degree, tweak_request, tweak_slots, tweak_source
from diff.rewrite_prompts import TWEAK_PROVISIONAL, TWEAK_SLOTS, render_tweak
from diff.study_protocol import LIVE_CONDITION, TWEAK_MAX_CHARS

RATINGS = ("F1", "F2", "F3", "BF1", "BF2", "BF3", "E1", "E2", "E3", "P1", "P2")


def first_step(
    *, sound: str = "s2", obj: str | None = "road", fascination: int = 6, rest: int = 3
) -> dict[str, dict[str, Any]]:
    ratings = {item: (fascination if item.startswith("F") else rest) for item in RATINGS}
    return {
        "P2": {"answers": {"V1": []}, "matches": [{"object_id": obj}]},
        "P3": {"answers": {"A1s": sound}},
        "P4": {"answers": ratings},
        "P5": {"answers": {"FA1": "no", "FA2": 0, "FA3": 6}},
    }


@pytest.fixture
def clip(tmp_path: Path) -> dict[str, Any]:
    return approved_setup(tmp_path)["clips"][0]  # kyoto_101


def test_the_slots_carry_the_answers_and_the_degrees(clip: dict[str, Any]) -> None:
    slots = tweak_slots(clip, first_step())
    assert set(slots) == set(TWEAK_SLOTS)
    assert slots == {
        "caption": "버스가 공회전하는 소리",
        "sounds": "버스 소리",
        "attended_sound": "남자의 말소리",
        "attended_object": "도로",
        "fascination": "100",
        "being_away": "50",
        "coherence": "50",
        "familiarity": "0",
    }
    assert degree([0, 6]) == 50 and degree([5, 4, 6]) == 83


def test_a_point_on_no_object_and_a_missing_page(clip: dict[str, Any]) -> None:
    assert tweak_slots(clip, first_step(obj=None))["attended_object"] == NOT_SEEN
    records = first_step()
    del records["P4"]
    with pytest.raises(TweakError, match=r"no \['P4'\]"):
        tweak_slots(clip, records)


def test_the_prompt_names_every_slot_in_korean_only(clip: dict[str, Any]) -> None:
    prompt = render_tweak(tweak_slots(clip, first_step()))
    for text in (
        "「버스가 공회전하는 소리」",
        "이 영상에서 들리는 소리: 버스 소리.",
        "「남자의 말소리」",
        "「도로」",
        ": 100%",
    ):
        assert text in prompt.text
    assert not re.search(r"[A-Za-z]", prompt.text)
    assert TWEAK_PROVISIONAL is True
    with pytest.raises(ValueError, match="exactly the slots"):
        render_tweak({"caption": "x"})


def test_the_job_is_phase_one_at_the_live_condition_and_follows_the_answers(clip: dict[str, Any]) -> None:
    source, job, record = tweak_request(clip, first_step(), "smoke")
    assert (job.phase, job.condition, job.cell) == (1, LIVE_CONDITION, None)
    assert job.spec()["generate"]["max_new_tokens"] == 128 and source.text == clip["caption"]["text"]
    assert record["intended_r2_key"] == ["m1"] and record["job_id"] == job.job_id
    assert tweak_request(clip, first_step(), "smoke")[1].job_id == job.job_id
    assert tweak_request(clip, first_step(fascination=0), "smoke")[1].job_id != job.job_id
    with pytest.raises(PlanError, match="participant session or a smoke run"):
        tweak_request(clip, first_step(), "development")


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
