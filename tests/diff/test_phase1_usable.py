"""Phase 1's gate for a second caption: what it refuses, what it only flags, and what passes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from phase1_fixtures import approved_setup
from test_live_server import full_output
from test_phase1_tweak import first_step

from diff.phase1_tweak import tweak_request
from diff.phase1_usable import USABLE_POLICY, attended_named, check_record


def gate(root: Path, text: str, *, clip: int = 0, assignment: str = "follow") -> dict[str, Any]:
    """The check record for ``text`` as kyoto_101's (by default) second caption, scene-aware when followed."""
    chosen = approved_setup(root)["clips"][clip]
    source, _, steer = tweak_request(chosen, first_step(), "smoke", root=root, assignment=assignment)
    return check_record(source, steer, {"output": full_output(text)})


def codes(record: dict[str, Any], kind: str = "failures") -> list[str]:
    return [finding["code"] for finding in record[kind]]


def test_a_rewrite_that_centres_the_attended_sound_passes(tmp_path: Path) -> None:
    record = gate(tmp_path, "버스가 낮게 공회전하고, 남자가 천천히 말한다.")
    assert not record["excluded"] and codes(record) == [] and codes(record, "flags") == []
    assert record["gate"] == USABLE_POLICY and record["profile"] == "tweak-v3"
    # The screen itself is no invented place for a scene-aware caption, nor a place the prompt gave.
    assert not gate(tmp_path, "화면 속 도로에서 버스가 낮게 공회전한다.")["excluded"]
    # A noun ending after a clause is no list (the endings stay free, 2026-10-18).
    assert not gate(tmp_path, "버스가 낮게 오래 공회전하는 소리")["excluded"]
    assert not gate(tmp_path, "버스가 낮게 공회전하고, 남자가 천천히 말해요.")["excluded"]
    assert not gate(tmp_path, "버스 엔진 소리가 낮게 깔린 채, 남자가 말한다.")["excluded"]  # no cut verb


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("버스 소리가 퍼, 남자가 말한다.", "damaged_words"),  # a verb cut at a comma
        ("버스가 낮게 공회전하며.", "cut_off"),
        ("버스가 공회전하는 소리.", "near_copy"),
        ("남자가 천천히 말한다.", "attended_unnamed"),
        ("버스가 공회전하고 음악이 흐른다.", "distractor_named"),  # R2's distractor, not in the audio
        ("버스 소리와 말소리가 어우러진다.", "item_wording"),  # E2's word, before P9 asks it
        ("버스가 광장에서 공회전한다.", "place_invented"),
        ("버스가 공회전하고 버스가 공회전하고 버스가 공회전한다.", "repetition"),  # a flag of the check
        ("버스 소리, 남자의 말소리", "list_of_names"),  # names, no rewrite
        ("버스 소리가 낮게 이어져 흥미로운 느낌이다.", "item_wording"),  # F2's word
        ("버스 소리가 들리며 잠시 다른 곳에 와 있는 듯하다.", "emphasis_echo"),  # the instruction, stated
        ("버스 소리가 들리는 듯 잠시 다른 곳에 서 있다.", "emphasis_echo"),  # and paraphrased
        ("버스 소리가 이어져 잠시 다른 공간에 있는 듯하다.", "emphasis_echo"),
    ],
)
def test_the_gate_refuses_what_a_participant_must_not_be_shown(tmp_path: Path, text: str, code: str) -> None:
    record = gate(tmp_path, text)
    assert record["excluded"] and code in codes(record), codes(record)


def test_an_acousmatic_caption_may_name_no_place_on_screen(tmp_path: Path) -> None:
    text = "버스가 도로에서 낮게 공회전한다."
    assert not gate(tmp_path, text)["excluded"]  # scene-aware: the prompt gave the road
    assert codes(gate(tmp_path, text, assignment="counter")) == ["place_named"]
    # A place word inside a sound's own name is the sound, not a place (lima's "차량").
    assert not gate(tmp_path, "차량이 오가는 소리가 멀리서 끊임없이 이어진다.", clip=1, assignment="counter")[
        "excluded"
    ]


def test_relations_and_unnamed_sounds_are_flagged_never_refused(tmp_path: Path) -> None:
    record = gate(tmp_path, "버스가 하늘 너머로 낮게 공회전한다.")
    assert not record["excluded"] and codes(record, "flags") == ["relation_word"]


def test_vehicles_name_one_another_for_the_attended_sound() -> None:
    assert attended_named("자동차들이 끊임없이 오간다.", "차량이 오가는 소리")
    assert attended_named("경적이 울린다.", "경적 소리")
    assert attended_named("차가 쉼 없이 지나간다.", "도로를 달리는 차량 소리")
    assert not attended_named("차가운 바람이 분다.", "도로를 달리는 차량 소리")
    assert not attended_named("새가 지저귄다.", "차량이 오가는 소리")
