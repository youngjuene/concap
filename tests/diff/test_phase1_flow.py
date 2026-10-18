"""Phase 1 in diff: regen's items and rules, one viewing per clip, and a blind page payload."""

from __future__ import annotations

import json
import random
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from phase1_fixtures import DURATION_MS, approved_setup

import diff.workbook_instrument as workbook_instrument
import dpo.regen.playback as regen_playback
from diff.phase1_flow import CLIP_PAGES, UNWATCHED, Phase1, checkpoint, clip_order, record_playback
from diff.phase1_setup import stripped_options
from diff.session_store import Conflict
from dpo.regen.study_store import Conflict as RegenConflict

ANSWERS: dict[str, dict[str, Any]] = {
    "P3": {"A1s": "s1"},
    "P4": {item: 4 for item in ("F1", "F2", "F3", "BF1", "BF2", "BF3", "E1", "E2", "E3", "P1", "P2")},
    "P5": {"FA1": "no", "FA2": 3, "FA3": 5},
    "P10": {"S1": 5, "S2": 2, "S3": 6},
    "P11": {"R1": 6, "R2": ["m1"]},
}


@pytest.fixture
def flow(tmp_path: Path) -> Phase1:
    return Phase1(approved_setup(tmp_path), root=tmp_path)


def watch(flow: Phase1, state: dict[str, Any], now: float) -> float:
    for sequence, position in enumerate(range(0, DURATION_MS + 1, 1000)):
        now += 1.0
        data = {"page": "P1", "position_ms": position, "playing": True, "hidden": False, "sequence": sequence}
        flow.apply(state, "playback", data, now)
    flow.apply(state, "view-ended", {"page": "P1"}, now)
    return now


def complete_clip(flow: Phase1, state: dict[str, Any], now: float, point: tuple[float, float]) -> float:
    now = watch(flow, state, now)
    x, y = point
    flow.apply(state, "submit", {"page": "P2", "answers": {"V1": [{"frame_id": "0", "x": x, "y": y}]}}, now)
    for page in ("P3", "P4", "P5", "P10", "P11"):
        flow.apply(state, "submit", {"page": page, "answers": ANSWERS[page], "submitted_at": now}, now)
    return now


def through_practice(flow: Phase1, state: dict[str, Any]) -> None:
    flow.apply(state, "intro-complete", {"page": "P0"}, 1.0)
    answers = {
        "practice_rating": 3,
        "practice_sound": "s1",
        "practice_visual": [{"frame_id": "0", "x": 0.5, "y": 0.5}],
    }
    flow.apply(state, "practice-complete", {"page": "practice", "answers": answers}, 2.0)


def test_a_whole_phase_one_walk(flow: Phase1) -> None:
    state = flow.new_state("s-test", "P1", seed=7)
    assert state["clip_order"] == clip_order(["kyoto_101", "lima_202", "oslo_303"], 7)
    through_practice(flow, state)
    assert state["practice"][0]["analysis_excluded"] is True and state["page"] == "P1"
    now = 10.0
    for _ in range(3):
        now = complete_clip(flow, state, now, (0.5, 0.9))
    assert state["page"] == "handoff"
    assert [r["page"] for r in state["responses"]] == [p for p in CLIP_PAGES if p != "P1"] * 3
    assert len(state["viewings"]) == 3 and state["viewings"][0]["coverage"] == [[0.0, 10000.0]]
    assert {r["view_id"] for r in state["responses"] if r["clip_index"] == 0} == {"s-test-0-P1"}
    p2 = state["responses"][0]
    assert p2["matches"][0]["object_id"] == "road" and p2["labels"] == ["Road"]
    observations, ratings = flow.profile_inputs(state)
    first = state["clip_order"][0]
    family = flow.clips[first]["options"]["A1s"][0]["family"]
    assert observations[0] == {"heard": {family: True}, "labels": ["Road"]}
    assert ratings == [{"S1": 5, "S2": 2, "S3": 6}] * 3


def test_items_and_rules_are_regen_s(flow: Phase1) -> None:
    state = flow.new_state("s-test", "P1", seed=1)
    state.update(stage="phase1", page="P11")
    shown = flow.present(state, lambda name: name)
    clip = flow.clip(state)
    options = {"A1s": stripped_options(clip["options"]["A1s"]), "R2": stripped_options(clip["options"]["R2"])}
    sheet = workbook_instrument.page_spec("P11", options=options, session_key="s-test")
    assert [item["id"] for item in shown["items"]] == ["R1", "R2"]
    assert shown["items"][1]["options"] == [
        {"value": "m1", "label": "버스 소리"},
        {"value": "m2", "label": "음악"},
        {"value": "memory_unknown", "label": "기억나지 않음", "exclusive": True},
    ]
    assert [item["text"] for item in shown["items"]] == [item["text"] for item in sheet["items"]]
    for page, wrong, message in (
        ("P11", {"R1": 6, "R2": ["m1", "memory_unknown"]}, "exclusive"),
        ("P4", {"F1": 7}, "missing answers"),
        ("P10", {"S1": True, "S2": 1, "S3": 1}, "rating must be an integer"),
    ):
        state["page"] = page
        with pytest.raises(ValueError, match=message):
            flow.apply(state, "submit", {"page": page, "answers": wrong}, 1.0)


def test_page_titles_use_the_approved_wording(flow: Phase1) -> None:
    state = flow.new_state("s-test", "P1", seed=1)
    titles = {}
    for page in ("P0", "P2", "P10", "P11"):
        state["page"] = page
        titles[page] = flow.present(state, lambda name: name)["title"]
    assert titles == {"P0": "도입", "P2": "시각 지각", "P10": "자막 평가", "P11": "자막 읽음 확인"}
    state["page"] = "P0"
    intro = flow.present(state, lambda name: name)
    assert "두 번" not in intro["instruction"] and "인터뷰" not in intro["instruction"]
    assert intro["consent_text"] is None


def test_no_clip_key_family_or_answer_key_reaches_the_browser(flow: Phase1) -> None:
    state = flow.new_state("s-test", "P1", seed=3)
    through_practice(flow, state)
    payloads = [flow.present(state, lambda name: f"media/{name}")]
    now = 10.0
    for page in CLIP_PAGES:
        payloads.append(flow.present(state, lambda name: f"media/{name}"))
        if page == "P1":
            now = watch(flow, state, now)
        elif page == "P2":
            flow.apply(
                state,
                "submit",
                {"page": "P2", "answers": {"V1": [{"frame_id": "1", "x": 0.2, "y": 0.2}]}},
                now,
            )
        else:
            flow.apply(state, "submit", {"page": page, "answers": ANSWERS[page]}, now)
    text = json.dumps(payloads, ensure_ascii=False)
    for secret in (
        "kyoto_101",
        "lima_202",
        "oslo_303",
        "family",
        "mentioned",
        "audioset",
        "orig_phrase",
        "orig_text",
        "a bus idles",
        "things",
        "human",
        ".png",
        ".mp4",
    ):
        assert secret not in text, secret
    assert payloads[1]["media"]["url"] == "media/video"
    assert payloads[2]["frames"][2]["url"] == "media/frame-2"


def test_only_the_current_page_s_media_are_served(flow: Phase1) -> None:
    state = flow.new_state("s-test", "P1", seed=3)
    assert flow.media_files(state) == {}
    through_practice(flow, state)
    assert set(flow.media_files(state)) == {"video"}
    state["page"] = "P2"
    assert set(flow.media_files(state)) == {"frame-0", "frame-1", "frame-2"}
    state["page"] = "P3"
    assert flow.media_files(state) == {}


def test_the_clip_must_be_watched_to_the_end(flow: Phase1) -> None:
    state = flow.new_state("s-test", "P1", seed=3)
    through_practice(flow, state)
    for sequence, position in enumerate((0, 1000, 2000)):
        tick = {"page": "P1", "position_ms": position, "playing": True, "sequence": sequence}
        flow.apply(state, "playback", tick, 10.0 + sequence)
    with pytest.raises(Conflict, match=UNWATCHED):
        flow.apply(state, "view-ended", {"page": "P1"}, 20.0)
    with pytest.raises(Conflict, match="out of order"):
        flow.apply(
            state, "playback", {"page": "P1", "position_ms": 3000, "playing": True, "sequence": 1}, 21.0
        )


def test_answers_belong_to_their_page(flow: Phase1) -> None:
    state = flow.new_state("s-test", "P1", seed=3)
    with pytest.raises(Conflict, match="another page"):
        flow.apply(state, "submit", {"page": "P3", "answers": {"A1s": "s1"}}, 1.0)
    with pytest.raises(ValueError, match="three practice responses"):
        flow.apply(state, "intro-complete", {"page": "P0"}, 1.0)
        flow.apply(state, "practice-complete", {"page": "practice", "answers": {"practice_rating": 3}}, 2.0)


def test_consent_is_required_once_consent_text_exists(tmp_path: Path) -> None:
    setup = approved_setup(tmp_path)
    setup["consent_text"] = "연구 참여에 동의합니다."
    flow = Phase1(setup, root=tmp_path)
    state = flow.new_state("s-test", "P1", seed=3)
    with pytest.raises(Conflict, match="consent"):
        flow.apply(state, "intro-complete", {"page": "P0"}, 1.0)
    flow.apply(state, "intro-complete", {"page": "P0", "consented": True}, 1.0)
    assert state["consent"]["accepted"] is True and state["consent"]["text_sha256"]


def test_the_handoff_closes_phase_one(flow: Phase1) -> None:
    state = flow.new_state("s-test", "P1", seed=3)
    with pytest.raises(Conflict, match="short clips"):
        flow.hand_off(state, {"sha256": "x"}, 5.0)
    state["page"] = "handoff"
    event = flow.hand_off(state, {"sha256": "abc"}, 5.0)
    assert state["stage"] == "phase2" and event["calibration_sha256"] == "abc"
    with pytest.raises(Conflict, match="complete"):
        flow.present(state, lambda name: name)


def _outcome(run: Any) -> str:
    try:
        run()
    except (Conflict, RegenConflict):
        return "conflict"
    except ValueError:
        return "invalid"
    return "ok"


def test_the_coverage_rule_matches_regen_s(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = random.Random(20261018)
    for _ in range(300):
        ours, theirs = checkpoint(), checkpoint()
        now, position, sequence = 1000.0, 0.0, 0
        for _ in range(rng.randint(3, 14)):
            now += rng.choice([0.2, 0.5, 1.0, 1.0, 1.0, 3.0, 9.0])
            position = min(10000.0, max(0.0, position + rng.choice([-500, 0, 800, 1000, 1000, 2500, 8000])))
            sequence += rng.choice([1, 1, 1, 0])
            tick = {
                "position_ms": position if rng.random() > 0.03 else True,
                "playing": rng.random() > 0.2,
                "hidden": rng.random() < 0.1,
                "seek": rng.random() < 0.05,
                "sequence": sequence,
            }
            monkeypatch.setattr(regen_playback, "time", SimpleNamespace(time=lambda at=now: at))
            mine = _outcome(partial(record_playback, ours, tick, 10000, now))
            regen = _outcome(partial(regen_playback.record_playback, theirs, dict(tick), 10000))
            assert mine == regen
            assert ours == theirs
