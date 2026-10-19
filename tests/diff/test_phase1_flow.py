"""Phase 1 in diff: the sheet's items and rules, two viewings per clip, and a blind page payload."""

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
from diff.phase1_flow import (
    PREPARING,
    STEP_A,
    STEP_B,
    UNWATCHED,
    Phase1,
    checkpoint,
    clip_order,
    record_playback,
)
from diff.phase1_setup import stripped_options
from diff.session_store import Conflict
from dpo.regen.study_store import Conflict as RegenConflict

RATED = ("F1", "F2", "F3", "BF1", "BF2", "BF3", "E1", "E2", "E3", "P1", "P2")
ANSWERS: dict[str, dict[str, Any]] = {
    "P3": {"A1s": "s1"},
    "P4": {item: 4 for item in RATED},
    "P5": {"FA1": "no", "FA2": 3, "FA3": 5},
    "P8": {"A2s": "s2"},
    "P9": {f"{item}_2": 5 for item in RATED},
    "P10": {"S1": 5, "S2": 2, "S3": 6},
    "P11": {"R1": 6, "R2": ["m1"]},
}
POINTS = {"P2": "V1", "P7": "V2"}


@pytest.fixture
def flow(tmp_path: Path) -> Phase1:
    return Phase1(approved_setup(tmp_path), root=tmp_path)


def watch(flow: Phase1, state: dict[str, Any], now: float) -> float:
    page = state["page"]
    for sequence, position in enumerate(range(0, DURATION_MS + 1, 1000)):
        now += 1.0
        data = {"page": page, "position_ms": position, "playing": True, "hidden": False, "sequence": sequence}
        flow.apply(state, "playback", data, now)
    flow.apply(state, "view-ended", {"page": page}, now)
    return now


def answer(flow: Phase1, state: dict[str, Any], now: float, point: tuple[float, float] = (0.5, 0.9)) -> None:
    page = state["page"]
    if page in POINTS:
        x, y = point
        answers: dict[str, Any] = {POINTS[page]: [{"frame_id": "0", "x": x, "y": y}]}
    else:
        answers = ANSWERS[page]
    flow.apply(state, "submit", {"page": page, "answers": answers, "submitted_at": now}, now)


def step(flow: Phase1, state: dict[str, Any], now: float, point: tuple[float, float] = (0.5, 0.9)) -> float:
    """One clip's step: its viewing, then the step's pages."""
    now = watch(flow, state, now)
    for _ in range(len(STEP_A if state["page"] in STEP_A else STEP_B) - 1):
        answer(flow, state, now, point)
    return now


def written(flow: Phase1, state: dict[str, Any], status: str = "ready") -> None:
    """Trida's outcome for every pending second caption, as the server writes it back."""
    for key, _, attempt in flow.pending_tweaks(state):
        outcome = {"status": status, "attempt": attempt, "result_status": "completed", "job_id": "job-x"}
        if status == "ready":
            outcome["text"] = f"두 번째 자막 {state['clip_order'].index(key)}"
        else:
            outcome["reason"] = "check failed"
        flow.record_tweak(state, key, outcome, 50.0)


def through_stage_a(flow: Phase1, state: dict[str, Any], now: float) -> float:
    """Every clip's first step, then the stage-B interstitial."""
    for _ in range(len(state["clip_order"])):
        now = step(flow, state, now)
    assert (state["step"], state["clip_index"], state["page"]) == ("B", 0, "interlude")
    flow.apply(state, "interlude-complete", {"page": "interlude"}, now)
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
    now = through_stage_a(flow, state, 10.0)
    assert (state["step"], state["clip_index"], state["page"]) == ("B", 0, "P6")
    first = state["clip_order"][0]
    expected = flow.clips[first]["options"]["A1s"][0]["label"]
    assert state["second_captions"][first]["slots"]["attended_sound"] == expected
    waiting = flow.present(state, lambda name: name)
    assert waiting["kind"] == "preparing" and waiting["instruction"] == PREPARING and "media" not in waiting
    written(flow, state)
    shown = flow.present(state, lambda name: name)
    assert shown["kind"] == "view" and shown["media"]["captions"][0]["text"] == "두 번째 자막 0"
    for _ in range(3):
        now = step(flow, state, now, (0.5, 0.1))
    assert state["page"] == "handoff"
    pages = [r["page"] for r in state["responses"]]
    assert pages == [*(list(STEP_A[1:]) * 3), *(list(STEP_B[1:]) * 3)]
    assert [v["page"] for v in state["viewings"]] == ["P1"] * 3 + ["P6"] * 3
    assert (
        state["viewings"][3]["caption_source"] == "trida"
        and state["viewings"][0]["caption_source"] == "prepared"
    )
    views = {(r["page"], r["view_id"]) for r in state["responses"] if r["clip_index"] == 0}
    assert {view for page, view in views if page in STEP_A} == {"s-test-0-P1"}
    assert {view for page, view in views if page in STEP_B} == {"s-test-0-P6"}
    p2, p7 = (next(r for r in state["responses"] if r["page"] == page) for page in ("P2", "P7"))
    assert p2["labels"] == ["Road"] and p7["labels"] == ["Sky"]
    p9 = next(r for r in state["responses"] if r["page"] == "P9")
    assert p9["presented_order"] == [
        i["id"] for i in workbook_instrument.page_spec("P9", session_key="s-test")["items"]
    ]
    observations, ratings = flow.profile_inputs(state)
    family = flow.clips[first]["options"]["A1s"][0]["family"]
    assert observations[0] == {"heard": {family: True}, "labels": ["Road"]}
    assert ratings == [{"S1": 5, "S2": 2, "S3": 6}] * 3


def test_a_failed_second_caption_skips_that_clip_s_second_step(flow: Phase1) -> None:
    state = flow.new_state("s-test", "P1", seed=7)
    through_practice(flow, state)
    now = through_stage_a(flow, state, 10.0)
    written(flow, state, "failed")
    assert flow.present(state, lambda name: name)["kind"] == "skipped"
    assert flow.media_files(state) == {}
    with pytest.raises(Conflict, match="no viewing"):
        flow.apply(state, "view-ended", {"page": "P6"}, now)
    for index in range(3):
        flow.apply(state, "skip-second", {"page": "P6"}, now)
        assert state["skipped"][index]["reason"] == "check failed"
    assert state["page"] == "handoff"
    observations, ratings = flow.profile_inputs(state)
    assert len(observations) == 3 and ratings == []
    with pytest.raises(Conflict, match="not skipped"):
        state.update(page="P6", clip_index=0, second_captions={})
        flow.apply(state, "skip-second", {"page": "P6"}, now)


def test_one_retry_is_kept_across_restarts_and_late_outcomes_are_ignored(flow: Phase1) -> None:
    state = flow.new_state("s-test", "P1", seed=7)
    through_practice(flow, state)
    step(flow, state, 10.0)
    ((key, slots, attempt),) = flow.pending_tweaks(state)
    assert attempt == 1 and slots == state["second_captions"][key]["slots"]
    retry = {"status": "retry", "attempt": 1, "result_status": "worker_lost", "reason": "worker_lost: x"}
    flow.record_tweak(state, key, retry, 20.0)
    assert flow.pending_tweaks(state) == [(key, slots, 2)]
    ready = {"status": "ready", "attempt": 2, "result_status": "completed", "job_id": "j", "text": "새 자막"}
    flow.record_tweak(state, key, ready, 21.0)
    flow.record_tweak(state, key, {**ready, "text": "늦게 온 자막"}, 22.0)
    entry = state["second_captions"][key]
    assert entry["status"] == "ready" and entry["text"] == "새 자막" and len(entry["attempts"]) == 2
    assert flow.pending_tweaks(state) == []


def test_every_page_says_where_it_sits_in_the_session(flow: Phase1) -> None:
    state = flow.new_state("s-test", "P1", seed=7)

    def where(live_minutes: tuple[int, ...] = ()) -> tuple[str, str, dict[str, Any]]:
        payload = flow.present(state, lambda name: name, live_minutes)
        return payload["page"], payload["step"], payload["progress"]

    assert where() == ("P0", "A", {"segment": 0, "fraction": 0.0})
    with pytest.raises(Conflict, match="not been reached"):
        flow.apply(state, "interlude-complete", {"page": "P0"}, 1.0)
    through_practice(flow, state)
    assert where() == ("P1", "A", {"segment": 2, "fraction": 0.0})
    now = watch(flow, state, 10.0)
    assert where() == ("P2", "A", {"segment": 2, "fraction": 0.067})  # P2 of clip 1 of 3: (0 + 1/5) / 3
    for _ in range(len(STEP_A) - 1):
        answer(flow, state, now)
    for _ in range(2):
        now = step(flow, state, now)
    # The stage-B interstitial: phase 1's one break, before the clip count restarts at 1.
    assert where() == ("interlude", "B", {"segment": 3, "fraction": 0.0})
    interlude = flow.present(state, lambda name: name, (5, 6))
    assert (interlude["kind"], interlude["title"]) == ("interlude", "첫 번째 시청을 모두 마쳤습니다")
    assert interlude["instruction"].startswith("이제 앞에서 본 영상 3편을 한 번 더 봅니다.")
    assert "자막" not in interlude["instruction"]  # P0 already says the captions differ (D1)
    assert interlude["remaining"] == [
        {"label": "두 번째 시청 · 영상 3편", "time": None},
        {"label": "긴 영상 2편 · 자막 조절", "time": "약 11분"},
        {"label": "짧은 인터뷰", "time": "2~3분"},
    ]
    without_videos = flow.present(state, lambda name: name)["remaining"]
    assert without_videos[1] == {"label": "긴 영상 · 자막 조절", "time": None}
    assert flow.media_files(state) == {}
    event = flow.apply(state, "interlude-complete", {"page": "interlude"}, now)
    assert (event["phase"], event["analysis_excluded"], event["clip_index"]) == (0, True, None)
    assert where() == ("P6", "B", {"segment": 3, "fraction": 0.0})


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


def test_page_texts_are_the_sheet_s_word_for_word(flow: Phase1) -> None:
    state = flow.new_state("s-test", "P1", seed=1)
    titles = {}
    for page in ("P0", "P2", "P10", "P11"):
        state["page"] = page
        titles[page] = flow.present(state, lambda name: name)["title"]
    assert titles == {
        "P0": "도입",
        "P2": "1차 시각 지각",
        "P10": "2차 자막 평가",
        "P11": "2차 자막 읽음 확인",
    }
    state["page"] = "P0"
    intro = flow.present(state, lambda name: name)
    assert intro["instruction"] == workbook_instrument.page_spec("P0")["instruction"]
    assert "두 번" in intro["instruction"] and "인터뷰" in intro["instruction"]
    assert intro["consent_text"] is None


def test_no_clip_key_family_answer_key_or_prompt_reaches_the_browser(flow: Phase1) -> None:
    state = flow.new_state("s-test", "P1", seed=3)
    through_practice(flow, state)
    payloads = [flow.present(state, lambda name: f"media/{name}")]
    now = 10.0
    while state["page"] != "handoff":
        if state["page"] == "P6" and flow.pending_tweaks(state):
            payloads.append(flow.present(state, lambda name: f"media/{name}"))
            written(flow, state)
        payloads.append(flow.present(state, lambda name: f"media/{name}"))
        if state["page"] in ("P1", "P6"):
            now = watch(flow, state, now)
        elif state["page"] == "interlude":
            flow.apply(state, "interlude-complete", {"page": "interlude"}, now)
        else:
            answer(flow, state, now, (0.2, 0.2))
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
        "slots",
        "fascination",
        "job-x",
        ".png",
        ".mp4",
    ):
        assert secret not in text, secret
    assert payloads[1]["media"]["url"] == "media/video"
    assert payloads[2]["frames"][2]["url"] == "media/frame-2"
    assert sum(payload["kind"] == "preparing" for payload in payloads) == 1


def test_only_the_current_page_s_media_are_served(flow: Phase1) -> None:
    state = flow.new_state("s-test", "P1", seed=3)
    assert flow.media_files(state) == {}
    through_practice(flow, state)
    assert set(flow.media_files(state)) == {"video"}
    for page, expected in (("P2", {"frame-0", "frame-1", "frame-2"}), ("P3", set()), ("P6", set())):
        state["page"] = page
        assert set(flow.media_files(state)) == expected
    state["second_captions"][flow.clip(state)["key"]] = {"status": "ready", "text": "자막", "attempts": []}
    assert set(flow.media_files(state)) == {"video"}
    state["page"] = "P7"
    assert set(flow.media_files(state)) == {"frame-0", "frame-1", "frame-2"}


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
