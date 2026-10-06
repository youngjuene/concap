"""The workbook is the source of questionnaire wording, bounds and provenance."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from dpo.regen.sheet_instrument import catalogue, page_item_order, page_spec, snapshot, validate_answers


def test_all_four_source_tabs_and_row_notes_remain_inspectable() -> None:
    document = catalogue()
    assert [sheet["properties"]["title"] for sheet in document["source_workbook"]["sheets"]] == [
        "안내",
        "문항표",
        "척도라벨",
        "참고문헌",
    ]
    assert len(document["references"]) == 15
    assert document["phase2"]["surveys"] == []
    assert document["phase2"]["collection"] == "interaction-logs-only"
    assert len(page_spec("P4")["items"]) == len(page_spec("P9")["items"]) == 11
    assert page_spec("P4")["items"][0]["source_row"] == "문항표!8"
    assert page_spec("P4")["items"][0]["text"] == "나는 영상 속 소리 환경이 마음에 든다"
    assert all(not item["reverse"] for item in page_spec("P4")["items"])
    assert page_spec("P9")["items"][0]["id"] == "F1_2"
    assert page_spec("P12")["kind"] == "interview"
    assert [item["id"] for item in page_spec("P12")["items"]] == ["D1", "D2"]


def test_source_labels_keep_zero_and_s3_bipolar_meaning() -> None:
    recovery = page_spec("P4")["items"][0]
    assert recovery["min"] == 0
    assert recovery["max"] == 6
    assert recovery["labels"] == [
        "전혀 그렇지 않다",
        "거의 그렇지 않다",
        "조금 그렇다",
        "중간 정도 그렇다",
        "꽤 그렇다",
        "매우 그렇다",
        "완전히 그렇다",
    ]
    s3 = page_spec("P10")["items"][2]
    assert s3["labels"] == [
        "매우 방해되었다",
        "방해되었다",
        "약간 방해되었다",
        "영향 없었다",
        "약간 도움이 되었다",
        "도움이 되었다",
        "매우 도움이 되었다",
    ]
    assert page_spec("P5")["items"][1]["labels"] == [
        "전혀 익숙하지 않다",
        "",
        "",
        "중간 정도 익숙하다",
        "",
        "",
        "완전히 익숙하다",
    ]


def test_zero_is_an_answer_and_unknown_or_partial_answers_are_rejected() -> None:
    answers = {item["id"]: 0 for item in page_spec("P4")["items"]}
    assert validate_answers("P4", answers) == answers
    with pytest.raises(ValueError, match="missing"):
        validate_answers("P4", {key: value for key, value in answers.items() if key != "F1"})
    with pytest.raises(ValueError, match="unknown"):
        validate_answers("P4", {**answers, "art_1": 5})


@pytest.mark.parametrize("value", [-1, 7, True, False, 1.5, "0", None])
def test_invalid_ratings_cannot_be_coerced_or_silently_saved(value: Any) -> None:
    with pytest.raises(ValueError, match="S1"):
        validate_answers("P10", {"S1": value, "S2": 6, "S3": 3})


def test_single_sound_choices_require_configured_video_options() -> None:
    with pytest.raises(ValueError, match="options"):
        validate_answers("P3", {"A1s": "birds"})
    options = {"A1s": [{"value": "birds", "label": "새 울음"}, {"value": "water", "label": "물소리"}]}
    assert validate_answers("P3", {"A1s": "birds"}, options=options) == {"A1s": "birds"}
    with pytest.raises(ValueError, match="A1s"):
        validate_answers("P3", {"A1s": ["birds", "water"]}, options=options)


def test_r2_unknown_option_is_exclusive_and_unconfigured_sounds_are_rejected() -> None:
    options = {"R2": ["새 울음", "물소리", "자동차 소리"]}
    assert validate_answers("P11", {"R1": 0, "R2": ["memory_unknown"]}, options=options) == {
        "R1": 0,
        "R2": ["memory_unknown"],
    }
    assert validate_answers("P11", {"R1": 6, "R2": ["새 울음", "물소리"]}, options=options)["R2"] == [
        "새 울음",
        "물소리",
    ]
    for selected in (["새 울음", "memory_unknown"], ["기차"], [], ["새 울음", "새 울음"]):
        with pytest.raises(ValueError, match="R2"):
            validate_answers("P11", {"R1": 3, "R2": selected}, options=options)


def test_p9_order_is_deterministic_for_frozen_session_and_does_not_change_p4() -> None:
    canonical = [item["id"] for item in page_spec("P9")["items"]]
    first = page_item_order("P9", "participant-a/video-1")
    assert first == page_item_order("P9", "participant-a/video-1")
    assert set(first) == set(canonical)
    assert first != canonical
    assert first != page_item_order("P9", "participant-b/video-1")
    assert [item["id"] for item in page_spec("P9", session_key="participant-a/video-1")["items"]] == first
    assert page_item_order("P4", "participant-a") == [item["id"] for item in page_spec("P4")["items"]]


def test_snapshots_are_stable_copies_and_hashes_cover_all_source_tabs() -> None:
    frozen = snapshot()
    assert frozen == snapshot()
    payload = {key: value for key, value in frozen.items() if key != "hash"}
    assert (
        frozen["hash"]
        == hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    frozen["pages"]["P4"]["items"][0]["text"] = "changed"
    assert snapshot()["pages"]["P4"]["items"][0]["text"] == "나는 영상 속 소리 환경이 마음에 든다"


def test_language_fallback_and_unresolved_material_are_not_fabricated() -> None:
    page = page_spec("P0", "en")
    assert page["language"] == "ko"
    assert page["requested_language"] == "en"
    assert page["instruction"].startswith("본 연구는")
    assert "consent_text" not in page
    assert page_spec("P13")["requires_config"] == ["debrief_text"]
    assert page_spec("P3")["items"][0]["requires_options"]


def test_visual_response_is_one_bounded_point_and_keeps_frame_reference() -> None:
    point = {"frame_id": "frame-2", "x": 0.0, "y": 1.0}
    assert validate_answers("P2", {"V1": [point]}) == {"V1": [point]}
    for points in (
        [],
        [point, point],
        [{**point, "x": -0.01}],
        [{**point, "y": float("nan")}],
        [{**point, "x": True}],
        [{**point, "x": 10**400}],
        [{"x": 0.5, "y": 0.5}],
    ):
        with pytest.raises(ValueError, match="V1"):
            validate_answers("P2", {"V1": points})


def test_second_audio_pass_reuses_same_source_video_choices() -> None:
    options = {"A1s": [{"value": "birds", "label": "새 울음"}]}
    assert page_spec("P8", options=options)["items"][0]["options"] == options["A1s"]
    assert validate_answers("P8", {"A2s": "birds"}, options=options) == {"A2s": "birds"}


def test_configured_memory_label_cannot_bypass_exclusivity() -> None:
    options = {"R2": ["새 울음", "기억나지 않음"]}
    rendered = page_spec("P11", options=options)["items"][1]["options"]
    assert rendered == [
        {"value": "새 울음", "label": "새 울음"},
        {"value": "memory_unknown", "label": "기억나지 않음", "exclusive": True},
    ]
    with pytest.raises(ValueError, match="R2"):
        validate_answers("P11", {"R1": 6, "R2": ["새 울음", "기억나지 않음"]}, options=options)


def test_oral_interview_cannot_accidentally_be_recorded_as_participant_survey() -> None:
    with pytest.raises(ValueError, match="oral interview"):
        validate_answers("P12", {"D1": "changed"})
