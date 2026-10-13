"""Calibration v2: midpoint sliders, and a tuner system message from phase-1 answers and ratings."""

from __future__ import annotations

from diff.live_calibration import calibrate, heard_families_text, rating_means
from diff.rewrite_prompts import TUNER_PROVISIONAL, TUNER_SYSTEM

OBSERVATIONS = [
    {"heard": {"human": True}, "labels": ["person", "car"]},
    {"heard": {"human": True}, "labels": ["car"]},
    {"heard": {"things": True}, "labels": []},
]
RATINGS = [{"S1": 4, "S2": 3, "S3": 5}, {"S1": 5, "S2": 2, "S3": 6}]


def test_sliders_start_at_the_midpoint_like_regen() -> None:
    calibration = calibrate(OBSERVATIONS, RATINGS)
    assert (calibration.texture, calibration.context) == (0.5, 0.5)


def test_the_tuner_carries_heard_families_and_rating_means() -> None:
    calibration = calibrate(OBSERVATIONS, RATINGS)
    assert "사람 소리 2번, 교통·기계 소리 1번" in calibration.system_message
    assert "소리 표현 4.5" in calibration.system_message and "장면과의 일치 2.5" in calibration.system_message
    assert calibration.record["caption_rating_means"] == {"S1": 4.5, "S2": 2.5, "S3": 5.5}
    assert calibration.record["tuner"]["template"] == TUNER_SYSTEM.template_id
    assert calibration.record["policy"] == "diff.live-calibration/v2"


def test_visual_labels_never_reach_the_prompt() -> None:
    # Free-text labels (often English) invite Latin letters into Korean captions.
    calibration = calibrate(OBSERVATIONS, RATINGS)
    assert "person" not in calibration.system_message and "car" not in calibration.system_message
    assert calibration.record["visual_observations"] == {"car": 2, "person": 1}


def test_missing_answers_and_ratings_read_as_not_observed() -> None:
    calibration = calibrate([], [])
    assert "없음" in calibration.system_message
    assert heard_families_text({"birdsong": 1}) == "없음"
    assert rating_means([{"S1": True}]) == {"S1": None, "S2": None, "S3": None}


def test_the_role_text_is_provisional_until_the_researcher_writes_it() -> None:
    assert calibrate([], []).provisional is TUNER_PROVISIONAL is True
