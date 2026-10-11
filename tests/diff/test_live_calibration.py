"""Calibration copies regen's sheet-handoff rule: midpoint sliders, guidance from heard families."""

from __future__ import annotations

from diff.live_calibration import calibrate, guidance_sentence


def test_sliders_start_at_the_midpoint_like_regen() -> None:
    calibration = calibrate([{"heard": {"human": True}, "labels": ["person"]}])
    assert (calibration.texture, calibration.context) == (0.5, 0.5)


def test_heard_families_become_one_korean_guidance_sentence() -> None:
    observations = [
        {"heard": {"human": True}, "labels": ["person", "car"]},
        {"heard": {"human": True}, "labels": ["car"]},
        {"heard": {"things": True}, "labels": []},
    ]
    calibration = calibrate(observations)
    assert "사람 소리 2번, 교통·기계 소리 1번" in calibration.guidance
    # Visual labels stay out of the prompt: Dream echoes Latin letters into captions.
    assert "person" not in calibration.guidance and "car" not in calibration.guidance
    assert calibration.record["visual_observations"] == {"car": 2, "person": 1}
    assert calibration.record["policy"] == "diff.live-calibration/v1"


def test_no_observations_means_no_guidance() -> None:
    assert calibrate([]).guidance == "" and guidance_sentence({}) == ""
    unknown = calibrate([{"heard": {"birdsong": True}, "labels": []}])
    assert unknown.guidance == "" and unknown.record["unmapped_families"] == ["birdsong"]
