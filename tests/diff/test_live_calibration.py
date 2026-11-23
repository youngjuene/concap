"""Calibration v5: counterbalanced slider starts per video, and one fixed tuner sentence for everyone."""

from __future__ import annotations

from collections import Counter

from diff.live_calibration import ORDERS, calibrate, heard_families_text, rating_means, start_orders
from diff.rewrite_prompts import TUNER_PROVISIONAL, TUNER_SYSTEM

OBSERVATIONS = [
    {"heard": {"human": True}, "labels": ["person", "car"]},
    {"heard": {"human": True}, "labels": ["car"]},
    {"heard": {"things": True}, "labels": []},
]
RATINGS = [{"S1": 4, "S2": 3, "S3": 5}, {"S1": 5, "S2": 2, "S3": 6}]


def test_each_slider_starts_at_every_stop_once_over_the_three_videos() -> None:
    for number in range(1, 49):
        starts = start_orders(f"P{number}")
        assert starts is not None and len(starts) == 3
        for axis in (0, 1):
            assert sorted(start[axis] for start in starts) == [0.0, 0.5, 1.0]
    calibration = calibrate(OBSERVATIONS, RATINGS, "P1")
    assert (calibration.texture, calibration.context) == calibration.start(0) == calibration.starts[0]
    assert calibration.record["counterbalanced"] is True
    assert [tuple(s.values()) for s in calibration.record["starts"]] == list(calibration.starts)


def test_the_orders_are_balanced_in_every_block_of_six_and_unrelated_across_sliders() -> None:
    def orders(numbers: range) -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
        found = []
        for n in numbers:
            starts = start_orders(f"P{n}")
            assert starts is not None
            found.append((tuple(s[0] for s in starts), tuple(s[1] for s in starts)))
        return found

    for block in range(8):
        six = orders(range(6 * block + 1, 6 * block + 7))
        assert {texture for texture, _ in six} == set(ORDERS) == {context for _, context in six}
    assert len(set(orders(range(1, 37)))) == 36  # every pair of orders once in 36 participants
    first_video = (
        Counter(o[0][0] for o in orders(range(1, 49))),
        Counter(o[1][0] for o in orders(range(1, 49))),
    )
    assert all(set(c.values()) == {16} for c in first_video)  # N = 48: each stop first 16 times


def test_without_a_participant_number_both_sliders_start_at_the_midpoint() -> None:
    assert start_orders(None) is None and start_orders("smoke") is None
    calibration = calibrate(OBSERVATIONS, RATINGS)
    assert calibration.starts == ((0.5, 0.5),) * 3 and calibration.record["counterbalanced"] is False
    assert calibration.start(7) == (0.5, 0.5)


def test_the_tuner_is_one_fixed_sentence_and_phase_one_reaches_no_prompt() -> None:
    # Chain spec, 2026-11-23: phase 1 reaches phase 2 only as the predictions, never through the captions.
    one, other = calibrate(OBSERVATIONS, RATINGS, "P1"), calibrate([], [], "P2")
    assert one.system_message == other.system_message == TUNER_SYSTEM.text and "{" not in TUNER_SYSTEM.text
    assert "사람 소리" not in one.system_message and "4.5" not in one.system_message
    assert "person" not in one.system_message
    # The profile is still recorded, for description.
    assert one.record["caption_rating_means"] == {"S1": 4.5, "S2": 2.5, "S3": 5.5}
    assert one.record["visual_observations"] == {"car": 2, "person": 1}
    assert one.record["tuner"]["template"] == TUNER_SYSTEM.template_id and TUNER_SYSTEM.version == 1
    assert one.record["policy"] == "diff.live-calibration/v5"
    assert "scene_profile" not in one.record  # a cue's mode comes from the masks alone


def test_missing_answers_and_ratings_read_as_not_observed() -> None:
    assert heard_families_text({"birdsong": 1}) == "없음"
    assert rating_means([{"S1": True}]) == {"S1": None, "S2": None, "S3": None}


def test_the_role_text_is_provisional_until_the_researcher_approves_it() -> None:
    assert calibrate([], []).provisional is TUNER_PROVISIONAL is True
