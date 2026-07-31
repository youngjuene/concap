"""Congruency as a measured axis: what a slider stop is allowed to mean."""

from __future__ import annotations

import pytest

from dpo.evaluation.congruency import (
    MIN_CONGRUENCY_SPAN,
    CandidateSpec,
    ClipLadder,
    CongruencyError,
    ScoredCaption,
    build_ladders,
    candidate_specs,
    ladder_summary,
    select_rungs,
)


def _scored(*pairs: tuple[str, float]) -> list[ScoredCaption]:
    return [ScoredCaption(text=text, congruency=value, conditioning="audio") for text, value in pairs]


def test_candidate_specs_bind_the_empty_instruction_to_the_contract_prompt() -> None:
    specs = candidate_specs("Describe what can clearly be heard.")
    assert specs[0].instruction == "Describe what can clearly be heard."
    assert any(spec.conditioning == "audio+video" for spec in specs)
    # The study's own trained prompt is always in the pool, so the caption the
    # model actually produces is always a candidate for a rung.
    assert sum(1 for spec in specs if spec.instruction == "Describe what can clearly be heard.") >= 1


def test_select_rungs_returns_an_ascending_evenly_spread_ladder() -> None:
    scored = _scored(("a", 0.0), ("b", 0.05), ("c", 0.10), ("d", 0.5), ("e", 1.0))
    chosen = select_rungs(scored, 3)
    values = [rung.congruency for rung in chosen]
    assert values == sorted(values)
    assert values[0] == 0.0 and values[-1] == 1.0
    # The middle rung is the candidate nearest the midpoint of the MEASURED
    # range, not the middle of the candidate list.
    assert chosen[1].congruency == 0.5


def test_select_rungs_refuses_an_axis_a_participant_could_not_feel() -> None:
    flat = _scored(("a", 0.0), ("b", MIN_CONGRUENCY_SPAN / 4), ("c", MIN_CONGRUENCY_SPAN / 2))
    with pytest.raises(CongruencyError, match="below"):
        select_rungs(flat, 3)


def test_select_rungs_needs_enough_distinct_captions() -> None:
    with pytest.raises(CongruencyError, match="distinct captions"):
        select_rungs(_scored(("same", 0.0), ("same", 1.0)), 3)
    with pytest.raises(CongruencyError, match="at least two rungs"):
        select_rungs(_scored(("a", 0.0), ("b", 1.0)), 1)


def test_positions_place_stops_where_the_measure_put_them() -> None:
    ladder = ClipLadder(clip_id="clip-a", rungs=tuple(_scored(("low", 0.0), ("mid", 0.8), ("high", 1.0))))
    # 0.8 of the way up the range, not the middle: a drag between two captions
    # covers their distance on the axis, not an artefact of the rung count.
    assert ladder.positions() == (0.0, 0.8, 1.0)
    assert ladder.span == pytest.approx(1.0)


def test_build_ladders_over_generates_then_selects() -> None:
    specs = [
        CandidateSpec("audio", "heard only", 0.0),
        CandidateSpec("audio+video", "name the source", 0.0),
        CandidateSpec("audio+video", "bind sound to sight", 0.0),
    ]
    # A synthetic model whose captions get more visually grounded per spec, and
    # a measure that reports exactly that.
    scores = {"heard only": 0.0, "name the source": 0.4, "bind sound to sight": 0.9}
    ladders = build_ladders(
        ["clip-a", "clip-b"],
        specs,
        rungs=3,
        generate=lambda clip_id, spec: spec.instruction,
        measure=lambda clip_id, caption: scores[caption],
    )
    assert [ladder.clip_id for ladder in ladders] == ["clip-a", "clip-b"]
    for ladder in ladders:
        values = [rung.congruency for rung in ladder.rungs]
        assert values == sorted(values) == [0.0, 0.4, 0.9]


def test_build_ladders_rejects_a_clip_with_no_usable_candidate() -> None:
    with pytest.raises(CongruencyError, match="no usable candidate"):
        build_ladders(
            ["clip-a"],
            [CandidateSpec("audio", "x", 0.0)],
            rungs=2,
            generate=lambda clip_id, spec: "   ",
            measure=lambda clip_id, caption: 0.0,
        )


def test_summary_reports_span_and_the_incongruent_end() -> None:
    """A negative bottom rung is the principled incongruent end, and worth naming."""
    negative = ClipLadder(clip_id="clip-neg", rungs=tuple(_scored(("wrong", -0.3), ("right", 0.6))))
    positive = ClipLadder(clip_id="clip-pos", rungs=tuple(_scored(("plain", 0.1), ("bound", 0.5))))
    summary = ladder_summary([negative, positive])
    assert summary["clips"] == 2 and summary["rungs"] == 2
    assert summary["negative_congruency_clips"] == ["clip-neg"]
    assert summary["congruency_span"]["max"] == pytest.approx(0.9)
