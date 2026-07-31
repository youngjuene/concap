"""The congruency ladder: what a slider position is allowed to mean."""

from __future__ import annotations

import pytest

from dpo.evaluation.congruency import (
    CONGRUENCY_LADDER,
    ClipLadder,
    CongruencyError,
    LadderRung,
    collapsed_clips,
    generate_ladders,
    ladder_for,
    ladder_summary,
)


def test_bottom_rung_is_the_studys_own_prompt_on_audio_alone() -> None:
    """Level 0 must be the caption the trained model actually produces."""
    rungs = ladder_for("Describe what can clearly be heard.")
    assert rungs[0].level == 0.0
    assert rungs[0].conditioning == "audio"
    assert rungs[0].instruction == "Describe what can clearly be heard."
    # Every rung above it sees the frame — congruency with the real video is not
    # expressible by a model that has never been shown it.
    assert all(rung.conditioning == "audio+video" for rung in rungs[1:])
    assert [rung.level for rung in rungs] == sorted(rung.level for rung in rungs)


def test_shipped_ladder_spans_the_full_slider_range() -> None:
    rungs = ladder_for("base")
    assert rungs[0].level == 0.0 and rungs[-1].level == 1.0
    assert len(rungs) == len(CONGRUENCY_LADDER)


@pytest.mark.parametrize(
    ("rungs", "match"),
    [
        ((), "at least one rung"),
        ((LadderRung(1.0, "audio", "x"), LadderRung(0.0, "audio+video", "y")), "ascending"),
        ((LadderRung(0.0, "audio", "x"), LadderRung(0.0, "audio+video", "y")), "ascending"),
        ((LadderRung(0.0, "audio+video", "x"),), "bottom rung must be audio-only"),
    ],
)
def test_ladder_rejects_a_malformed_axis(rungs: tuple[LadderRung, ...], match: str) -> None:
    with pytest.raises(CongruencyError, match=match):
        ladder_for("base", rungs=rungs)


def test_generate_ladders_walks_every_clip_and_rung() -> None:
    rungs = ladder_for("base")
    seen: list[tuple[str, float]] = []

    def generate(clip_id: str, rung: LadderRung) -> str:
        seen.append((clip_id, rung.level))
        return f"{clip_id} at {rung.level}"

    ladders = generate_ladders(["clip-a", "clip-b"], rungs, generate)
    assert len(seen) == 2 * len(rungs)
    assert [ladder.clip_id for ladder in ladders] == ["clip-a", "clip-b"]
    assert all(len(ladder.captions) == len(rungs) for ladder in ladders)


def test_generate_ladders_rejects_an_empty_rung() -> None:
    rungs = ladder_for("base")
    with pytest.raises(CongruencyError, match="empty caption"):
        generate_ladders(["clip-a"], rungs, lambda clip_id, rung: "" if rung.level > 0 else "ok")


def test_collapsed_clips_names_sliders_that_would_do_nothing() -> None:
    """A slider whose caption never changes is a broken instrument, so it is reported."""
    same = ClipLadder(clip_id="clip-flat", captions=("identical", "identical", "identical"))
    varied = ClipLadder(clip_id="clip-ok", captions=("low", "middle", "high"))
    assert collapsed_clips([same, varied]) == ["clip-flat"]
    summary = ladder_summary([same, varied], ladder_for("base")[:3])
    assert summary["collapsed_clips"] == ["clip-flat"]
    assert summary["distinct_captions_per_clip"] == {"min": 1, "max": 3}


def test_clip_ladder_document_pairs_each_caption_with_its_level() -> None:
    rungs = ladder_for("base")[:2]
    document = ClipLadder(clip_id="clip-a", captions=("low", "high")).document(rungs)
    assert document == {
        "clip_id": "clip-a",
        "levels": [{"level": 0.0, "text": "low"}, {"level": 0.25, "text": "high"}],
    }
