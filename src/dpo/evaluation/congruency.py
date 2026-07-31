"""The audiovisual congruency ladder: one clip, one caption per slider position.

A user-study participant moves a slider and the caption changes tone — from a
description of the soundtrack that never mentions what is on screen, to one
that names the visible object and says which sound it makes. The slider
position is the study's independent variable, so the ladder it walks has to be
generated deliberately rather than discovered.

Two things vary together across the rungs, and both are load-bearing.

**Conditioning.** The bottom rung is audio-only: the model cannot see the frame,
so it cannot attribute a sound to anything visible even if asked. That rung is
the caption the study's own trained model produces under the contract's own
prompt — the thing the rest of the pipeline optimizes. Every rung above it adds
the video, because congruency with *the actual frame* is not expressible by a
model that has never seen it. Prompting an audio-only model to "name the
visible source" would make it invent one, which is a different variable
entirely — plausibility, not congruency.

**Instruction.** With the video available, the prompt controls how strongly the
caption is asked to bind sound to sight.

This is stimulus construction, not training or evaluation. It deliberately
steps outside the audio track's modality isolation, and it may only be used to
build what a participant reads — never to produce a caption that is scored,
trained on, or compared against the preference data.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class LadderRung:
    """One slider position: how the model is conditioned and what it is asked."""

    level: float
    conditioning: str  # "audio" | "audio+video"
    instruction: str

    def document(self) -> dict[str, object]:
        return {
            "level": self.level,
            "conditioning": self.conditioning,
            "instruction": self.instruction,
        }


# Code-owned, like the experiment matrix: a study may choose how many rungs to
# use, never redefine what a rung means. The instruction at level 0.0 is
# replaced at generation time by the contract's own audio prompt, so the bottom
# of every ladder is the caption the trained model actually produces.
CONGRUENCY_LADDER: tuple[LadderRung, ...] = (
    LadderRung(
        0.00,
        "audio",
        "",  # filled from the caption contract; see ladder_for()
    ),
    LadderRung(
        0.25,
        "audio+video",
        "Describe in one short English sentence what can clearly be heard."
        " Mention something visible only when the sound plainly comes from it;"
        " otherwise describe the sound alone.",
    ),
    LadderRung(
        0.50,
        "audio+video",
        "Describe in one short English sentence what can clearly be heard, naming"
        " the visible thing that makes the most prominent sound.",
    ),
    LadderRung(
        0.75,
        "audio+video",
        "Describe in one short English sentence what can clearly be heard, tying"
        " each salient sound to the thing visible in the frame that produces it.",
    ),
    LadderRung(
        1.00,
        "audio+video",
        "Describe in one short English sentence what is visible and what is heard"
        " as one event: name the things in the frame and say which sound each of"
        " them makes.",
    ),
)


class CongruencyError(ValueError):
    """Raised when a congruency ladder is degenerate or misconfigured."""


def ladder_for(
    base_prompt: str, *, rungs: Sequence[LadderRung] = CONGRUENCY_LADDER
) -> tuple[LadderRung, ...]:
    """The ladder with its bottom rung bound to the study's own caption prompt."""
    if not rungs:
        raise CongruencyError("a congruency ladder needs at least one rung")
    levels = [rung.level for rung in rungs]
    if levels != sorted(levels) or len(set(levels)) != len(levels):
        raise CongruencyError("congruency levels must be unique and ascending")
    if rungs[0].conditioning != "audio":
        raise CongruencyError(
            "the bottom rung must be audio-only: it is the caption the trained model produces"
        )
    return tuple(
        LadderRung(rung.level, rung.conditioning, base_prompt if index == 0 else rung.instruction)
        for index, rung in enumerate(rungs)
    )


@dataclass(frozen=True)
class ClipLadder:
    """Every rung's caption for one clip, ordered by ascending congruency."""

    clip_id: str
    captions: tuple[str, ...]

    def document(self, rungs: Sequence[LadderRung]) -> dict[str, object]:
        return {
            "clip_id": self.clip_id,
            "levels": [
                {"level": rung.level, "text": text} for rung, text in zip(rungs, self.captions, strict=True)
            ],
        }


def generate_ladders(
    clip_ids: Sequence[str],
    rungs: Sequence[LadderRung],
    generate: Callable[[str, LadderRung], str],
) -> tuple[ClipLadder, ...]:
    """One caption per (clip, rung), via a caller-supplied generation function.

    ``generate`` receives the clip id and the rung so the caller owns model
    loading, media resolution, and decoding; this module owns only what the
    ladder means.
    """
    if not clip_ids:
        raise CongruencyError("a congruency ladder needs at least one clip")
    ladders = []
    for clip_id in clip_ids:
        captions = tuple(generate(clip_id, rung).strip() for rung in rungs)
        if any(not text for text in captions):
            raise CongruencyError(f"clip {clip_id!r} produced an empty caption on some rung")
        ladders.append(ClipLadder(clip_id=clip_id, captions=captions))
    return tuple(ladders)


def collapsed_clips(ladders: Sequence[ClipLadder]) -> list[str]:
    """Clips whose rungs are all the same text — the slider would do nothing.

    Reported rather than raised: a genuinely uniform soundtrack can legitimately
    read the same at every congruency level, and whether that clip belongs in
    the study is the researcher's call, not the pipeline's. But a participant
    moving a slider that never changes the caption is a broken instrument, so
    the count has to be visible before anyone runs the session.
    """
    return sorted(ladder.clip_id for ladder in ladders if len(set(ladder.captions)) == 1)


def ladder_summary(ladders: Sequence[ClipLadder], rungs: Sequence[LadderRung]) -> dict[str, object]:
    distinct = [len(set(ladder.captions)) for ladder in ladders]
    return {
        "clips": len(ladders),
        "rungs": len(rungs),
        "levels": [rung.level for rung in rungs],
        "distinct_captions_per_clip": {
            "min": min(distinct) if distinct else 0,
            "max": max(distinct) if distinct else 0,
        },
        "collapsed_clips": collapsed_clips(ladders),
    }
