"""Audiovisual congruency as a measured axis, and the slider ladder built on it.

A participant drags a handle and the caption changes tone — from a description
of the soundtrack that never mentions what is on screen, to one that names the
visible object and says which sound it makes. The handle's position is the
study's independent variable, so the axis it walks has to be a quantity, not an
assumption.

**The measure.** For a caption ``c`` on one clip::

    congruency(c) = [ log P(c | audio, video) - log P(c | audio) ] / |c|

in nats per token: how much does *seeing* the clip help explain this sentence?
A caption describing sound alone gains nothing from the video and scores near
zero. One that names the thing visibly making the sound scores strongly
positive. One that names something not on screen scores NEGATIVE — the video
makes those words less likely — which is a principled incongruent end rather
than a staged one.

This reuses the pipeline's own log-probability primitive, the same one that
scores preference pairs. Nothing here re-defines what likelihood means.

**Why measured rather than prompted.** Writing five instructions of increasing
"tie sound to sight" strength and declaring that ordering to be the axis is an
assumption no output verifies: the model may well answer rung three less
congruently than rung two, leaving the study's independent variable silently
unordered. So candidates are over-generated across conditionings and wordings,
each scored, and the ladder is SELECTED from them by measured congruency — with
non-monotone or imperceptibly narrow ladders refused rather than shipped.

**Why the video is needed at all.** Congruency is defined against what is
actually in frame, so the upper rungs must condition on the video. An
audio-only model asked to "name the visible source" invents one, which measures
plausibility instead. That makes ladder construction a deliberate step outside
the audio track's modality isolation — stimulus building only, never scoring,
training, or preference comparison.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

# Enough spread that the selector has a real range to choose rungs from, without
# making generation cost balloon: every extra spec is one generation plus two
# scoring passes per clip.
CANDIDATE_SPECS: tuple[tuple[str, str, float], ...] = (
    # (conditioning, instruction, temperature). The empty instruction is filled
    # from the caption contract, so the study's own trained prompt is always a
    # candidate and normally lands at the bottom of the axis.
    ("audio", "", 0.0),
    ("audio", "", 0.7),
    (
        "audio",
        "Describe in one short English sentence only what can be heard, without"
        " referring to anything that might be visible.",
        0.0,
    ),
    (
        "audio+video",
        "Describe in one short English sentence what can clearly be heard."
        " Mention something visible only when the sound plainly comes from it.",
        0.0,
    ),
    (
        "audio+video",
        "Describe in one short English sentence what can clearly be heard, naming"
        " the visible thing that makes the most prominent sound.",
        0.0,
    ),
    (
        "audio+video",
        "Describe in one short English sentence what can clearly be heard, tying"
        " each salient sound to the thing visible in the frame that produces it.",
        0.0,
    ),
    (
        "audio+video",
        "Describe in one short English sentence what is visible and what is heard"
        " as one event: name the things in the frame and say which sound each of"
        " them makes.",
        0.0,
    ),
    (
        "audio+video",
        "Describe in one short English sentence what is visible and what is heard"
        " as one event: name the things in the frame and say which sound each of"
        " them makes.",
        0.7,
    ),
)

# A ladder whose ends differ by less than this is not a slider a participant can
# feel: dragging it would change wording without changing meaning. In nats per
# token, on captions of roughly twenty tokens.
MIN_CONGRUENCY_SPAN = 0.02


@dataclass(frozen=True)
class CandidateSpec:
    """One over-generation setting: how to condition, what to ask, how to sample."""

    conditioning: str  # "audio" | "audio+video"
    instruction: str
    temperature: float


@dataclass(frozen=True)
class ScoredCaption:
    """One candidate caption with its measured congruency, in nats per token."""

    text: str
    congruency: float
    conditioning: str


@dataclass(frozen=True)
class ClipLadder:
    """The selected rungs for one clip, ascending in measured congruency."""

    clip_id: str
    rungs: tuple[ScoredCaption, ...]

    @property
    def span(self) -> float:
        return self.rungs[-1].congruency - self.rungs[0].congruency if self.rungs else 0.0

    def positions(self) -> tuple[float, ...]:
        """Each rung's place on a 0..1 slider, proportional to measured congruency.

        The control's geometry is the measurement: rungs that differ little sit
        close together, so the distance a participant drags between two captions
        is the distance between them on the axis, not an artefact of how many
        candidates happened to be generated.
        """
        if not self.rungs:
            return ()
        low = self.rungs[0].congruency
        span = self.span
        if span <= 0:
            return tuple(index / max(1, len(self.rungs) - 1) for index in range(len(self.rungs)))
        return tuple((rung.congruency - low) / span for rung in self.rungs)

    def document(self) -> dict[str, object]:
        return {
            "clip_id": self.clip_id,
            "congruency_span": self.span,
            "levels": [
                {
                    "position": position,
                    "congruency": rung.congruency,
                    "conditioning": rung.conditioning,
                    "text": rung.text,
                }
                for rung, position in zip(self.rungs, self.positions(), strict=True)
            ],
        }


class CongruencyError(ValueError):
    """Raised when a congruency ladder is degenerate or misconfigured."""


def candidate_specs(base_prompt: str) -> tuple[CandidateSpec, ...]:
    """The over-generation grid, with empty instructions bound to the contract's."""
    return tuple(
        CandidateSpec(conditioning, instruction or base_prompt, temperature)
        for conditioning, instruction, temperature in CANDIDATE_SPECS
    )


def select_rungs(scored: Sequence[ScoredCaption], count: int) -> tuple[ScoredCaption, ...]:
    """``count`` distinct captions, ascending and as evenly spread as the data allows.

    Even spacing on the MEASURED axis, not on the candidate list: the goal is
    that consecutive rungs feel equally far apart to a participant, which is a
    property of congruency, not of generation order.
    """
    if count < 2:
        raise CongruencyError("a slider needs at least two rungs")
    unique: dict[str, ScoredCaption] = {}
    for candidate in sorted(scored, key=lambda item: item.congruency):
        unique.setdefault(candidate.text, candidate)
    ordered = sorted(unique.values(), key=lambda item: item.congruency)
    if len(ordered) < count:
        raise CongruencyError(
            f"only {len(ordered)} distinct captions available; need {count} rungs"
            " — widen CANDIDATE_SPECS or lower the rung count"
        )
    low, high = ordered[0].congruency, ordered[-1].congruency
    if high - low < MIN_CONGRUENCY_SPAN:
        raise CongruencyError(
            f"congruency span {high - low:.4f} nats/token is below {MIN_CONGRUENCY_SPAN}:"
            " every candidate explains this clip equally well with and without the video,"
            " so a slider over them would change wording without changing meaning"
        )
    chosen: list[ScoredCaption] = [ordered[0]]
    remaining = list(ordered[1:-1])
    for step in range(1, count - 1):
        target = low + (high - low) * step / (count - 1)
        if not remaining:
            break
        best = min(remaining, key=lambda item: abs(item.congruency - target))
        remaining.remove(best)
        chosen.append(best)
    chosen.append(ordered[-1])
    chosen.sort(key=lambda item: item.congruency)
    if len(chosen) != count:
        raise CongruencyError(f"selected {len(chosen)} rungs, expected {count}")
    return tuple(chosen)


def build_ladders(
    clip_ids: Sequence[str],
    specs: Sequence[CandidateSpec],
    *,
    rungs: int,
    generate: Callable[[str, CandidateSpec], str],
    measure: Callable[[str, str], float],
) -> tuple[ClipLadder, ...]:
    """Over-generate, measure congruency, and select the rungs, per clip.

    ``generate(clip_id, spec) -> caption`` and ``measure(clip_id, caption) ->
    nats/token`` are injected so this module owns only what the axis means; the
    caller owns model loading, media resolution, and decoding.
    """
    if not clip_ids:
        raise CongruencyError("a congruency ladder needs at least one clip")
    if not specs:
        raise CongruencyError("over-generation needs at least one candidate spec")
    ladders = []
    for clip_id in clip_ids:
        scored: list[ScoredCaption] = []
        for spec in specs:
            text = generate(clip_id, spec).strip()
            if not text:
                continue
            scored.append(
                ScoredCaption(text=text, congruency=measure(clip_id, text), conditioning=spec.conditioning)
            )
        if not scored:
            raise CongruencyError(f"clip {clip_id!r} produced no usable candidate captions")
        ladders.append(ClipLadder(clip_id=clip_id, rungs=select_rungs(scored, rungs)))
    return tuple(ladders)


def ladder_summary(ladders: Sequence[ClipLadder]) -> dict[str, object]:
    spans = [ladder.span for ladder in ladders]
    return {
        "clips": len(ladders),
        "rungs": len(ladders[0].rungs) if ladders else 0,
        "congruency_span": {
            "min": min(spans) if spans else 0.0,
            "max": max(spans) if spans else 0.0,
        },
        "negative_congruency_clips": sorted(
            ladder.clip_id for ladder in ladders if ladder.rungs[0].congruency < 0
        ),
    }
