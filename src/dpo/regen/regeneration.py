"""§6: writing the second viewing's caption track from what the participant said.

Between §5's submit and §7's start button, the instrument turns a report — the
objects the participant pointed at, the sources they selected — into the
caption track the other segment will carry. Everything about how that is done
is constrained by the comparison the study is making.

*Only the text is generated.* The slots and their timings come from the
segment's own track (§6, §9.4), so the regenerated track has the same number of
captions, at the same moments, for the same durations as the prepared one. If
generation could move a cue, a difference between §3 and §8 could be a
difference in when captions appeared.

*Failure is bounded and visible.* Three things can go wrong — the model can
take too long, it can return something that fails validation, it can raise —
and all three land in the same place: the segment's default fallback track,
with the reason recorded. A participant never waits past the ceiling and never
sees a blank cue. The log says `fallback: true` and why, so no analysis
silently averages fallback viewings in with generated ones.

*The latency ceiling is checked between slots, not around the whole call.* A
slot that is already running cannot be interrupted without abandoning the model
mid-decode, so the ceiling is enforced at the only points where stopping is
clean. A run that trips it has still spent one slot's time past the ceiling,
and the recorded duration says so rather than reporting the ceiling.

*A slot is the finest honest unit of progress.* ``on_slot`` reports completed
slots and nothing else, because nothing else here is countable. A decode gives
no usable fraction of itself: ``max_new_tokens`` is a cap rather than a target,
so a token counter runs to some arbitrary point and jumps, and one slot can be
two model calls anyway — the shared writer retries a caption that came back too
long, and the retry is what ``gemma-tightened`` in the record means. Slots are
also the granularity the ceiling is already reasoned at, so a progress report
and a timeout can never disagree about where the run had got to.

*What is recorded is the whole thing.* The assembled prompt for every slot, the
writer's returned string for every slot before this module strips or validates
it, the final track, both label sets, the duration, the fallback flag, and the
writer's identity with its settings. The shared writer's own retries are inside
its returned :class:`~dpo.caption.writer.Written`, whose ``writer`` field says
which path produced the text — ``gemma``, ``gemma-tightened``, ``gemma-cut``,
``template-fallback`` — so a caption that came from the model's second try is
distinguishable from one that came from its first.

Section numbers cite ``docs/v3-regen/spec-behavior.md``.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dpo.caption.writer import (
    CAPTION_MAX_CHARS,
    CaptionRequest,
    CaptionWriter,
    SourceSpec,
    WriterError,
    Written,
    as_sentence,
    join_clauses,
    writer_identity,
    written_by,
)
from dpo.regen.captions import Cue, TrackError, retimed, validate_generated
from dpo.regen.config import Configuration

TASK = "regenerated"
LEVEL = "regen"

# The system text. Assembled per slot from the participant's own report, which
# is the whole point: a caption track that does not change when the report
# changes is not a regeneration, and the study would be comparing one prepared
# track against another.
REGEN_PREAMBLE = (
    "You write one line of a sound caption for a ten-second street scene. "
    "The line is shown over the footage while it plays, so it is read in a "
    "moment and never re-read. One sentence, at most {chars} characters. "
    "Write it in {language}, the language the viewer is reading, and in no "
    "other.\n"
)
# What the preamble calls the language. A tag is what the configuration holds
# and what validation checks; a model is told the name.
LANGUAGE_NAMES: Mapping[str, str] = {
    "en": "English",
    "ko": "Korean",
    "ja": "Japanese",
    "zh": "Chinese",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "nl": "Dutch",
    "th": "Thai",
}
REGEN_FRAMING = (
    "A viewer has watched a different segment of this scene and said what they "
    "noticed. They pointed at these things in the frame: {seen}. They picked "
    "out these sounds: {heard}. Write for that viewer: name what they attend "
    "to, in their terms, and add nothing they did not mention.\n"
)
REGEN_SLOT = (
    "This is line {position} of {total}, covering seconds {start:.1f} to {end:.1f}. "
    "Describe what can be heard in that window and nothing outside it."
)
# The report can be empty on the auditory side — §5 allows a submit with no
# selection — and a prompt that says "these sounds: " with nothing after it
# reads as a truncation to a model. Said in words instead.
NOTHING_HEARD = "nothing in particular"
NOTHING_SEEN = "nothing they could name"


def _listed(values: Sequence[str], empty: str) -> str:
    """``join_clauses`` indexes its argument, so an empty report is words here."""
    return join_clauses(list(values)) if values else empty


class RegenerationError(RuntimeError):
    """The regeneration could not run at all, before any slot was attempted."""


@dataclass(frozen=True)
class Report:
    """What §4 and §5 produced, in the order the participant produced it.

    ``visual_labels`` has already had the unclassified points removed (§6): a
    point that matched no mask carries no label to condition on. The count of
    what was dropped is in the §4 record, not here.
    """

    visual_labels: tuple[str, ...]
    auditory_labels: tuple[str, ...]
    auditory_ids: tuple[str, ...] = ()

    def sources(self) -> tuple[SourceSpec, ...]:
        """The report as the shared writer's source specs.

        The auditory selections are the sources: they are what the caption is
        about. The visual labels reach the model through the framing sentence
        rather than as sources, because a caption that names a building the
        viewer looked at but could not hear is describing the picture.
        """
        ids = self.auditory_ids or tuple(label.lower().replace(" ", "_") for label in self.auditory_labels)
        return tuple(
            SourceSpec(id=source_id, token=label, prose=label.lower(), phrases=("", ""))
            for source_id, label in zip(ids, self.auditory_labels, strict=True)
        )


@dataclass(frozen=True)
class Regeneration:
    """The result, and everything §6 logs about how it was reached."""

    cues: tuple[Cue, ...]
    fallback: bool
    reason: str
    prompts: tuple[str, ...]
    raw: tuple[str, ...]
    writers: tuple[str, ...]
    duration_ms: int
    writer_identity: str
    language: str = "en"
    settings: Mapping[str, Any] = field(default_factory=dict)
    report: Report = field(default_factory=lambda: Report((), ()))

    def record(self) -> dict[str, Any]:
        """§6's log line. Nothing here is summarised."""
        return {
            "prompt": "\n\n".join(self.prompts),
            "raw_output": list(self.raw),
            "writers": list(self.writers),
            "track": [cue.record(self.language) for cue in self.cues],
            "language": self.language,
            "visual_labels": list(self.report.visual_labels),
            "auditory_labels": list(self.report.auditory_labels),
            "duration_ms": self.duration_ms,
            "fallback": self.fallback,
            "fallback_reason": self.reason,
            "writer": self.writer_identity,
            "settings": dict(self.settings),
        }


class RegenRequestBuilder:
    """Turns a report and a slot into the request the shared writer takes.

    ``language`` is the one the session is being read in, not the study's
    first: a bilingual study generates for the participant in front of it.
    """

    def __init__(self, configuration: Configuration, language: str | None = None) -> None:
        self.configuration = configuration
        self.language = language or configuration.calibration.language

    def instruction(self, request: CaptionRequest) -> str:
        """The exact system text for a request, assembled from the constants.

        Read back out of the request rather than closed over, so the string the
        model is given is a function of the request alone — which is what makes
        it safe to record one prompt per slot and call it the prompt.
        """
        heard = _listed([spec.prose for spec in request.sources], NOTHING_HEARD)
        return (
            REGEN_PREAMBLE.format(
                chars=self.configuration.calibration.slot_max_chars,
                language=LANGUAGE_NAMES.get(self.language.split("-")[0].lower(), self.language),
            )
            + REGEN_FRAMING.format(seen=request.scene_prose or NOTHING_SEEN, heard=heard)
            + request.atmosphere_prose
        )

    def request(
        self, clip_id: str, cue: Cue, total: int, report: Report, media: Path | None
    ) -> CaptionRequest:
        # Lowercased like the auditory prose beside it. The labels are common
        # nouns from a vocabulary that title-cases them; left as they came,
        # the framing sentence read "these things in the frame: Person and
        # Building. They picked out these sounds: bus", which invites a model
        # to treat one list as proper nouns and the other as description. The
        # record keeps the vocabulary's own casing; only the prose changes.
        seen = _listed([label.lower() for label in report.visual_labels], NOTHING_SEEN)
        slot = REGEN_SLOT.format(
            position=cue.index + 1, total=total, start=cue.start_ms / 1000, end=cue.end_ms / 1000
        )
        return CaptionRequest(
            clip_id=clip_id,
            shot_id=f"cue{cue.index}",
            task=TASK,
            level=LEVEL,
            # Identical settings must return the identical caption, and the
            # settings here are the report plus the slot. Two participants who
            # reported the same thing get the same track, which is a property
            # the study wants: the caption is a function of the report.
            # The language is in the key: the same report in two languages is
            # two captions, and a cache that could not tell them apart would
            # serve one participant the other's reading.
            settings_key="|".join(
                (
                    str(cue.index),
                    self.language,
                    ",".join(report.visual_labels),
                    ",".join(report.auditory_labels),
                )
            ),
            sources=report.sources(),
            heads=(),
            scene_prose=seen,
            atmosphere_prose=slot,
            media_path=media,
        )


def slot_of(shot_id: str) -> int:
    """The cue index out of a request's shot id.

    :class:`RegenRequestBuilder` writes ``cue<index>`` and the template writer
    reads it back. The contract is between two classes in this module and is
    stated in both, rather than being smuggled through ``level``, which the
    shared writer uses for something else.
    """
    return int(shot_id.removeprefix("cue")) if shot_id.startswith("cue") else 0


class RegenTemplateWriter:
    """Deterministic prose from the report; no model, no randomness.

    What the instrument is built, tested and demonstrated on, and what §6 falls
    back to when a model is not attached at all. It conditions on the same two
    lists the model does, so a template track and a Gemma track move in the
    same direction when the report changes — which is what makes a template run
    a real rehearsal of the study rather than a fixed string.

    It also varies across the slots, for the same reason. A track of four
    identical sentences is not a rehearsal of a track of four different ones:
    the caption band would stop changing, and a pilot participant would be
    reading something no real participant ever sees. The lead rotates through
    the reported sources and the sentence takes one of three shapes, so a
    four-slot track reads as four captions while staying a function of the
    report alone.
    """

    identity = "template"

    def write_attributed(self, request: CaptionRequest) -> Written:
        """Says who wrote the caption, as every writer the log records does.

        Without this the template's slots reach §6's record as ``unknown``,
        and a demo run would be indistinguishable in the log from a model run
        whose writer declined to say — which is the one distinction the
        ``writers`` field exists to make.
        """
        return Written(self.write(request), self.identity)

    def write(self, request: CaptionRequest) -> str:
        heard = [spec.prose for spec in request.sources]
        position = slot_of(request.shot_id)
        if not heard:
            return as_sentence(f"{request.scene_prose}, and little to hear")
        lead = heard[position % len(heard)]
        rest = [name for name in heard if name != lead]
        if not rest or position == 0:
            written = as_sentence(f"{lead} over {request.scene_prose}")
        elif position % 2:
            written = as_sentence(f"{lead}, with {join_clauses(rest)} behind")
        else:
            written = as_sentence(f"{join_clauses(rest)} under {lead}")
        if len(written) <= CAPTION_MAX_CHARS:
            return written
        # The scene is what a long report inflates; the sources are what the
        # caption is about, so the scene is what goes.
        return as_sentence(join_clauses(heard))[:CAPTION_MAX_CHARS]


def regenerate(
    writer: CaptionWriter,
    configuration: Configuration,
    *,
    clip_id: str,
    slots: Sequence[Cue],
    fallback: Sequence[Cue],
    report: Report,
    language: str | None = None,
    media: Path | None = None,
    settings: Mapping[str, Any] | None = None,
    on_slot: Callable[[int, int], None] | None = None,
) -> Regeneration:
    """Write one track for ``slots``, or fall back and say why (§6).

    ``slots`` supplies the timings — in practice the segment's own prepared
    track, whose cues are the fixed slots — ``language`` is the one the
    participant is reading, and ``fallback`` is the segment's default track,
    used whole if anything goes wrong. Both have already been
    validated against the configuration by the document loader, so the fallback
    is never itself a risk.

    ``on_slot`` is called with the number of slots written and the number
    there are, once before the first and once after each one lands. It is an
    observer: it is called inside a try, because nothing a progress display
    does may decide whether a participant gets a caption track.
    """
    reading = language or configuration.calibration.language
    builder = RegenRequestBuilder(configuration, reading)
    ceiling = configuration.calibration.latency_ceiling_ms
    started = time.monotonic()
    prompts: list[str] = []
    raw: list[str] = []
    writers: list[str] = []
    reason = ""

    def elapsed_ms() -> int:
        return int((time.monotonic() - started) * 1000)

    def announce(done: int) -> None:
        if on_slot is None:
            return
        with contextlib.suppress(Exception):  # an observer may not decide the run
            on_slot(done, len(slots))

    announce(0)
    for position, cue in enumerate(slots):
        request = builder.request(clip_id, cue, len(slots), report, media)
        prompts.append(builder.instruction(request))
        try:
            written = written_by(writer, request)
        except WriterError as exc:
            reason = f"the writer failed on slot {cue.index}: {exc}"
            break
        except Exception as exc:  # noqa: BLE001 — see below
            # Anything the writer did not declare. Caught with the declared
            # failure rather than left to reach the participant, because
            # there is a correct caption track to show either way and a
            # participant stranded on the waiting screen has no way forward.
            # The reason names the type, so an undeclared failure is loud in
            # the log without being loud on the screen.
            reason = f"the writer raised {type(exc).__name__} on slot {cue.index}: {exc}"
            break
        raw.append(written.caption)
        writers.append(written.writer)
        # After the slot lands and before the ceiling is read, so a run that
        # trips the ceiling still reports the slot it finished paying for.
        announce(position + 1)
        if elapsed_ms() > ceiling:
            reason = f"the latency ceiling of {ceiling}ms was exceeded after slot {cue.index}"
            break
    else:
        try:
            cues = retimed(slots, raw, reading)
            validate_generated(cues, configuration.calibration, reading)
        except TrackError as exc:
            reason = f"the generated track failed validation: {exc}"
        else:
            return Regeneration(
                cues=cues,
                fallback=False,
                reason="",
                prompts=tuple(prompts),
                raw=tuple(raw),
                writers=tuple(writers),
                duration_ms=elapsed_ms(),
                writer_identity=writer_identity(writer),
                language=reading,
                settings=dict(settings or {}),
                report=report,
            )

    return Regeneration(
        cues=tuple(fallback),
        fallback=True,
        reason=reason,
        prompts=tuple(prompts),
        raw=tuple(raw),
        writers=tuple(writers),
        duration_ms=elapsed_ms(),
        writer_identity=writer_identity(writer),
        language=reading,
        settings=dict(settings or {}),
        report=report,
    )
