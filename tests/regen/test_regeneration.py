"""§6: writing the second track from the report, and the three ways it falls back."""

from __future__ import annotations

import time
from typing import Any

from dpo.caption.writer import CaptionRequest, WriterError
from dpo.regen.captions import Cue
from dpo.regen.config import Calibration, Configuration
from dpo.regen.regeneration import RegenTemplateWriter, Report, regenerate

SLOTS = tuple(Cue(index=i, start_ms=i * 2500, end_ms=(i + 1) * 2500, text={"en": ""}) for i in range(4))
FALLBACK = tuple(
    Cue(index=i, start_ms=i * 2500, end_ms=(i + 1) * 2500, text={"en": f"Default {i}."}) for i in range(4)
)


def said(cues: Any, language: str = "en") -> list[str]:
    """What a track reads as, for the tests that care about the words."""
    return [cue.say(language) for cue in cues]


REPORT = Report(visual_labels=("Building", "Person"), auditory_labels=("Traffic", "Bird"))


def _configuration(**overrides: Any) -> Configuration:
    return Configuration(study_id="s", corpus_id="c", calibration=Calibration(cue_slots=4, **overrides))


def _run(writer: Any, **overrides: Any) -> Any:
    return regenerate(
        writer,
        _configuration(**overrides),
        clip_id="amsterdam_012",
        slots=SLOTS,
        fallback=FALLBACK,
        report=REPORT,
    )


class Slow:
    identity = "slow"

    def write(self, request: CaptionRequest) -> str:
        time.sleep(0.02)
        return "A car passes."


class Breaks:
    identity = "breaks"

    def write(self, request: CaptionRequest) -> str:
        raise WriterError("no model here")


class Empty:
    identity = "empty"

    def write(self, request: CaptionRequest) -> str:
        return "   "


class Undeclared:
    """A writer that fails in a way it never said it could."""

    identity = "undeclared"

    def write(self, request: CaptionRequest) -> str:
        raise RuntimeError("the cache file is on a full disk")


class TestSuccess:
    def test_the_template_writer_produces_a_full_track(self) -> None:
        result = _run(RegenTemplateWriter())
        assert not result.fallback
        assert len(result.cues) == 4
        assert all(said(result.cues))

    def test_generation_writes_text_into_the_fixed_slots(self) -> None:
        result = _run(RegenTemplateWriter())
        assert [(c.index, c.start_ms, c.end_ms) for c in result.cues] == [
            (c.index, c.start_ms, c.end_ms) for c in SLOTS
        ]

    def test_the_caption_is_a_function_of_the_report(self) -> None:
        # If the track does not change when the report does, this is not a
        # regeneration and the study is comparing two prepared tracks.
        other = Report(visual_labels=("Road",), auditory_labels=("Siren",))
        first = _run(RegenTemplateWriter()).cues
        second = regenerate(
            RegenTemplateWriter(),
            _configuration(),
            clip_id="amsterdam_012",
            slots=SLOTS,
            fallback=FALLBACK,
            report=other,
        ).cues
        assert said(first) != said(second)

    def test_the_slots_do_not_all_say_the_same_thing(self) -> None:
        # A track of four identical sentences is not a rehearsal of a track of
        # four different ones: the band would stop changing, and a pilot
        # participant would read something no real participant sees.
        texts = said(_run(RegenTemplateWriter()).cues)
        assert len(set(texts)) > 1

    def test_every_slot_stays_inside_the_caption_budget(self) -> None:
        wordy = Report(
            visual_labels=tuple(f"Object number {n}" for n in range(12)),
            auditory_labels=("Traffic noise, roadway noise", "Vehicle horn, car horn, honking"),
        )
        result = regenerate(
            RegenTemplateWriter(),
            _configuration(),
            clip_id="amsterdam_012",
            slots=SLOTS,
            fallback=FALLBACK,
            report=wordy,
        )
        assert not result.fallback
        assert all(len(text) <= 96 for text in said(result.cues))

    def test_an_empty_auditory_report_still_writes_a_track(self) -> None:
        # §5 allows a submit with no selection, so §6 must survive one.
        result = regenerate(
            RegenTemplateWriter(),
            _configuration(),
            clip_id="amsterdam_012",
            slots=SLOTS,
            fallback=FALLBACK,
            report=Report(visual_labels=("Building",), auditory_labels=()),
        )
        assert not result.fallback
        assert all(said(result.cues))


class TestFallback:
    def test_a_writer_that_raises_falls_back_and_says_so(self) -> None:
        result = _run(Breaks())
        assert result.fallback
        assert result.cues == FALLBACK
        assert "the writer failed on slot 0" in result.reason

    def test_a_writer_that_fails_undeclared_falls_back_rather_than_stranding_the_wait(self) -> None:
        # There is a correct track to show either way, and a participant on the
        # waiting screen has no way forward. The reason names the type, so an
        # undeclared failure is loud in the log without being loud on screen.
        result = _run(Undeclared())
        assert result.fallback
        assert result.cues == FALLBACK
        assert "RuntimeError" in result.reason and "slot 0" in result.reason

    def test_output_that_fails_validation_falls_back(self) -> None:
        result = _run(Empty())
        assert result.fallback
        assert "failed validation" in result.reason

    def test_the_latency_ceiling_falls_back_between_slots(self) -> None:
        result = _run(Slow(), latency_ceiling_ms=1)
        assert result.fallback
        assert "latency ceiling" in result.reason

    def test_the_recorded_duration_is_what_was_spent_not_the_ceiling(self) -> None:
        result = _run(Slow(), latency_ceiling_ms=1)
        assert result.duration_ms >= 1


class TestPrompt:
    def test_both_label_sets_read_as_prose_in_the_framing_sentence(self) -> None:
        # The vocabulary title-cases its labels. Left as they came, the
        # sentence read "these things in the frame: Person and Building. They
        # picked out these sounds: bus" — one list as proper nouns, the other
        # as description, from the same report.
        prompt = _run(RegenTemplateWriter()).record()["prompt"]
        assert "building and person" in prompt or "person and building" in prompt
        assert "Building" not in prompt and "Person" not in prompt

    def test_the_record_keeps_the_vocabularys_own_casing(self) -> None:
        record = _run(RegenTemplateWriter()).record()
        assert record["visual_labels"] == ["Building", "Person"]


class TestRecord:
    def test_the_record_carries_the_prompt_the_output_and_both_label_sets(self) -> None:
        record = _run(RegenTemplateWriter()).record()
        assert "building" in record["prompt"] and "traffic" in record["prompt"]
        assert len(record["raw_output"]) == 4
        assert record["visual_labels"] == ["Building", "Person"]
        assert record["auditory_labels"] == ["Traffic", "Bird"]
        assert record["fallback"] is False
        assert record["writer"]

    def test_a_fallback_record_still_carries_what_was_attempted(self) -> None:
        record = _run(Empty()).record()
        assert record["fallback"] is True
        assert record["fallback_reason"]
        assert record["raw_output"] == ["   "] * 4


class FailsOnThird:
    """Writes two slots and then gives up, the shape a fallback takes mid-run."""

    identity = "fails-on-third"

    def __init__(self) -> None:
        self.calls = 0

    def write(self, request: CaptionRequest) -> str:
        self.calls += 1
        if self.calls > 2:
            raise WriterError("the model went away")
        return "A car passes."


class TestSlotProgress:
    """§6 reports completed slots, because nothing finer than a slot is countable."""

    def test_it_reports_none_written_before_it_starts(self) -> None:
        seen: list[tuple[int, int]] = []
        regenerate(
            RegenTemplateWriter(),
            _configuration(),
            clip_id="amsterdam_012",
            slots=SLOTS,
            fallback=FALLBACK,
            report=REPORT,
            on_slot=lambda done, total: seen.append((done, total)),
        )
        # The first report is what lets a waiting screen draw an empty bar of
        # the right width rather than growing one as the model goes.
        assert seen[0] == (0, 4)
        assert seen == [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]

    def test_a_run_that_falls_back_reports_only_the_slots_it_wrote(self) -> None:
        seen: list[tuple[int, int]] = []
        result = regenerate(
            FailsOnThird(),
            _configuration(),
            clip_id="amsterdam_012",
            slots=SLOTS,
            fallback=FALLBACK,
            report=REPORT,
            on_slot=lambda done, total: seen.append((done, total)),
        )
        assert result.fallback is True
        assert seen == [(0, 4), (1, 4), (2, 4)]

    def test_a_run_that_trips_the_ceiling_still_reports_the_slot_it_paid_for(self) -> None:
        seen: list[tuple[int, int]] = []
        result = regenerate(
            Slow(),
            _configuration(latency_ceiling_ms=1),
            clip_id="amsterdam_012",
            slots=SLOTS,
            fallback=FALLBACK,
            report=REPORT,
            on_slot=lambda done, total: seen.append((done, total)),
        )
        assert result.fallback is True
        assert seen == [(0, 4), (1, 4)]

    def test_a_sink_that_raises_does_not_cost_the_participant_a_track(self) -> None:
        def hostile(done: int, total: int) -> None:
            raise RuntimeError("the terminal went away")

        result = regenerate(
            RegenTemplateWriter(),
            _configuration(),
            clip_id="amsterdam_012",
            slots=SLOTS,
            fallback=FALLBACK,
            report=REPORT,
            on_slot=hostile,
        )
        # A progress display is an observer. Nothing it does may decide whether
        # a participant gets a caption track.
        assert result.fallback is False
        assert len(result.cues) == 4
