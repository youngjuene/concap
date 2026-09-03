"""§9.6: one track schema, and the validation both conditions pass through."""

from __future__ import annotations

import pytest

from dpo.regen.captions import (
    Cue,
    TrackError,
    in_language,
    record_of,
    retimed,
    validate_generated,
    validate_track,
)
from dpo.regen.config import Calibration


def cues(count: int = 4, text: str = "A car passes.", **more: str) -> tuple[Cue, ...]:
    """A track whose slots read ``text`` in English, plus any other language given."""
    said = {"en": text, **more}
    return tuple(
        Cue(index=i, start_ms=i * 2500, end_ms=(i + 1) * 2500, text=dict(said)) for i in range(count)
    )


class TestSlotCount:
    def test_a_track_of_the_configured_length_passes(self) -> None:
        validate_track(cues(4), Calibration(cue_slots=4), path="prepared")

    def test_a_track_of_another_length_is_refused_by_slot_count(self) -> None:
        # §9.4's whole point: the prepared and regenerated tracks are held to
        # one number, so caption density cannot confound provenance.
        with pytest.raises(TrackError, match="exactly 4 cue slots"):
            validate_track(cues(3), Calibration(cue_slots=4), path="prepared")


class TestTimings:
    def test_a_cue_that_ends_before_it_starts_is_refused(self) -> None:
        bad = (Cue(0, 1000, 500, {"en": "x"}), *cues(4)[1:])
        with pytest.raises(TrackError, match="end_ms must be after start_ms"):
            validate_track(bad, Calibration(cue_slots=4), path="t")

    def test_overlapping_cues_are_refused(self) -> None:
        overlapping = (
            Cue(0, 0, 3000, {"en": "a"}),
            Cue(1, 2000, 5000, {"en": "b"}),
            Cue(2, 5000, 7000, {"en": "c"}),
            Cue(3, 7000, 9000, {"en": "d"}),
        )
        with pytest.raises(TrackError, match="before the previous cue ends"):
            validate_track(overlapping, Calibration(cue_slots=4), path="t")

    def test_a_shuffled_track_is_refused_by_its_indices(self) -> None:
        shuffled = tuple(reversed(cues(4)))
        with pytest.raises(TrackError, match=r"\[0\].index"):
            validate_track(shuffled, Calibration(cue_slots=4), path="t")


class TestLimits:
    def test_text_past_the_character_cap_is_refused(self) -> None:
        long = cues(4, text="x" * 200)
        with pytest.raises(TrackError, match="at most 96 characters"):
            validate_track(long, Calibration(cue_slots=4), path="t")

    def test_text_past_the_line_cap_is_refused(self) -> None:
        many = cues(4, text="a\nb\nc")
        with pytest.raises(TrackError, match="at most 2 lines"):
            validate_track(many, Calibration(cue_slots=4), path="t")


class TestGenerated:
    def test_an_empty_slot_is_refused(self) -> None:
        # A failure only generation can produce; an authored track's empty
        # string never reaches here, the document schema stops it first.
        with pytest.raises(TrackError, match="returned nothing"):
            validate_generated(retimed(cues(4), ["a", "", "c", "d"], "en"), Calibration(cue_slots=4), "en")

    def test_text_in_another_script_is_refused_when_the_study_names_one(self) -> None:
        korean = Calibration(cue_slots=4, languages=("ko",))
        with pytest.raises(TrackError, match="not in the language on screen"):
            validate_generated(retimed(cues(4), ["a car passes"] * 4, "ko"), korean, "ko")

    def test_the_study_s_own_script_passes(self) -> None:
        korean = Calibration(cue_slots=4, languages=("ko",))
        validate_generated(retimed(cues(4), ["차가 지나간다."] * 4, "ko"), korean, "ko")

    def test_generation_is_checked_in_the_language_on_screen_not_the_studys_first(self) -> None:
        # A bilingual study generates for the participant in front of it. A
        # Korean track validated against the study's English would be refused
        # for being in the language it was asked for.
        both = Calibration(cue_slots=4, languages=("en", "ko"))
        validate_generated(retimed(cues(4), ["차가 지나간다."] * 4, "ko"), both, "ko")
        with pytest.raises(TrackError, match="not in the language on screen"):
            validate_generated(retimed(cues(4), ["a car passes"] * 4, "ko"), both, "ko")

    def test_a_latin_script_study_makes_no_language_claim_it_cannot_support(self) -> None:
        # English and French are one script; guessing between them on 96
        # characters would reject good captions, so the check passes.
        assert in_language("une voiture passe", "en")
        assert in_language("a car passes", "fr")


class TestRetiming:
    def test_generation_writes_text_and_never_moves_a_slot(self) -> None:
        slots = cues(4)
        written = retimed(slots, ["one", "two", "three", "four"], "en")
        assert [cue.say("en") for cue in written] == ["one", "two", "three", "four"]
        assert [(c.index, c.start_ms, c.end_ms) for c in written] == [
            (c.index, c.start_ms, c.end_ms) for c in slots
        ]

    def test_the_wrong_number_of_texts_is_refused(self) -> None:
        with pytest.raises(TrackError, match="expected 4 slot texts"):
            retimed(cues(4), ["one"], "en")

    def test_the_record_carries_text_with_its_timings(self) -> None:
        assert record_of(cues(1), "en")[0] == {
            "index": 0,
            "start_ms": 0,
            "end_ms": 2500,
            "text": "A car passes.",
            "language": "en",
        }

    def test_the_record_says_which_language_was_read(self) -> None:
        # §2 and §7 log the text a participant actually read. A row that
        # carried the text without the tag would leave which language it was
        # to be guessed from the characters.
        row = record_of(cues(1, ko="차가 지나간다."), "ko")[0]
        assert (row["text"], row["language"]) == ("차가 지나간다.", "ko")


class TestLanguages:
    def test_a_slot_missing_a_language_the_study_offers_is_refused(self) -> None:
        both = Calibration(cue_slots=4, languages=("en", "ko"))
        with pytest.raises(TrackError, match="no text in"):
            validate_track(cues(4), both, path="prepared")

    def test_a_slot_carrying_both_passes(self) -> None:
        both = Calibration(cue_slots=4, languages=("en", "ko"))
        validate_track(cues(4, ko="차가 지나간다."), both, path="prepared")

    def test_every_language_is_held_to_the_same_budget(self) -> None:
        both = Calibration(cue_slots=4, languages=("en", "ko"))
        with pytest.raises(TrackError, match=r"text\[ko\]"):
            validate_track(cues(4, ko="가" * 200), both, path="prepared")

    def test_a_cue_asked_for_a_language_it_does_not_carry_says_which_it_has(self) -> None:
        with pytest.raises(TrackError, match="has \\['en'\\]"):
            cues(1)[0].say("ko")
