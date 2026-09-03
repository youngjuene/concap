"""§9.1: six steps in order, no way back, no way to repeat one."""

from __future__ import annotations

import pytest

from dpo.regen import progress
from dpo.regen.progress import ProgressError


class TestOrder:
    def test_a_session_with_no_snapshot_is_at_the_first_step(self) -> None:
        assert progress.current(None) == progress.VIEW_PREPARED

    def test_the_six_participant_steps_exclude_the_wait_and_the_end(self) -> None:
        assert progress.REGENERATING not in progress.PARTICIPANT_STEPS
        assert progress.DONE not in progress.PARTICIPANT_STEPS
        assert len(progress.PARTICIPANT_STEPS) == 6

    def test_advancing_walks_the_sequence_once(self) -> None:
        snapshot = {"step": progress.VIEW_PREPARED}
        seen = [snapshot["step"]]
        for step in progress.STEPS[:-1]:
            snapshot = progress.advance(snapshot, step)
            seen.append(snapshot["step"])
        assert tuple(seen) == progress.STEPS


class TestGate:
    def test_the_step_in_progress_is_allowed(self) -> None:
        progress.require({"step": progress.VISUAL}, progress.VISUAL)

    def test_a_completed_step_cannot_be_re_entered(self) -> None:
        # §8 asks the ART items again; a participant who could go back to §3
        # after the second viewing would be answering a different question.
        with pytest.raises(ProgressError, match="already complete"):
            progress.require({"step": progress.SURVEY}, progress.ART)

    def test_a_later_step_cannot_be_skipped_to(self) -> None:
        with pytest.raises(ProgressError, match="run in order"):
            progress.require({"step": progress.ART}, progress.SURVEY)

    def test_a_snapshot_naming_no_step_is_treated_as_the_start(self) -> None:
        progress.require({}, progress.VIEW_PREPARED)

    def test_an_unknown_step_is_an_error_rather_than_a_position(self) -> None:
        with pytest.raises(ProgressError, match="no step named"):
            progress.require({"step": progress.ART}, "debrief")
