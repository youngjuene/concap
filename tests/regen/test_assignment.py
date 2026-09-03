"""§1: the condition-segment assignment, and the properties it is arithmetic for."""

from __future__ import annotations

import pytest

from dpo.regen.assignment import PREPARED, REGENERATED, AssignmentError, assign


class TestAlternation:
    def test_even_sequences_watch_a_prepared(self) -> None:
        assert assign(0).prepared_segment == "A"
        assert assign(0).regenerated_segment == "B"

    def test_odd_sequences_watch_b_prepared(self) -> None:
        assert assign(1).prepared_segment == "B"
        assert assign(1).regenerated_segment == "A"

    def test_every_prefix_of_the_run_is_balanced_to_within_one(self) -> None:
        # The reason this is parity and not a coin: a prefix of a random
        # sequence is only balanced in expectation, and a small study is a
        # prefix.
        for length in range(1, 40):
            on_a = sum(1 for sequence in range(length) if assign(sequence).prepared_segment == "A")
            assert abs(on_a - (length - on_a)) <= 1

    def test_the_two_viewings_are_never_the_same_segment(self) -> None:
        for sequence in range(20):
            assignment = assign(sequence)
            assert assignment.prepared_segment != assignment.regenerated_segment


class TestLookup:
    def test_condition_of_a_segment_is_the_inverse_of_segment_of_a_condition(self) -> None:
        for sequence in range(6):
            assignment = assign(sequence)
            for condition in (PREPARED, REGENERATED):
                assert assignment.condition_of(assignment.segment_of(condition)) == condition

    def test_a_segment_outside_the_assignment_is_an_error(self) -> None:
        with pytest.raises(AssignmentError):
            assign(0).condition_of("C")

    def test_an_unknown_condition_is_an_error(self) -> None:
        with pytest.raises(AssignmentError):
            assign(0).segment_of("control")


class TestRefusals:
    @pytest.mark.parametrize("value", [-1, "0", 1.0, True, None])
    def test_a_sequence_number_that_is_not_a_count_is_refused(self, value: object) -> None:
        with pytest.raises(AssignmentError):
            assign(value)  # type: ignore[arg-type]
