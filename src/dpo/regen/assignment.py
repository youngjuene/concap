"""§1: who watches which segment under which condition, decided once on entry.

The instrument compares two caption conditions over two segments of one scene.
Which segment carries which condition has to alternate, or condition and
segment are the same variable and the study measures neither.

The rule is parity of the participant's sequence number: even sequence numbers
get the prepared track on segment A, odd ones get it on segment B. Two
properties follow, and both are why this is arithmetic rather than a draw.

*A run of participants is balanced by construction.* Any prefix of the sequence
is balanced to within one, where a coin would only be balanced in expectation
and a small study is exactly where that difference bites.

*The analysis can reconstruct the assignment from the sequence number alone.*
Nothing has to be stored and kept consistent with the log; a record that names
its sequence number names its assignment. The log stores it anyway, because a
record should be readable without rerunning the code that wrote it, but the two
are checkable against each other.

The order of *viewing* is not a second variable. §2 is always the prepared
condition and §7 always the regenerated one, because the regenerated track is
generated from what the participant reports in §4 and §5, which they cannot do
before they have watched anything. Only the segment moves.

Section numbers cite ``docs/v3-regen/spec-behavior.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

PREPARED = "prepared"
REGENERATED = "regenerated"
SEGMENTS = ("A", "B")


class AssignmentError(ValueError):
    """A sequence number or a segment that cannot be assigned."""


@dataclass(frozen=True)
class Assignment:
    """One participant's fixed condition-segment pairing.

    ``first`` is the segment watched under the prepared track on §2, ``second``
    the segment watched under the regenerated track on §7. The two are always
    different segments, which is what makes this a within-participant contrast
    rather than a repeat viewing.
    """

    sequence: int
    first: str
    second: str

    @property
    def prepared_segment(self) -> str:
        return self.first

    @property
    def regenerated_segment(self) -> str:
        return self.second

    def condition_of(self, segment: str) -> str:
        """Which track segment carries for this participant."""
        if segment == self.first:
            return PREPARED
        if segment == self.second:
            return REGENERATED
        raise AssignmentError(f"segment {segment!r} is not in this assignment")

    def segment_of(self, condition: str) -> str:
        if condition == PREPARED:
            return self.first
        if condition == REGENERATED:
            return self.second
        raise AssignmentError(f"condition {condition!r} is neither {PREPARED} nor {REGENERATED}")

    def record(self) -> dict[str, object]:
        """What the log and the page are told. No measured value is in here."""
        return {
            "sequence": self.sequence,
            "prepared_segment": self.first,
            "regenerated_segment": self.second,
        }


def assign(sequence: int) -> Assignment:
    """The assignment for one sequence number (§1).

    Even: prepared on A, regenerated on B. Odd: the reverse.
    """
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise AssignmentError("the participant sequence number is a non-negative integer")
    first, second = SEGMENTS if sequence % 2 == 0 else tuple(reversed(SEGMENTS))
    return Assignment(sequence=sequence, first=first, second=second)
