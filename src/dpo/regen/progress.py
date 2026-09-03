"""§9.1: the six steps in order, with no way back and no way to repeat one.

The order is the instrument. §8 asks the ART items a second time and the
comparison against §3 is the study's main measure, so a participant who could
return to §3 and revise after seeing the second viewing would be answering a
different question with the same items. A participant who could skip forward
would answer §8 without the regeneration their answers are about.

Enforced on the server, not in the page. A page can hide a back button; it
cannot stop a reload, a second tab, or a typed URL, and every one of those is
something a participant does by accident in a real session. The gate is a
function of what has been *recorded*, so it survives all three: the state lives
in the snapshot, and a step is entered only from the step before it.

``REGENERATING`` is in the sequence but is not one of §9.1's six steps. It is
where §6 runs, and it is a step here for one reason: a reload while the model
is writing must come back to the waiting screen rather than to §5's submit,
which would run the regeneration a second time from the same report.

Section numbers cite ``docs/v3-regen/spec-behavior.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

VIEW_PREPARED = "view_prepared"
ART = "art"
VISUAL = "visual"
AUDITORY = "auditory"
REGENERATING = "regenerating"
VIEW_REGENERATED = "view_regenerated"
SURVEY = "survey"
DONE = "done"

# In order. Index in this tuple is the only ordering there is.
STEPS = (VIEW_PREPARED, ART, VISUAL, AUDITORY, REGENERATING, VIEW_REGENERATED, SURVEY, DONE)
# The six §9.1 counts, for the page's progress indicator. REGENERATING is a
# wait, not a step a participant takes, and DONE is the end of the last one.
PARTICIPANT_STEPS = (VIEW_PREPARED, ART, VISUAL, AUDITORY, VIEW_REGENERATED, SURVEY)


class ProgressError(ValueError):
    """A step that cannot be entered from where the participant is."""


def position(step: str) -> int:
    try:
        return STEPS.index(step)
    except ValueError:
        raise ProgressError(f"no step named {step!r}") from None


def current(snapshot: Mapping[str, Any] | None) -> str:
    """Where the participant is. A session with no snapshot has not begun."""
    if snapshot is None:
        return VIEW_PREPARED
    step = snapshot.get("step")
    if not isinstance(step, str):
        return VIEW_PREPARED
    position(step)
    return step


def require(snapshot: Mapping[str, Any] | None, step: str) -> None:
    """Refuse a submission that is not for the step in progress.

    One rule covers both directions, which is why it is stated as equality
    rather than as two comparisons: a step behind the current one is a
    re-entry to something completed, and a step ahead is a skip. Neither is a
    thing the instrument allows, and neither should be reported as the other.
    """
    at = current(snapshot)
    if step == at:
        return
    if position(step) < position(at):
        raise ProgressError(f"{step} is already complete; the session is at {at} and does not go back (§9.1)")
    raise ProgressError(f"{step} cannot be entered from {at}; the steps run in order (§9.1)")


def advance(snapshot: Mapping[str, Any], step: str) -> dict[str, Any]:
    """The snapshot moved on from ``step`` to the one after it."""
    require(snapshot, step)
    return {**snapshot, "step": STEPS[position(step) + 1]}
