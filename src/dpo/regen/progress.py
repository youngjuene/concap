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
OVERALL = "overall"
DONE = "done"
FLOW_VERSION = "clip-caption-prss-v2"

# In order. Index in this tuple is the only ordering there is.
STEPS = (VIEW_PREPARED, ART, VISUAL, AUDITORY, REGENERATING, VIEW_REGENERATED, SURVEY, DONE)
CLIP_STEPS = (VIEW_PREPARED, ART, VISUAL, AUDITORY, REGENERATING, VIEW_REGENERATED, SURVEY)
VERSIONED_STEPS = (*CLIP_STEPS, OVERALL, DONE)
# The six §9.1 counts, for the page's progress indicator. REGENERATING is a
# wait, not a step a participant takes, and DONE is the end of the last one.
PARTICIPANT_STEPS = (VIEW_PREPARED, ART, VISUAL, AUDITORY, VIEW_REGENERATED, SURVEY)
VERSIONED_PARTICIPANT_STEPS = (VIEW_PREPARED, ART, VISUAL, AUDITORY, VIEW_REGENERATED, SURVEY, OVERALL)


class ProgressError(ValueError):
    """A step that cannot be entered from where the participant is."""


def position(step: str) -> int:
    try:
        return STEPS.index(step)
    except ValueError:
        raise ProgressError(f"no step named {step!r}") from None


def versioned_position(step: str) -> int:
    try:
        return VERSIONED_STEPS.index(step)
    except ValueError:
        raise ProgressError(f"no step named {step!r}") from None


def is_versioned(snapshot: Mapping[str, Any] | None) -> bool:
    return snapshot is not None and snapshot.get("flow_version") == FLOW_VERSION


def current(snapshot: Mapping[str, Any] | None) -> str:
    """Where the participant is. A session with no snapshot has not begun."""
    if snapshot is None:
        return VIEW_PREPARED
    step = snapshot.get("step")
    if not isinstance(step, str):
        return VIEW_PREPARED
    if is_versioned(snapshot):
        versioned_position(step)
    else:
        position(step)
    return step


def clip_index(snapshot: Mapping[str, Any] | None) -> int:
    if not is_versioned(snapshot):
        return 0
    assert snapshot is not None
    index = snapshot.get("clip_index")
    return index if isinstance(index, int) and index >= 0 else 0


def clip_count(snapshot: Mapping[str, Any] | None, default: int = 1) -> int:
    if not is_versioned(snapshot):
        return default
    assert snapshot is not None
    count = snapshot.get("clip_count")
    return count if isinstance(count, int) and count > 0 else default


def require_clip(snapshot: Mapping[str, Any] | None, submitted: Any) -> None:
    if not is_versioned(snapshot):
        return
    if not isinstance(submitted, int) or isinstance(submitted, bool):
        raise ProgressError("clip_index is required for this versioned session")
    if submitted != clip_index(snapshot):
        raise ProgressError(
            f"clip_index {submitted!r} does not match the current clip {clip_index(snapshot)}"
        )


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
    order = versioned_position if is_versioned(snapshot) else position
    if order(step) < order(at):
        raise ProgressError(f"{step} is already complete; the session is at {at} and does not go back (§9.1)")
    raise ProgressError(f"{step} cannot be entered from {at}; the steps run in order (§9.1)")


def advance(snapshot: Mapping[str, Any], step: str, *, clip_count: int | None = None) -> dict[str, Any]:
    """The snapshot moved on from ``step`` to the one after it."""
    require(snapshot, step)
    if is_versioned(snapshot):
        count = clip_count if clip_count is not None else globals()["clip_count"](snapshot)
        index = globals()["clip_index"](snapshot)
        if step == SURVEY:
            if index + 1 < count:
                return {**snapshot, "step": VIEW_PREPARED, "clip_index": index + 1, "clip_count": count}
            return {**snapshot, "step": OVERALL, "clip_index": index, "clip_count": count}
        if step == OVERALL:
            return {**snapshot, "step": DONE, "clip_index": index, "clip_count": count}
        return {
            **snapshot,
            "step": CLIP_STEPS[CLIP_STEPS.index(step) + 1],
            "clip_index": index,
            "clip_count": count,
        }
    return {**snapshot, "step": STEPS[position(step) + 1]}
