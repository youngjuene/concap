"""Shared playback coverage accounting for calibration and long viewing."""

from __future__ import annotations

import time
from typing import Any

from dpo.regen.study_schema import number
from dpo.regen.study_store import Conflict


def merged(intervals: list[list[float]]) -> list[list[float]]:
    result: list[list[float]] = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def record_playback(state: dict[str, Any], data: dict[str, Any], duration: float) -> None:
    sequence = data.get("sequence")
    if type(sequence) is not int or sequence <= state["sequence"]:
        raise Conflict("Playback update is out of order")
    position = number(data.get("position_ms"), 0, duration)
    now, previous = time.time(), state.get("last_tick")
    covered = sum(end - begin for begin, end in state["coverage"])
    # HTTP requests can arrive in a burst after one delayed response. Bound
    # credited playback by the whole uninterrupted playing interval, not the
    # gap between two arrivals. Pausing/seeking starts a fresh wall-clock budget.
    clock = (
        previous.get("clock", {"at": previous["at"], "covered": covered})
        if previous and previous["playing"]
        else {
            # Resume can start before its acknowledgement arrives. Allow a
            # bounded two-second transport head start, never the whole pause.
            "at": max(previous["at"], now - 2) if previous else now,
            "covered": covered,
        }
    )
    if data.get("seek") is True:
        state["epoch"] += 1
        clock = {"at": max(previous["at"], now - 2) if previous else now, "covered": covered}
    elif previous and previous["playing"] and not data.get("hidden", False):
        delta = position - previous["position"]
        proposed = merged(state["coverage"] + [[previous["position"], position]])
        credited = sum(end - begin for begin, end in proposed) - clock["covered"]
        elapsed = (now - clock["at"]) * 1000
        if not (0 <= delta <= 7000 and credited <= elapsed * 1.25 + 300):
            raise Conflict("Resume playback from the last saved position after the interruption.")
        state["coverage"] = proposed
    state["position_ms"], state["sequence"] = position, sequence
    state["last_tick"] = {
        "position": position,
        "at": now,
        "playing": data.get("playing") is True and not data.get("hidden", False),
        "clock": clock,
    }
