"""Transport jitter must not erase real viewing or grant skipped coverage."""

from typing import Any

import pytest

from dpo.regen.study_api import record_playback
from dpo.regen.study_store import Conflict


def test_bunched_arrivals_keep_continuous_real_time_coverage(monkeypatch: Any) -> None:
    state: dict[str, Any] = {"sequence": -1, "coverage": [], "epoch": 0, "last_tick": None}
    for sequence, (arrival, position) in enumerate(((10, 0), (11.9, 1000), (12, 2000))):
        monkeypatch.setattr("dpo.regen.study_api.time.time", lambda arrival=arrival: arrival)
        record_playback(state, {"sequence": sequence, "position_ms": position, "playing": True}, 300000)
    assert state["coverage"] == [[0, 2000]]


def test_fast_forward_and_paused_time_do_not_create_credit(monkeypatch: Any) -> None:
    state: dict[str, Any] = {"sequence": -1, "coverage": [], "epoch": 0, "last_tick": None}
    for sequence, (arrival, position, playing, seek) in enumerate(
        (
            (10, 0, True, False),
            (10.1, 6000, True, False),
            (11, 0, False, False),
            (100, 0, True, False),
            (100.1, 6000, True, False),
            (101, 250000, True, True),
        )
    ):
        monkeypatch.setattr("dpo.regen.study_api.time.time", lambda arrival=arrival: arrival)
        payload = {"sequence": sequence, "position_ms": position, "playing": playing, "seek": seek}
        if sequence in (1, 4):
            with pytest.raises(Conflict):
                record_playback(state, payload, 300000)
            assert state["position_ms"] == 0
        else:
            record_playback(state, payload, 300000)
    assert sum(end - begin for begin, end in state["coverage"]) == 0
    assert state["epoch"] == 1


def test_long_unacknowledged_interval_keeps_recovery_checkpoint(monkeypatch: Any) -> None:
    state: dict[str, Any] = {"sequence": -1, "coverage": [], "epoch": 0, "last_tick": None}
    monkeypatch.setattr("dpo.regen.study_api.time.time", lambda: 10)
    record_playback(state, {"sequence": 0, "position_ms": 0, "playing": True}, 300000)
    monkeypatch.setattr("dpo.regen.study_api.time.time", lambda: 25)
    with pytest.raises(Conflict):
        record_playback(state, {"sequence": 1, "position_ms": 15000, "playing": True}, 300000)
    assert state["position_ms"] == 0 and state["coverage"] == []


def test_resume_acknowledgement_latency_does_not_erase_played_time(monkeypatch: Any) -> None:
    state: dict[str, Any] = {"sequence": -1, "coverage": [], "epoch": 0, "last_tick": None}
    for sequence, (arrival, position, playing) in enumerate(
        ((10, 0, False), (12, 0, True), (12.2, 1700, True), (13.2, 2700, True))
    ):
        monkeypatch.setattr("dpo.regen.study_api.time.time", lambda arrival=arrival: arrival)
        record_playback(state, {"sequence": sequence, "position_ms": position, "playing": playing}, 300000)
    assert state["coverage"] == [[0, 2700]]
