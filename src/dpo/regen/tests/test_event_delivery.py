"""Regression tests for idempotent client telemetry delivery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dpo.regen.log import EventLog, RegenLogError


def event(event_id: str, **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "event_id": event_id,
        "type": "control.changed",
        "phase": 2,
        "stage": "long-viewing",
        "clip_id": "long-0",
        "clip_index": 0,
        "view_id": "p1-v0",
        "condition": "personalized",
        "flow_version": "flow-test",
        "context_id": "ctx-0",
        "at": "2026-09-22T00:00:00.000Z",
        "position_ms": 1000,
    }
    base.update(overrides)
    return base


def events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_export_omits_event_signing_secret_without_mutating_snapshot(tmp_path: Path) -> None:
    log = EventLog(tmp_path, "config-a")
    log.write_snapshot("p1", {"step": "art", "event_secret": "private-signing-material"})
    assert log.export("p1", "study")["snapshot"] == {"step": "art"}
    assert (log.snapshot("p1") or {})["event_secret"] == "private-signing-material"


def test_append_events_acknowledges_retries_without_duplicate_write(tmp_path: Path) -> None:
    log = EventLog(tmp_path, "config-a")
    row = event("event-1")

    assert log.append_events("p1", [row]) == ["event-1"]
    assert log.append_events("p1", [row, dict(row)]) == ["event-1"]

    stored = events(log.events_path("p1"))
    assert [record["event_id"] for record in stored] == ["event-1"]
    assert stored[0]["config_hash"] == "config-a"
    assert "received_at" in stored[0]


def test_legacy_append_still_accepts_events_without_client_event_id(tmp_path: Path) -> None:
    log = EventLog(tmp_path, "config-a")

    assert log.append("p1", [{"type": "session.enrolled", "assignment": "A"}], {"step": "intro"}) == 1

    stored = events(log.events_path("p1"))
    assert stored[0]["type"] == "session.enrolled"
    assert "event_id" not in stored[0]
    assert log.snapshot("p1") == {"step": "intro"}


def test_append_events_rejects_changed_content_for_seen_event_id(tmp_path: Path) -> None:
    log = EventLog(tmp_path, "config-a")
    log.append_events("p1", [event("event-1", position_ms=1000)])

    with pytest.raises(RegenLogError, match="event_id .* different content"):
        log.append_events("p1", [event("event-1", position_ms=2000)])

    assert [record["position_ms"] for record in events(log.events_path("p1"))] == [1000]


def test_append_events_rejects_unbounded_or_nonfinite_payloads(tmp_path: Path) -> None:
    log = EventLog(tmp_path, "config-a")

    with pytest.raises(RegenLogError, match="event_id must be at most"):
        log.append_events("p1", [event("x" * 101)])
    with pytest.raises(RegenLogError, match="finite number"):
        log.append_events("p1", [event("event-bool-position", position_ms=True)])
    with pytest.raises(RegenLogError, match="finite number"):
        log.append_events("p1", [event("event-nan-position", position_ms=float("nan"))])
    with pytest.raises(RegenLogError, match="JSON serializable"):
        log.append_events("p1", [event("event-nan-nested", extra=float("nan"))])
    with pytest.raises(RegenLogError, match="exceeds 16384 bytes"):
        log.append_events("p1", [event("event-too-large", extra="x" * (16 * 1024))])

    assert not log.events_path("p1").exists()


def test_append_events_deduplicates_after_restart_and_ignores_torn_tail(tmp_path: Path) -> None:
    first = EventLog(tmp_path, "config-a")
    first.append_events("p1", [event("event-1")])
    path = first.events_path("p1")
    with path.open("ab") as handle:
        handle.write(b'{"event_id":"torn"')

    restarted = EventLog(tmp_path, "config-a")
    assert restarted.append_events("p1", [event("event-1"), event("event-2")]) == [
        "event-1",
        "event-2",
    ]

    stored = events(path)
    assert [record["event_id"] for record in stored] == ["event-1", "event-2"]
    assert all(record["event_id"] != "torn" for record in stored)


def test_append_events_reloads_index_after_partial_append_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = EventLog(tmp_path, "config-a")
    original_append = log._append
    failed = False

    def partial_append(path: Path, rows: list[dict[str, Any]]) -> int:
        nonlocal failed
        if not failed:
            failed = True
            original_append(path, rows[:1])
            raise OSError("simulated interrupted append")
        return original_append(path, rows)

    monkeypatch.setattr(log, "_append", partial_append)
    with pytest.raises(OSError, match="simulated interrupted append"):
        log.append_events("p1", [event("event-1"), event("event-2")])

    assert [record["event_id"] for record in events(log.events_path("p1"))] == ["event-1"]
    assert log.append_events("p1", [event("event-1"), event("event-2")]) == ["event-1", "event-2"]
    assert [record["event_id"] for record in events(log.events_path("p1"))] == ["event-1", "event-2"]


def test_append_events_does_not_concatenate_after_valid_line_without_newline(tmp_path: Path) -> None:
    log = EventLog(tmp_path, "config-a")
    path = log.events_path("p1")
    first = {
        **event("event-1"),
        "received_at": "2026-09-22T00:00:00.000Z",
        "config_hash": "config-a",
    }
    path.write_text(json.dumps(first, sort_keys=True), encoding="utf-8")

    assert log.append_events("p1", [event("event-2")]) == ["event-2"]

    assert [record["event_id"] for record in events(path)] == ["event-1", "event-2"]


def test_append_events_validates_batch_atomically(tmp_path: Path) -> None:
    log = EventLog(tmp_path, "config-a")
    log.append_events("p1", [event("event-1")])
    before = log.events_path("p1").read_text(encoding="utf-8")

    with pytest.raises(RegenLogError, match=r"events\[1\]"):
        log.append_events("p1", [event("event-2"), {"event_id": "event-3"}])
    assert log.events_path("p1").read_text(encoding="utf-8") == before

    with pytest.raises(RegenLogError, match="event_id .* different content"):
        log.append_events("p1", [event("event-4"), event("event-1", stage="changed")])
    assert log.events_path("p1").read_text(encoding="utf-8") == before
