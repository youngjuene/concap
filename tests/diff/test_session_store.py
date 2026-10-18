"""Durable sessions: entry by participant ID, resume, refusal when finished, exactly-once answers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from diff.session_store import Conflict, SessionFinished, SessionStore


def initial(session_id: str) -> dict[str, Any]:
    return {"page": "P0", "answers": []}


def answer(value: int):
    def apply(state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        state["answers"].append(value)
        return {"revision": state["revision"], "count": len(state["answers"])}, {
            "event": "answer",
            "value": value,
        }

    return apply


def test_an_id_creates_once_and_then_resumes_with_a_new_token(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.sqlite")
    first, created = store.enter("P1", initial)
    again, created_again = store.enter("P1", initial)
    assert created and not created_again
    assert again.session_id == first.session_id and again.token != first.token
    assert store.by_token(first.token) is None
    assert store.by_token(again.token).participant == "P1"


@pytest.mark.parametrize("participant", ["p1", "P0", "P01", "P12345", "1", "P1 ", "참가자1"])
def test_only_p_numbers_are_accepted(tmp_path: Path, participant: str) -> None:
    with pytest.raises(ValueError, match="P1, P2"):
        SessionStore(tmp_path / "s.sqlite").enter(participant, initial)


def test_a_session_survives_a_restart(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.sqlite")
    session, _ = store.enter("P7", initial)
    store.mutate(session.session_id, "k1", 0, {"a": 1}, answer(1))
    reopened = SessionStore(tmp_path / "s.sqlite")
    resumed, created = reopened.enter("P7", initial)
    assert not created and resumed.state["answers"] == [1] and resumed.revision == 1


def test_a_finished_id_is_refused(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.sqlite")
    session, _ = store.enter("P2", initial)
    store.finish(session.session_id)
    with pytest.raises(SessionFinished):
        store.enter("P2", initial)
    with pytest.raises(Conflict, match="finished"):
        store.mutate(session.session_id, "k1", 0, {}, answer(1))


def test_a_retried_request_returns_the_stored_reply(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.sqlite")
    session, _ = store.enter("P3", initial)
    reply = store.mutate(session.session_id, "k1", 0, {"a": 1}, answer(1))
    assert (
        store.mutate(session.session_id, "k1", 0, {"a": 1}, answer(1)) == reply == {"revision": 1, "count": 1}
    )
    assert store.get(session.session_id).state["answers"] == [1]
    with pytest.raises(Conflict, match="different request"):
        store.mutate(session.session_id, "k1", 1, {"a": 2}, answer(2))


def test_a_stale_revision_is_a_conflict(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.sqlite")
    session, _ = store.enter("P4", initial)
    store.mutate(session.session_id, "k1", 0, {"a": 1}, answer(1))
    for stale in (0, 2, True):
        with pytest.raises(Conflict, match="stale revision"):
            store.mutate(session.session_id, f"k-{stale}", stale, {"a": 2}, answer(2))


def test_a_failing_apply_stores_nothing(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.sqlite")
    session, _ = store.enter("P5", initial)

    def broken(state: dict[str, Any]) -> tuple[dict[str, Any], None]:
        state["answers"].append(9)
        raise ValueError("the next page cannot render")

    with pytest.raises(ValueError, match="cannot render"):
        store.mutate(session.session_id, "k1", 0, {"a": 9}, broken)
    after = store.get(session.session_id)
    assert after.state["answers"] == [] and after.revision == 0
    assert store.mutate(session.session_id, "k1", 0, {"a": 1}, answer(1))["count"] == 1


def test_keys_are_bounded(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.sqlite")
    session, _ = store.enter("P6", initial)
    for key in ("", "k" * 101, None):
        with pytest.raises(ValueError, match="idempotency key"):
            store.mutate(session.session_id, key, 0, {}, answer(1))  # type: ignore[arg-type]


def test_events_record_every_change_in_order(tmp_path: Path) -> None:
    clock = iter(range(100, 200))
    store = SessionStore(tmp_path / "s.sqlite", clock=lambda: float(next(clock)))
    session, _ = store.enter("P8", initial)
    store.mutate(session.session_id, "k1", 0, {"a": 1}, answer(1))
    store.update(session.session_id, lambda state: state.update(video_index=1), {"event": "video_started"})
    store.finish(session.session_id)
    events = store.events(session.session_id)
    assert [event["event"] for event in events] == [
        "session_created",
        "answer",
        "video_started",
        "session_finished",
    ]
    assert events[1]["value"] == 1 and events[0]["at"] < events[-1]["at"]
    assert store.get(session.session_id).state["video_index"] == 1
    assert [stored.participant for stored in store.sessions()] == ["P8"]


def test_a_store_of_another_schema_is_refused(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "s.sqlite"
    SessionStore(path)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE meta SET value = 'other/v9' WHERE name = 'schema'")
    with pytest.raises(ValueError, match="other/v9"):
        SessionStore(path)
