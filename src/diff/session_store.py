"""Durable participant sessions: one row per participant ID, every change atomic and idempotent.

Researcher's decisions of 2026-10-18: the experimenter types the participant
ID (``P1``, ``P2``, …). The same ID resumes an unfinished session, also after a
server restart, and a finished one is refused.

Each answer is applied the way regen's ``study_store`` applies it, without
importing it:
- under ``BEGIN IMMEDIATE``;
- against the revision the browser last saw;
- under an idempotency key.

A retried request with the same key and body returns the stored reply. A
different body under the same key, or a stale revision, is a :class:`Conflict`
(HTTP 409). If ``apply`` raises, for instance because the next page cannot
render, nothing is stored.

A new entry for an ID issues a new cookie token and voids the old one, so only
the last browser the experimenter opened drives the session.
"""

from __future__ import annotations

import copy
import json
import re
import secrets
import sqlite3
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from diff.study_protocol import canonical_json, sha256_of

STORE_SCHEMA: Final = "diff.session-store/v1"
PARTICIPANT_ID: Final = re.compile(r"^P[1-9][0-9]{0,3}$")
KEY_MAX: Final = 100
TABLES: Final = (
    "CREATE TABLE IF NOT EXISTS meta (name TEXT PRIMARY KEY, value TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS sessions ("
    " session_id TEXT PRIMARY KEY, participant TEXT NOT NULL UNIQUE, token TEXT NOT NULL UNIQUE,"
    " state TEXT NOT NULL, revision INTEGER NOT NULL, created_at REAL NOT NULL, finished_at REAL)",
    "CREATE TABLE IF NOT EXISTS receipts ("
    " session_id TEXT NOT NULL, key TEXT NOT NULL, request_sha256 TEXT NOT NULL, reply TEXT NOT NULL,"
    " at REAL NOT NULL, PRIMARY KEY (session_id, key))",
    "CREATE TABLE IF NOT EXISTS events ("
    " id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, at REAL NOT NULL, body TEXT NOT NULL)",
)

Apply = Callable[[dict[str, Any]], tuple[dict[str, Any], Mapping[str, Any] | None]]


class Conflict(Exception):
    """A stale revision or a reused key: the browser must reload the state (HTTP 409)."""


class SessionFinished(Exception):
    """The participant ID belongs to a finished session; it cannot be entered again."""


@dataclass(frozen=True)
class Session:
    session_id: str
    participant: str
    token: str
    state: dict[str, Any]
    revision: int
    created_at: float
    finished_at: float | None


class SessionStore:
    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time) -> None:
        self.path = path
        self.clock = clock
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            for table in TABLES:
                db.execute(table)
            db.execute("INSERT OR IGNORE INTO meta VALUES ('schema', ?)", (STORE_SCHEMA,))
            (schema,) = db.execute("SELECT value FROM meta WHERE name = 'schema'").fetchone()
        if schema != STORE_SCHEMA:
            raise ValueError(f"{path} is a {schema} store, not {STORE_SCHEMA}")

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, isolation_level=None, timeout=10)
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
            except BaseException:
                db.execute("ROLLBACK")
                raise
            db.execute("COMMIT")

    @staticmethod
    def _session(row: tuple[Any, ...]) -> Session:
        session_id, participant, token, state, revision, created_at, finished_at = row
        return Session(session_id, participant, token, json.loads(state), revision, created_at, finished_at)

    @staticmethod
    def _event(db: sqlite3.Connection, session_id: str, at: float, body: Mapping[str, Any]) -> None:
        db.execute(
            "INSERT INTO events (session_id, at, body) VALUES (?, ?, ?)",
            (session_id, at, canonical_json(body)),
        )

    def enter(self, participant: str, initial: Callable[[str], dict[str, Any]]) -> tuple[Session, bool]:
        """The participant's session with a fresh token, created with ``initial(session_id)`` if new."""
        if not PARTICIPANT_ID.fullmatch(participant):
            raise ValueError("participant IDs look like P1, P2, …")
        now, token = self.clock(), secrets.token_urlsafe(24)
        with self._transaction() as db:
            row = db.execute("SELECT * FROM sessions WHERE participant = ?", (participant,)).fetchone()
            if row is not None:
                existing = self._session(row)
                if existing.finished_at is not None:
                    raise SessionFinished(participant)
                db.execute("UPDATE sessions SET token = ? WHERE session_id = ?", (token, existing.session_id))
                self._event(db, existing.session_id, now, {"event": "session_resumed"})
                return Session(**{**existing.__dict__, "token": token}), False
            session_id = "s-" + secrets.token_hex(8)
            state = initial(session_id)
            db.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, 0, ?, NULL)",
                (session_id, participant, token, canonical_json(state), now),
            )
            self._event(db, session_id, now, {"event": "session_created"})
            return Session(session_id, participant, token, state, 0, now, None), True

    def by_token(self, token: str | None) -> Session | None:
        if not token:
            return None
        with self._connection() as db:
            row = db.execute("SELECT * FROM sessions WHERE token = ?", (token,)).fetchone()
        return None if row is None else self._session(row)

    def get(self, session_id: str) -> Session:
        with self._connection() as db:
            row = db.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if row is None:
            raise KeyError(session_id)
        return self._session(row)

    def mutate(
        self, session_id: str, key: str, revision: int, request: Mapping[str, Any], apply: Apply
    ) -> dict[str, Any]:
        """Apply one browser request exactly once; the reply is stored for retries."""
        if not (isinstance(key, str) and 1 <= len(key) <= KEY_MAX):
            raise ValueError(f"the idempotency key must be 1–{KEY_MAX} characters")
        request_sha256 = sha256_of(request)
        now = self.clock()
        with self._transaction() as db:
            receipt = db.execute(
                "SELECT request_sha256, reply FROM receipts WHERE session_id = ? AND key = ?",
                (session_id, key),
            ).fetchone()
            if receipt is not None:
                if receipt[0] != request_sha256:
                    raise Conflict("this request key was already used for a different request")
                return dict(json.loads(receipt[1]))
            row = db.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            if row is None:
                raise KeyError(session_id)
            current = self._session(row)
            if current.finished_at is not None:
                raise Conflict("the session is finished")
            if type(revision) is not int or revision != current.revision:
                raise Conflict(f"stale revision {revision!r}; the session is at {current.revision}")
            state = copy.deepcopy(current.state)
            state["revision"] = current.revision + 1
            reply, event = apply(state)
            db.execute(
                "UPDATE sessions SET state = ?, revision = ? WHERE session_id = ?",
                (canonical_json(state), current.revision + 1, session_id),
            )
            db.execute(
                "INSERT INTO receipts VALUES (?, ?, ?, ?, ?)",
                (session_id, key, request_sha256, canonical_json(reply), now),
            )
            if event is not None:
                self._event(db, session_id, now, event)
        return reply

    def update(
        self, session_id: str, change: Callable[[dict[str, Any]], None], event: Mapping[str, Any]
    ) -> None:
        """A server-side change (live progress) with no browser revision or key."""
        now = self.clock()
        with self._transaction() as db:
            row = db.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            if row is None:
                raise KeyError(session_id)
            current = self._session(row)
            state = copy.deepcopy(current.state)
            change(state)
            db.execute(
                "UPDATE sessions SET state = ? WHERE session_id = ?", (canonical_json(state), session_id)
            )
            self._event(db, session_id, now, event)

    def finish(self, session_id: str) -> None:
        now = self.clock()
        with self._transaction() as db:
            db.execute("UPDATE sessions SET finished_at = ? WHERE session_id = ?", (now, session_id))
            self._event(db, session_id, now, {"event": "session_finished"})

    def sessions(self) -> list[Session]:
        with self._connection() as db:
            rows = db.execute("SELECT * FROM sessions ORDER BY created_at").fetchall()
        return [self._session(row) for row in rows]

    def events(self, session_id: str) -> list[dict[str, Any]]:
        with self._connection() as db:
            rows = db.execute(
                "SELECT at, body FROM events WHERE session_id = ? ORDER BY id", (session_id,)
            ).fetchall()
        return [{"at": at, **json.loads(body)} for at, body in rows]
