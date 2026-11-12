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

**The watch's shared state** (researcher, 2026-11-12: participants are remote, and on Vercel any
instance may take any request). Until then a long video's state lived in one server's memory, so a
request that reached a second instance restarted the video. Three small tables hold what every
instance must agree on while a video plays:

- ``watch_now``, one row per session and video: the sliders in force, the last playback report, and
  how much of the video was played (``watched_ms`` credits only new territory, and no more than
  the wall clock allows, as phase 1's coverage rule does);
- ``watch_results``: every caption attempt that ended, in order, so an instance that joins replays
  them and serves the same captions;
- ``watch_claims``: who is writing what, so two instances do not pay for the same caption.

``links`` maps a participant's personal link token to their ID.
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
# The same SQL on SQLite and Postgres (``deploy/vercel/pg_store.py`` adds its own ``watch_results``,
# whose serial key is spelt differently).
SHARED_TABLES: Final = (
    "CREATE TABLE IF NOT EXISTS watch_now ("
    " session_id TEXT NOT NULL, video_id TEXT NOT NULL, texture DOUBLE PRECISION NOT NULL,"
    " context DOUBLE PRECISION NOT NULL, spec_at DOUBLE PRECISION NOT NULL, position_ms INTEGER NOT NULL,"
    " playing INTEGER NOT NULL, play_at DOUBLE PRECISION NOT NULL, is_open INTEGER NOT NULL,"
    " watched_ms INTEGER NOT NULL, max_position_ms INTEGER NOT NULL, restarts INTEGER NOT NULL,"
    " PRIMARY KEY (session_id, video_id))",
    "CREATE INDEX IF NOT EXISTS watch_results_session ON watch_results (session_id, id)",
    "CREATE TABLE IF NOT EXISTS watch_claims ("
    " session_id TEXT NOT NULL, key TEXT NOT NULL, at DOUBLE PRECISION NOT NULL,"
    " PRIMARY KEY (session_id, key))",
    "CREATE TABLE IF NOT EXISTS links ("
    " token TEXT PRIMARY KEY, participant TEXT NOT NULL UNIQUE, created_at DOUBLE PRECISION NOT NULL)",
)
WATCH_COLUMNS: Final = (
    "video_id",
    "texture",
    "context",
    "spec_at",
    "position_ms",
    "playing",
    "play_at",
    "is_open",
    "watched_ms",
    "max_position_ms",
    "restarts",
)
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
    "CREATE TABLE IF NOT EXISTS watch_results ("
    " id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, video_id TEXT NOT NULL,"
    " cue INTEGER NOT NULL, job_id TEXT NOT NULL, origin TEXT NOT NULL, result TEXT NOT NULL,"
    " at REAL NOT NULL)",
    *SHARED_TABLES,
)

# Phase 1's coverage rule (``phase1_flow.record_playback``): how far a report may run ahead of the clock.
CREDIT_RATE: Final = 1.25
CREDIT_GRACE_MS: Final = 300

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
        self, session_id: str, change: Callable[[dict[str, Any]], None], event: Mapping[str, Any] | None
    ) -> None:
        """A server-side change (live progress) with no browser revision or key; ``event`` if worth one."""
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
            if event is not None:
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

    # -- the watch's shared state ------------------------------------------------
    def watch_open(self, session_id: str, video_id: str, texture: float, context: float) -> None:
        """The video is the one being watched, under these sliders; its playback record is kept."""
        now = self.clock()
        with self._connection() as db:
            db.execute(
                "INSERT INTO watch_now (session_id, video_id, texture, context, spec_at, position_ms,"
                " playing, play_at, is_open, watched_ms, max_position_ms, restarts)"
                " VALUES (?, ?, ?, ?, ?, 0, 0, ?, 1, 0, 0, 0)"
                " ON CONFLICT (session_id, video_id) DO UPDATE SET texture = excluded.texture,"
                " context = excluded.context, spec_at = excluded.spec_at, is_open = 1",
                (session_id, video_id, texture, context, now, now),
            )

    def watch_settings(self, session_id: str, video_id: str, texture: float, context: float) -> None:
        now = self.clock()
        with self._connection() as db:
            db.execute(
                "UPDATE watch_now SET texture = ?, context = ?, spec_at = ?"
                " WHERE session_id = ? AND video_id = ?",
                (texture, context, now, session_id, video_id),
            )

    def watch_playback(self, session_id: str, video_id: str, position_ms: int, playing: bool) -> None:
        """One playback report: where the video is, and what it adds to the time watched.

        Only new territory counts (a restarted video earns nothing for what was already played), only
        while the last report said the video was playing, and no more than the wall clock allows
        (phase 1's rule), so a position that jumps ahead earns nothing for the part skipped.
        """
        now = self.clock()
        with self._transaction() as db:
            row = db.execute(
                "SELECT position_ms, playing, play_at, watched_ms, max_position_ms FROM watch_now"
                " WHERE session_id = ? AND video_id = ?",
                (session_id, video_id),
            ).fetchone()
            if row is None:
                return
            last, was_playing, at, watched, furthest = row
            if was_playing and position_ms >= last:
                allowed = (now - at) * 1000 * CREDIT_RATE + CREDIT_GRACE_MS
                watched += int(max(0, min(position_ms - max(last, furthest), allowed)))
            db.execute(
                "UPDATE watch_now SET position_ms = ?, playing = ?, play_at = ?, watched_ms = ?,"
                " max_position_ms = ? WHERE session_id = ? AND video_id = ?",
                (position_ms, int(playing), now, watched, max(furthest, position_ms), session_id, video_id),
            )

    def watch_restart(self, session_id: str, video_id: str) -> None:
        """The browser lost the video (a reload): it starts again from 0, and that is counted."""
        with self._connection() as db:
            db.execute(
                "UPDATE watch_now SET position_ms = 0, playing = 0, play_at = ?, restarts = restarts + 1"
                " WHERE session_id = ? AND video_id = ?",
                (self.clock(), session_id, video_id),
            )

    def watch_close(self, session_id: str, video_id: str) -> None:
        with self._connection() as db:
            db.execute(
                "UPDATE watch_now SET is_open = 0, playing = 0 WHERE session_id = ? AND video_id = ?",
                (session_id, video_id),
            )

    def watch_now(self, session_id: str, video_id: str) -> dict[str, Any] | None:
        """The video's row, with ``now`` (this store's clock) to age its reports against."""
        with self._connection() as db:
            row = db.execute(
                f"SELECT {', '.join(WATCH_COLUMNS)} FROM watch_now WHERE session_id = ? AND video_id = ?",
                (session_id, video_id),
            ).fetchone()
        return None if row is None else {**dict(zip(WATCH_COLUMNS, row, strict=True)), "now": self.clock()}

    def watch_videos(self, session_id: str) -> list[dict[str, Any]]:
        with self._connection() as db:
            rows = db.execute(
                f"SELECT {', '.join(WATCH_COLUMNS)} FROM watch_now WHERE session_id = ? ORDER BY spec_at",
                (session_id,),
            ).fetchall()
        return [dict(zip(WATCH_COLUMNS, row, strict=True)) for row in rows]

    def watch_put(
        self, session_id: str, video_id: str, cue: int, job_id: str, origin: str, result: Mapping[str, Any]
    ) -> None:
        with self._connection() as db:
            db.execute(
                "INSERT INTO watch_results (session_id, video_id, cue, job_id, origin, result, at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    video_id,
                    cue,
                    job_id,
                    origin,
                    json.dumps(result, ensure_ascii=False),
                    self.clock(),
                ),
            )

    def watch_rows(self, session_id: str, after: int) -> list[dict[str, Any]]:
        """Every attempt recorded for the session after row ``after``, oldest first."""
        with self._connection() as db:
            rows = db.execute(
                "SELECT id, video_id, cue, job_id, origin, result FROM watch_results"
                " WHERE session_id = ? AND id > ? ORDER BY id",
                (session_id, after),
            ).fetchall()
        return [
            {"id": row[0], "video_id": row[1], "cue": row[2], "job_id": row[3], "origin": row[4]}
            | {"result": json.loads(row[5])}
            for row in rows
        ]

    def watch_claim(self, session_id: str, key: str, ttl_s: float) -> bool:
        """Take the right to write ``key``; ``False`` while another writer's claim is younger than
        ``ttl_s`` (a writer that never reports back loses it after that)."""
        now = self.clock()
        with self._connection() as db:
            taken = db.execute(
                "INSERT INTO watch_claims (session_id, key, at) VALUES (?, ?, ?)"
                " ON CONFLICT (session_id, key) DO UPDATE SET at = excluded.at WHERE watch_claims.at < ?",
                (session_id, key, now, now - ttl_s),
            )
            return bool(taken.rowcount == 1)

    def watch_release(self, session_id: str, key: str) -> None:
        with self._connection() as db:
            db.execute("DELETE FROM watch_claims WHERE session_id = ? AND key = ?", (session_id, key))

    # -- personal links ----------------------------------------------------------
    def link_issue(self, participant: str) -> str:
        """The participant's link token, made on first asking and the same ever after."""
        if not PARTICIPANT_ID.fullmatch(participant):
            raise ValueError("participant IDs look like P1, P2, …")
        with self._transaction() as db:
            row = db.execute("SELECT token FROM links WHERE participant = ?", (participant,)).fetchone()
            if row is not None:
                return str(row[0])
            token = secrets.token_urlsafe(12)
            db.execute("INSERT INTO links VALUES (?, ?, ?)", (token, participant, self.clock()))
            return token

    def link_participant(self, token: str) -> str | None:
        with self._connection() as db:
            row = db.execute("SELECT participant FROM links WHERE token = ?", (token,)).fetchone()
        return None if row is None else str(row[0])

    def links(self) -> list[tuple[str, str]]:
        with self._connection() as db:
            rows = db.execute(
                "SELECT participant, token FROM links ORDER BY created_at, participant"
            ).fetchall()
        return [(str(participant), str(token)) for participant, token in rows]
