"""One transactional authority for v2 sessions, receipts and caption jobs."""

from __future__ import annotations

import json
import secrets
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class Conflict(ValueError):
    """A mutation targets an old revision or the wrong stage."""


class StudyStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connection() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, state TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS calibration_links(
                    source TEXT PRIMARY KEY, token TEXT UNIQUE NOT NULL);
                CREATE TABLE IF NOT EXISTS receipts(
                    session TEXT, key TEXT, request TEXT, response TEXT, PRIMARY KEY(session,key));
                CREATE TABLE IF NOT EXISTS events(
                    id INTEGER PRIMARY KEY, session TEXT, kind TEXT, body TEXT, at REAL);
                CREATE TABLE IF NOT EXISTS jobs(
                    id TEXT PRIMARY KEY, session TEXT, content_key TEXT, epoch INTEGER,
                    revision INTEGER, cue INTEGER, state TEXT, spec TEXT, result TEXT,
                    created REAL, deadline REAL);
                CREATE TABLE IF NOT EXISTS captions(key TEXT PRIMARY KEY, result TEXT);
                CREATE TABLE IF NOT EXISTS exposures(
                    session TEXT, id TEXT, body TEXT, PRIMARY KEY(session,id));
                CREATE INDEX IF NOT EXISTS jobs_delivery ON jobs(session,epoch,revision);
                CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(state,created);
            """)

    def recover(self) -> None:
        with self.connection() as db:
            db.execute("UPDATE jobs SET state='expired' WHERE state='running'")

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _initial_state(calibration_hash: str, language: str) -> dict[str, Any]:
        return {
            "revision": 0,
            "session_id": secrets.token_hex(12),
            "stage": "preferences",
            "clip_index": 0,
            "video_index": 0,
            "calibration_hash": calibration_hash,
            "language": language,
            "observations": [],
            "axes": {"texture": 0.5, "context": 0.5},
            "settings_revision": 0,
            "epoch": 0,
            "position_ms": 0,
            "sequence": -1,
            "coverage": [],
            "completions": [],
            "draft": {},
        }

    def create(self, calibration_hash: str, language: str) -> str:
        token = secrets.token_urlsafe(32)
        state = self._initial_state(calibration_hash, language)
        with self.connection() as db:
            db.execute("INSERT INTO sessions VALUES(?,?)", (token, json.dumps(state)))
        return token

    def link(self, source: str, calibration_hash: str, language: str, *, create: bool) -> str | None:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT token FROM calibration_links WHERE source=?", (source,)).fetchone()
            if row is not None:
                return str(row[0])
            if not create:
                return None
            token = secrets.token_urlsafe(32)
            state = self._initial_state(calibration_hash, language)
            state["stage"] = "awaiting-calibration"
            db.execute("INSERT INTO sessions VALUES(?,?)", (token, json.dumps(state)))
            db.execute("INSERT INTO calibration_links VALUES(?,?)", (source, token))
            return token

    def state(self, token: str) -> dict[str, Any]:
        with self.connection() as db:
            row = db.execute("SELECT state FROM sessions WHERE id=?", (token,)).fetchone()
        if row is None:
            raise PermissionError("Session not found")
        return dict(json.loads(row[0]))

    def mutate(
        self, token: str, key: str, revision: int, request: str, operation: Callable[[dict[str, Any]], None]
    ) -> dict[str, Any]:
        if not isinstance(key, str) or not 1 <= len(key) <= 100:
            raise ValueError("A bounded idempotency key is required")
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            receipt = db.execute(
                "SELECT request,response FROM receipts WHERE session=? AND key=?", (token, key)
            ).fetchone()
            if receipt:
                if receipt["request"] != request:
                    raise Conflict("This request key was already used for different input")
                # A replay acknowledges the original mutation but returns the latest state,
                # never an old snapshot that could rewind a resumed client.
                latest = db.execute("SELECT state FROM sessions WHERE id=?", (token,)).fetchone()
                return dict(json.loads(latest[0]))
            row = db.execute("SELECT state FROM sessions WHERE id=?", (token,)).fetchone()
            if row is None:
                raise PermissionError("Session not found")
            state = json.loads(row[0])
            if type(revision) is not int or state["revision"] != revision:
                raise Conflict("Session changed; reload its current state")
            operation(state)
            for entry in state.pop("_exposure_batch", []):
                body = json.dumps(entry, sort_keys=True, ensure_ascii=False, allow_nan=False)
                existing = db.execute(
                    "SELECT body FROM exposures WHERE session=? AND id=?", (token, entry["id"])
                ).fetchone()
                if existing and existing[0] != body:
                    raise Conflict("Exposure ID was reused for different content")
                db.execute("INSERT OR IGNORE INTO exposures VALUES(?,?,?)", (token, entry["id"], body))
            state["revision"] += 1
            encoded = json.dumps(state, ensure_ascii=False, allow_nan=False)
            db.execute("UPDATE sessions SET state=? WHERE id=?", (encoded, token))
            db.execute(
                "INSERT INTO receipts VALUES(?,?,?,?)",
                (token, key, request, json.dumps({"revision": state["revision"]})),
            )
            db.execute(
                "INSERT INTO events(session,kind,body,at) VALUES(?,?,?,?)",
                (token, "mutation", request, time.time()),
            )
            if state["stage"] != "watch":
                db.execute("UPDATE jobs SET state='superseded' WHERE session=? AND state='queued'", (token,))
            return dict(state)

    def enqueue(
        self, token: str, state: dict[str, Any], cue: int, content_key: str, spec: dict[str, Any], limit: int
    ) -> None:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            latest = json.loads(db.execute("SELECT state FROM sessions WHERE id=?", (token,)).fetchone()[0])
            if (latest["epoch"], latest["settings_revision"], latest["stage"]) != (
                state["epoch"],
                state["settings_revision"],
                "watch",
            ):
                return
            db.execute(
                "UPDATE jobs SET state='superseded' WHERE session=? AND state='queued' "
                "AND (epoch!=? OR revision!=? OR cue<?)",
                (token, state["epoch"], state["settings_revision"], spec["current_index"]),
            )
            existing = db.execute(
                "SELECT 1 FROM jobs WHERE session=? AND content_key=? AND epoch=? AND revision=?",
                (token, content_key, state["epoch"], state["settings_revision"]),
            ).fetchone()
            if existing:
                return
            cached = db.execute("SELECT result FROM captions WHERE key=?", (content_key,)).fetchone()
            if (
                not cached
                and db.execute("SELECT count(*) FROM jobs WHERE state='queued'").fetchone()[0] >= limit
            ):
                return
            db.execute(
                "INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    secrets.token_hex(16),
                    token,
                    content_key,
                    state["epoch"],
                    state["settings_revision"],
                    cue,
                    "succeeded" if cached else "queued",
                    json.dumps(spec),
                    cached[0] if cached else None,
                    time.time(),
                    time.time() + spec["queue_timeout"],
                ),
            )

    def claim(self, previous_session: str | None = None) -> dict[str, Any] | None:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("UPDATE jobs SET state='expired' WHERE state='queued' AND deadline<?", (time.time(),))
            # Round-robin preference between participants; within a participant FIFO is current then next.
            row = db.execute(
                "SELECT * FROM jobs WHERE state='queued' ORDER BY (session=?) ASC, created ASC LIMIT 1",
                (previous_session or "",),
            ).fetchone()
            if row is None:
                return None
            state = json.loads(
                db.execute("SELECT state FROM sessions WHERE id=?", (row["session"],)).fetchone()[0]
            )
            if state["stage"] != "watch" or (state["epoch"], state["settings_revision"]) != (
                row["epoch"],
                row["revision"],
            ):
                db.execute("UPDATE jobs SET state='superseded' WHERE id=?", (row["id"],))
                return None
            # Another participant may have finished identical content since enqueue.
            cached = db.execute("SELECT result FROM captions WHERE key=?", (row["content_key"],)).fetchone()
            if cached:
                db.execute("UPDATE jobs SET state='succeeded',result=? WHERE id=?", (cached[0], row["id"]))
                return None
            db.execute("UPDATE jobs SET state='running' WHERE id=?", (row["id"],))
            return dict(row)

    def finish(self, job: dict[str, Any], result: dict[str, Any]) -> None:
        encoded = json.dumps(result, ensure_ascii=False)
        with self.connection() as db:
            db.execute(
                "UPDATE jobs SET state=?,result=? WHERE id=?",
                ("succeeded" if not result.get("fallback") else "failed", encoded, job["id"]),
            )
            if not result.get("fallback"):
                db.execute("INSERT OR REPLACE INTO captions VALUES(?,?)", (job["content_key"], encoded))

    def captions(self, token: str, state: dict[str, Any]) -> list[dict[str, Any]]:
        with self.connection() as db:
            rows = db.execute(
                "SELECT id,cue,result,state,revision FROM jobs WHERE session=? AND epoch=? "
                "AND revision=? ORDER BY cue",
                (token, state["epoch"], state["settings_revision"]),
            ).fetchall()
        return [{**dict(row), "result": json.loads(row["result"]) if row["result"] else None} for row in rows]

    def export(self, token: str) -> dict[str, Any]:
        state = self.state(token)
        with self.connection() as db:
            events = [
                dict(row) for row in db.execute("SELECT kind,body,at FROM events WHERE session=?", (token,))
            ]
            jobs = [dict(row) for row in db.execute("SELECT * FROM jobs WHERE session=?", (token,))]
            exposures = [
                json.loads(row[0])
                for row in db.execute("SELECT body FROM exposures WHERE session=?", (token,))
            ]
        for job in jobs:
            job.pop("session", None)  # The session credential is never exported.
        return {"state": state, "events": events, "jobs": jobs, "exposures": exposures}
