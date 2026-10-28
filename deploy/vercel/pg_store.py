"""The lab's session store on Postgres, so every Vercel instance sees one set of sessions.

:class:`PgSessionStore` is :class:`diff.session_store.SessionStore` with only the connection layer
replaced: ``enter``, ``mutate``, ``update`` and ``finish`` (revisions, idempotency receipts,
events) are the lab's own code. SQLite's ``BEGIN IMMEDIATE`` takes the database's one write lock;
here every transaction takes one advisory lock instead, so writes are serialised the same way.

The SQL the base class sends is written for sqlite3 (``?`` placeholders); :class:`_Db` passes it
on with psycopg's ``%s``. None of that SQL holds a literal ``?``.

Smoke and participant sessions live in two databases of the one Neon project (:data:`DATABASES`),
as the lab keeps them in two stores: a test P1 never blocks a participant P1, and a pull of the
participants holds no test session. Only the URL's database name differs (:func:`on_database`).
"""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Final
from urllib.parse import urlsplit, urlunsplit

from diff.session_store import STORE_SCHEMA, SessionStore

# neondb is the one the Neon integration made; participant was created on 2026-10-28.
DATABASES: Final = {"smoke": "neondb", "participant": "participant"}
WRITE_LOCK: Final = 0x646966660001  # any constant; every writer takes the same one
TABLES: Final = (
    "CREATE TABLE IF NOT EXISTS meta (name TEXT PRIMARY KEY, value TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS sessions ("
    " session_id TEXT PRIMARY KEY, participant TEXT NOT NULL UNIQUE, token TEXT NOT NULL UNIQUE,"
    " state TEXT NOT NULL, revision INTEGER NOT NULL, created_at DOUBLE PRECISION NOT NULL,"
    " finished_at DOUBLE PRECISION)",
    "CREATE TABLE IF NOT EXISTS receipts ("
    " session_id TEXT NOT NULL, key TEXT NOT NULL, request_sha256 TEXT NOT NULL, reply TEXT NOT NULL,"
    " at DOUBLE PRECISION NOT NULL, PRIMARY KEY (session_id, key))",
    "CREATE TABLE IF NOT EXISTS events ("
    " id BIGSERIAL PRIMARY KEY, session_id TEXT NOT NULL, at DOUBLE PRECISION NOT NULL, body TEXT NOT NULL)",
)


LOGS: Final = (
    "CREATE TABLE IF NOT EXISTS logs ("
    " id BIGSERIAL PRIMARY KEY, name TEXT NOT NULL, line TEXT NOT NULL, at DOUBLE PRECISION NOT NULL)"
)


def on_database(url: str, name: str) -> str:
    """``url`` (Neon's ``DATABASE_URL``, pooled or not) pointed at the database ``name``."""
    return urlunsplit(urlsplit(url)._replace(path=f"/{name}"))


class _Db:
    """What the base class calls on a sqlite3 connection, on a psycopg one."""

    def __init__(self, connection: Any) -> None:
        self.connection = connection

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> Any:
        return self.connection.execute(sql.replace("?", "%s"), params)


class PgSessionStore(SessionStore):
    def __init__(self, url: str, *, clock: Callable[[], float] = time.time, max_size: int = 4) -> None:
        from psycopg_pool import ConnectionPool

        self.path = None  # type: ignore[assignment]
        self.clock = clock
        # Neon's pooled endpoint runs PgBouncer in transaction mode: no server-side prepares.
        self.pool = ConnectionPool(
            url,
            min_size=1,
            max_size=max_size,
            kwargs={"autocommit": True, "prepare_threshold": None},
            open=True,
        )
        with self._transaction() as db:
            for table in (*TABLES, LOGS):
                db.execute(table)
            db.execute("INSERT INTO meta VALUES ('schema', ?) ON CONFLICT DO NOTHING", (STORE_SCHEMA,))
            (schema,) = db.execute("SELECT value FROM meta WHERE name = 'schema'").fetchone()
        if schema != STORE_SCHEMA:
            raise ValueError(f"the database holds a {schema} store, not {STORE_SCHEMA}")

    @contextmanager
    def _connection(self) -> Iterator[_Db]:  # type: ignore[override]
        with self.pool.connection() as connection:
            yield _Db(connection)

    @contextmanager
    def _transaction(self) -> Iterator[_Db]:  # type: ignore[override]
        with self.pool.connection() as connection, connection.transaction():
            connection.execute("SELECT pg_advisory_xact_lock(%s)", (WRITE_LOCK,))
            yield _Db(connection)

    def log_dir(self) -> "PgLogDir":
        """Where the live logs go: rows of ``logs``, which ``pull.py`` writes back out as files."""
        return PgLogDir(self)


class PgLogDir:
    """A ``log_dir`` whose files are rows: the server only ever does ``(log_dir / name).open("a")``.

    ``live_event`` and ``LiveService`` append one JSON line per ``open``. On Vercel each instance's
    /tmp is its own and dies with it, so each line is inserted, in the writing thread, before the
    write returns.
    """

    def __init__(self, store: PgSessionStore, name: str = "") -> None:
        self.store, self.name = store, name

    def mkdir(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __truediv__(self, name: str) -> "PgLogDir":
        return PgLogDir(self.store, f"{self.name}/{name}" if self.name else name)

    @contextmanager
    def open(self, mode: str, encoding: str = "utf-8") -> Iterator["PgLogDir"]:
        if mode != "a":
            raise ValueError(f"the log store only appends, not {mode!r}")
        yield self

    def write(self, text: str) -> int:
        with self.store._connection() as db:
            for line in text.splitlines():
                db.execute(
                    "INSERT INTO logs (name, line, at) VALUES (?, ?, ?)",
                    (self.name, line, self.store.clock()),
                )
        return len(text)
