from __future__ import annotations

import json
import re
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


class AccessDenied(PermissionError):
    """Raised before bytes from a protected role are read."""


HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
# The one protected role a capability can open. The test role is sealed by
# construction rather than by policy: it is a capability-read role
# (``dpo.core.artifacts.CAPABILITY_READ_ROLES``) with no scope here, so no
# reservation can ever authorize it. The one-shot confirmatory apparatus that
# would open it exactly once lives in git history, not in this module.
ROLE_SCOPES = {"study": "human-study"}


@dataclass(frozen=True)
class AccessCapability:
    scope: str
    semantic_hash: str
    roles: frozenset[str]
    token: str
    fencing_token: int


class ProtectedAccessAuthority:
    """SQLite authority for fenced protected-pool capabilities."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise AccessDenied("protected access authority cannot be a symlink")
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS reservations (
                    fencing_token INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope TEXT NOT NULL,
                    semantic_hash TEXT NOT NULL,
                    roles_json TEXT NOT NULL,
                    token TEXT NOT NULL UNIQUE,
                    state TEXT NOT NULL CHECK (state IN ('reserved', 'completed')),
                    reserved_at REAL NOT NULL,
                    first_read_at REAL,
                    completed_at REAL,
                    UNIQUE(scope, semantic_hash)
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @staticmethod
    def _from_row(row: sqlite3.Row) -> AccessCapability:
        return AccessCapability(
            scope=str(row["scope"]),
            semantic_hash=str(row["semantic_hash"]),
            roles=frozenset(json.loads(str(row["roles_json"]))),
            token=str(row["token"]),
            fencing_token=int(row["fencing_token"]),
        )

    def reserve(self, *, scope: str, semantic_hash: str, roles: set[str]) -> AccessCapability:
        """The fence for one (scope, semantic hash): minted once, returned again on retry."""
        if (
            not HASH_RE.fullmatch(semantic_hash)
            or not roles
            or any(ROLE_SCOPES.get(role) != scope for role in roles)
        ):
            raise AccessDenied("reservation scope, semantic hash, and roles are required")
        roles_json = json.dumps(sorted(roles), separators=(",", ":"))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM reservations WHERE scope=? AND semantic_hash=?",
                (scope, semantic_hash),
            ).fetchone()
            if existing is not None:
                if str(existing["roles_json"]) != roles_json:
                    raise AccessDenied("protected reservation roles do not match the existing fence")
                return self._from_row(existing)
            token = secrets.token_urlsafe(32)
            connection.execute(
                """INSERT INTO reservations
                   (scope, semantic_hash, roles_json, token, state, reserved_at)
                   VALUES (?, ?, ?, ?, 'reserved', ?)""",
                (scope, semantic_hash, roles_json, token, time.time()),
            )
            row = connection.execute("SELECT * FROM reservations WHERE token=?", (token,)).fetchone()
            assert row is not None
            return self._from_row(row)

    def validate_and_mark_read(
        self,
        capability: AccessCapability,
        *,
        semantic_hash: str,
        role: str,
    ) -> None:
        if ROLE_SCOPES.get(role) != capability.scope:
            raise AccessDenied(f"protected capability scope does not authorize role {role!r}")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM reservations WHERE fencing_token=?",
                (capability.fencing_token,),
            ).fetchone()
            if (
                row is None
                or str(row["token"]) != capability.token
                or str(row["scope"]) != capability.scope
                or str(row["semantic_hash"]) != capability.semantic_hash
                or frozenset(json.loads(str(row["roles_json"]))) != capability.roles
            ):
                raise AccessDenied("protected capability has a stale fencing token")
            if str(row["state"]) != "reserved":
                raise AccessDenied("protected capability is no longer reserved")
            if str(row["semantic_hash"]) != semantic_hash:
                raise AccessDenied("protected capability semantic hash does not match")
            roles = set(json.loads(str(row["roles_json"])))
            if role not in roles:
                raise AccessDenied(f"protected capability does not authorize role {role!r}")
            connection.execute(
                "UPDATE reservations SET first_read_at=COALESCE(first_read_at, ?) WHERE fencing_token=?",
                (time.time(), capability.fencing_token),
            )

    def complete(self, capability: AccessCapability) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """UPDATE reservations SET state='completed', completed_at=?
                   WHERE fencing_token=? AND token=? AND state='reserved'""",
                (time.time(), capability.fencing_token, capability.token),
            )
            if cursor.rowcount != 1:
                raise AccessDenied("protected capability is stale or already completed")
