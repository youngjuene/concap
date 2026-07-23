from __future__ import annotations

import fcntl
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


class FencedLeaseError(RuntimeError):
    """Raised when a lease is held or a caller presents stale ownership."""


@dataclass(frozen=True)
class LeaseOwner:
    host: str
    pid: int
    boot_id: str


@dataclass(frozen=True)
class GpuLease:
    gpu_uuid: str
    fencing_token: int
    owner: LeaseOwner


class GpuLeaseDatabase:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        init_lock = self.path.with_suffix(self.path.suffix + ".init.lock")
        with init_lock.open("a+b") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                with self._connect() as connection:
                    connection.executescript(
                        """
                        PRAGMA journal_mode=WAL;
                        CREATE TABLE IF NOT EXISTS counters (
                            gpu_uuid TEXT PRIMARY KEY,
                            last_fence INTEGER NOT NULL
                        );
                        CREATE TABLE IF NOT EXISTS leases (
                            gpu_uuid TEXT PRIMARY KEY,
                            fencing_token INTEGER NOT NULL,
                            host TEXT NOT NULL,
                            pid INTEGER NOT NULL,
                            boot_id TEXT NOT NULL,
                            heartbeat_at REAL NOT NULL,
                            expires_at REAL NOT NULL
                        );
                        """
                    )
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def acquire(
        self,
        gpu_uuid: str,
        owner: LeaseOwner,
        *,
        ttl_seconds: float,
        now: float | None = None,
    ) -> GpuLease:
        timestamp = time.time() if now is None else now
        if not gpu_uuid or not owner.host or owner.pid <= 0 or not owner.boot_id or ttl_seconds <= 0:
            raise FencedLeaseError("complete lease identity and positive TTL are required")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM leases WHERE gpu_uuid=?", (gpu_uuid,)).fetchone()
            if row is not None and float(row["expires_at"]) > timestamp:
                raise FencedLeaseError(f"GPU {gpu_uuid} already has an active lease")
            counter = connection.execute(
                "SELECT last_fence FROM counters WHERE gpu_uuid=?", (gpu_uuid,)
            ).fetchone()
            fence = (int(counter["last_fence"]) if counter is not None else 0) + 1
            connection.execute(
                """INSERT INTO counters(gpu_uuid, last_fence) VALUES (?, ?)
                   ON CONFLICT(gpu_uuid) DO UPDATE SET last_fence=excluded.last_fence""",
                (gpu_uuid, fence),
            )
            connection.execute(
                """INSERT INTO leases
                   (gpu_uuid, fencing_token, host, pid, boot_id, heartbeat_at, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(gpu_uuid) DO UPDATE SET
                     fencing_token=excluded.fencing_token, host=excluded.host, pid=excluded.pid,
                     boot_id=excluded.boot_id, heartbeat_at=excluded.heartbeat_at,
                     expires_at=excluded.expires_at""",
                (
                    gpu_uuid,
                    fence,
                    owner.host,
                    owner.pid,
                    owner.boot_id,
                    timestamp,
                    timestamp + ttl_seconds,
                ),
            )
            return GpuLease(gpu_uuid=gpu_uuid, fencing_token=fence, owner=owner)

    @staticmethod
    def _identity_matches(row: sqlite3.Row, lease: GpuLease, owner: LeaseOwner) -> bool:
        return (
            int(row["fencing_token"]) == lease.fencing_token
            and str(row["host"]) == owner.host
            and int(row["pid"]) == owner.pid
            and str(row["boot_id"]) == owner.boot_id
            and lease.owner == owner
        )

    def heartbeat(
        self,
        lease: GpuLease,
        owner: LeaseOwner,
        *,
        ttl_seconds: float,
        now: float | None = None,
    ) -> None:
        timestamp = time.time() if now is None else now
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM leases WHERE gpu_uuid=?", (lease.gpu_uuid,)).fetchone()
            if row is None or not self._identity_matches(row, lease, owner):
                raise FencedLeaseError("stale fencing token or lease owner")
            connection.execute(
                "UPDATE leases SET heartbeat_at=?, expires_at=? WHERE gpu_uuid=?",
                (timestamp, timestamp + ttl_seconds, lease.gpu_uuid),
            )

    def release(self, lease: GpuLease, owner: LeaseOwner) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM leases WHERE gpu_uuid=?", (lease.gpu_uuid,)).fetchone()
            if row is None or not self._identity_matches(row, lease, owner):
                raise FencedLeaseError("lease release has stale fencing token or wrong owner")
            connection.execute("DELETE FROM leases WHERE gpu_uuid=?", (lease.gpu_uuid,))
