"""Configuration lock and test-once enforcement (PRD section 28).

Before any test payload is readable: validation completes, the selection
report is approved, the lock manifest freezes every knob that could move, and
the singleton test reservation binds the lock, the test split, the generation
configuration, and the statistical plan into one semantic hash. Only an
identical resume may touch the reservation again; a low score, a surprising
ranking, or the wish to change decoding are not permitted reruns.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from dpo.contracts.study_contract import EXPERIMENT_IDS, TRACKS, StudyContract
from dpo.core.identity import semantic_hash

HASH_PREFIX = "sha256:"


class LockError(ValueError):
    """Raised when the lock manifest or the test-once authority is violated."""


@dataclass(frozen=True)
class LockManifest:
    document: dict[str, object]
    lock_id: str


REQUIRED_LOCK_KEYS = frozenset(
    {
        "schema",
        "contract_hash",
        "checkpoint_hashes",
        "processor_hash",
        "preprocessing_hash",
        "decoding_hash",
        "evaluation_version",
        "metric_versions",
        "study_interface_version",
        "exclusion_policy_hash",
        "statistical_plan_hash",
        "selection_report_hash",
    }
)


def create_lock_manifest(
    contract: StudyContract,
    *,
    checkpoint_hashes: Mapping[str, Mapping[str, str]],
    processor_hash: str,
    preprocessing_hash: str,
    evaluation_version: str,
    metric_versions: Mapping[str, str],
    study_interface_version: str,
    selection_report_hash: str,
) -> LockManifest:
    for experiment_id in EXPERIMENT_IDS:
        tracks = checkpoint_hashes.get(experiment_id)
        if tracks is None or set(tracks) != set(TRACKS):
            raise LockError(f"lock manifest requires checkpoint hashes for {experiment_id} on both tracks")
        for track in TRACKS:
            value = str(tracks[track])
            if not value.startswith(HASH_PREFIX):
                raise LockError(f"checkpoint hash for {experiment_id}/{track} must be a sha256 hash")
    for name, value in (
        ("processor_hash", processor_hash),
        ("preprocessing_hash", preprocessing_hash),
        ("selection_report_hash", selection_report_hash),
    ):
        if not str(value).startswith(HASH_PREFIX):
            raise LockError(f"lock manifest {name} must be a sha256 hash")
    if not metric_versions:
        raise LockError("lock manifest requires explicit metric versions")
    document: dict[str, object] = {
        "schema": "dpo.lock-manifest/v1",
        "contract_hash": contract.contract_hash,
        "checkpoint_hashes": {
            experiment_id: {track: str(checkpoint_hashes[experiment_id][track]) for track in TRACKS}
            for experiment_id in EXPERIMENT_IDS
        },
        "processor_hash": processor_hash,
        "preprocessing_hash": preprocessing_hash,
        "decoding_hash": semantic_hash(dict(contract.validation)),
        "evaluation_version": evaluation_version,
        "metric_versions": dict(sorted(metric_versions.items())),
        "study_interface_version": study_interface_version,
        "exclusion_policy_hash": semantic_hash(dict(contract.annotation)),
        "statistical_plan_hash": semantic_hash(dict(contract.analysis)),
        "selection_report_hash": selection_report_hash,
    }
    return LockManifest(document=document, lock_id=semantic_hash(document))


def parse_lock_manifest(document: Mapping[str, object]) -> LockManifest:
    if set(document) != REQUIRED_LOCK_KEYS:
        raise LockError("lock manifest has an unexpected field set")
    if document.get("schema") != "dpo.lock-manifest/v1":
        raise LockError("lock manifest schema is unknown")
    body = dict(document)
    return LockManifest(document=body, lock_id=semantic_hash(body))


@dataclass(frozen=True)
class TestReservation:
    reservation_id: str
    semantic_hash: str
    fencing_token: int


class TestOnceResolver:
    """SQLite singleton binding one lock manifest to one confirmatory test run."""

    __test__ = False

    def __init__(self, path: str) -> None:
        self.path = path
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS reservation (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    reservation_id TEXT NOT NULL,
                    semantic_hash TEXT NOT NULL,
                    fencing_token INTEGER NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('reserved','read','completed')),
                    reserved_at REAL NOT NULL DEFAULT 0,
                    first_read_at REAL,
                    completed_at REAL
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def reserve(self, lock: LockManifest, *, test_split_hash: str) -> TestReservation:
        semantic = semantic_hash(
            {
                "schema": "test-once/v1",
                "lock_manifest": lock.lock_id,
                "test_split": test_split_hash,
                "generation": lock.document["decoding_hash"],
                "evaluation": lock.document["evaluation_version"],
                "analysis": lock.document["statistical_plan_hash"],
            }
        )
        reservation_id = semantic_hash({"semantic_hash": semantic, "kind": "reservation"})
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM reservation WHERE singleton=1").fetchone()
            if row is None:
                connection.execute(
                    """INSERT INTO reservation
                       (singleton, reservation_id, semantic_hash, fencing_token, state, reserved_at)
                       VALUES (1, ?, ?, 1, 'reserved', ?)""",
                    (reservation_id, semantic, time.time()),
                )
                return TestReservation(reservation_id, semantic, 1)
            if str(row["semantic_hash"]) != semantic:
                raise LockError(
                    "confirmatory test is already reserved for a different configuration;"
                    " a semantic change requires exploratory or erratum lineage, not a second"
                    " confirmatory run"
                )
            return TestReservation(str(row["reservation_id"]), semantic, int(row["fencing_token"]))

    def _transition(self, reservation: TestReservation, source: str, target: str) -> None:
        with self._connect() as connection:
            timestamp_column = {"read": "first_read_at", "completed": "completed_at"}.get(target)
            assignment = (
                "state=?"
                if timestamp_column is None
                else f"state=?, {timestamp_column}=COALESCE({timestamp_column}, ?)"
            )
            values: tuple[object, ...] = (
                (
                    target,
                    reservation.reservation_id,
                    reservation.semantic_hash,
                    reservation.fencing_token,
                    source,
                )
                if timestamp_column is None
                else (
                    target,
                    time.time(),
                    reservation.reservation_id,
                    reservation.semantic_hash,
                    reservation.fencing_token,
                    source,
                )
            )
            cursor = connection.execute(
                f"""UPDATE reservation SET {assignment} WHERE singleton=1 AND reservation_id=?
                   AND semantic_hash=? AND fencing_token=? AND state=?""",
                values,
            )
            if cursor.rowcount != 1:
                raise LockError("test reservation state or fencing token is invalid")

    def mark_first_read(self, reservation: TestReservation) -> None:
        self._transition(reservation, "reserved", "read")

    def complete(self, reservation: TestReservation) -> None:
        self._transition(reservation, "read", "completed")

    def state(self, reservation: TestReservation) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state, semantic_hash, fencing_token FROM reservation WHERE reservation_id=?",
                (reservation.reservation_id,),
            ).fetchone()
        if (
            row is None
            or str(row["semantic_hash"]) != reservation.semantic_hash
            or int(row["fencing_token"]) != reservation.fencing_token
        ):
            raise LockError("test reservation identity is invalid")
        return str(row["state"])

    def authority_record(self, reservation: TestReservation) -> dict[str, object]:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT reservation_id, semantic_hash, fencing_token, state,
                          reserved_at, first_read_at, completed_at
                   FROM reservation WHERE singleton=1"""
            ).fetchone()
        if (
            row is None
            or str(row["reservation_id"]) != reservation.reservation_id
            or str(row["semantic_hash"]) != reservation.semantic_hash
            or int(row["fencing_token"]) != reservation.fencing_token
        ):
            raise LockError("test reservation identity is invalid")
        return {
            "reservation_id": reservation.reservation_id,
            "semantic_hash": reservation.semantic_hash,
            "fencing_token": reservation.fencing_token,
            "state": str(row["state"]),
            "reserved_at": float(row["reserved_at"]),
            "first_read_at": None if row["first_read_at"] is None else float(row["first_read_at"]),
            "completed_at": None if row["completed_at"] is None else float(row["completed_at"]),
        }


def test_split_semantic_hash(test_clips: Iterable[str]) -> str:
    """The one formula binding a test-once reservation to its test split.

    Shared by the CLI and the canary; if it were computed twice and drifted,
    an identical resume of the one-shot confirmatory read would strand.
    """
    return semantic_hash({"test_clips": sorted(test_clips)})
