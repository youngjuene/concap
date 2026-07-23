"""Configuration lock and test-once enforcement tests (PRD section 28)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dpo.contracts.study_contract import EXPERIMENT_IDS, TRACKS, StudyContract
from dpo.core.identity import semantic_hash, sha256_bytes
from dpo.pipeline.lock import (
    LockError,
    TestOnceResolver,
    create_lock_manifest,
    parse_lock_manifest,
)


def _lock(contract: StudyContract):  # noqa: ANN202
    checkpoint_hashes = {
        experiment_id: {track: sha256_bytes(f"{experiment_id}-{track}".encode()) for track in TRACKS}
        for experiment_id in EXPERIMENT_IDS
    }
    return create_lock_manifest(
        contract,
        checkpoint_hashes=checkpoint_hashes,
        processor_hash=sha256_bytes(b"processor"),
        preprocessing_hash=sha256_bytes(b"preprocessing"),
        evaluation_version="evaluation/v1",
        metric_versions={"compliance": "v1"},
        study_interface_version="study/v1",
        selection_report_hash=sha256_bytes(b"selection"),
    )


def test_lock_manifest_freezes_every_required_knob(contract: StudyContract) -> None:
    lock = _lock(contract)
    parsed = parse_lock_manifest(lock.document)
    assert parsed.lock_id == lock.lock_id
    assert lock.document["decoding_hash"] == semantic_hash(dict(contract.validation))
    assert lock.document["statistical_plan_hash"] == semantic_hash(dict(contract.analysis))


def test_lock_requires_full_matrix_coverage(contract: StudyContract) -> None:
    checkpoint_hashes = {
        experiment_id: {track: sha256_bytes(b"x") for track in TRACKS}
        for experiment_id in EXPERIMENT_IDS
        if experiment_id != "RDPO"
    }
    with pytest.raises(LockError, match="RDPO"):
        create_lock_manifest(
            contract,
            checkpoint_hashes=checkpoint_hashes,
            processor_hash=sha256_bytes(b"p"),
            preprocessing_hash=sha256_bytes(b"q"),
            evaluation_version="v1",
            metric_versions={"compliance": "v1"},
            study_interface_version="v1",
            selection_report_hash=sha256_bytes(b"s"),
        )


def test_test_once_resume_is_identical_only(contract: StudyContract, tmp_path: Path) -> None:
    lock = _lock(contract)
    resolver = TestOnceResolver(str(tmp_path / "test.sqlite"))
    split_hash = sha256_bytes(b"test-split")
    reservation = resolver.reserve(lock, test_split_hash=split_hash)
    resumed = resolver.reserve(lock, test_split_hash=split_hash)
    assert resumed == reservation
    with pytest.raises(LockError, match="exploratory or erratum"):
        resolver.reserve(lock, test_split_hash=sha256_bytes(b"a-different-split"))


def test_low_scores_cannot_reset_the_reservation(contract: StudyContract, tmp_path: Path) -> None:
    # The permitted-rerun list is crashes and corruption, resumed identically.
    # Changing decoding (part of the lock) changes the semantic and is refused.
    lock = _lock(contract)
    resolver = TestOnceResolver(str(tmp_path / "test.sqlite"))
    reservation = resolver.reserve(lock, test_split_hash=sha256_bytes(b"split"))
    resolver.mark_first_read(reservation)
    mutated = dict(lock.document)
    mutated["decoding_hash"] = sha256_bytes(b"new-decoding")
    with pytest.raises(LockError):
        resolver.reserve(parse_lock_manifest(mutated), test_split_hash=sha256_bytes(b"split"))
    resolver.complete(reservation)
    assert resolver.state(reservation) == "completed"
    with pytest.raises(LockError):
        resolver.complete(reservation)


def test_state_transitions_are_fenced(contract: StudyContract, tmp_path: Path) -> None:
    lock = _lock(contract)
    resolver = TestOnceResolver(str(tmp_path / "test.sqlite"))
    reservation = resolver.reserve(lock, test_split_hash=sha256_bytes(b"split"))
    with pytest.raises(LockError):
        resolver.complete(reservation)  # cannot complete before first read
    resolver.mark_first_read(reservation)
    record = resolver.authority_record(reservation)
    assert record["state"] == "read"
    assert record["first_read_at"] is not None
