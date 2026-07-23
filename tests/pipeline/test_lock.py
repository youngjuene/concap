"""Configuration-lock manifest tests (PRD section 28)."""

from __future__ import annotations

import pytest

from dpo.contracts.study_contract import EXPERIMENT_IDS, TRACKS, StudyContract
from dpo.core.identity import semantic_hash, sha256_bytes
from dpo.pipeline.lock import (
    LockError,
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
        selection_report_hash=sha256_bytes(b"selection"),
    )


def test_lock_manifest_freezes_every_required_knob(contract: StudyContract) -> None:
    lock = _lock(contract)
    parsed = parse_lock_manifest(lock.document)
    assert parsed.lock_id == lock.lock_id
    assert lock.document["decoding_hash"] == semantic_hash(dict(contract.validation))
    assert lock.document["exclusion_policy_hash"] == semantic_hash(dict(contract.annotation))


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
            selection_report_hash=sha256_bytes(b"s"),
        )


def test_lock_manifest_rejects_an_unexpected_field_set(contract: StudyContract) -> None:
    lock = _lock(contract)
    mutated = dict(lock.document)
    mutated["surprise"] = "x"
    with pytest.raises(LockError, match="unexpected field set"):
        parse_lock_manifest(mutated)
