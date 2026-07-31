"""The configuration lock (PRD section 28).

Before any test access: validation completes, the selection report is
approved, and the lock manifest freezes every knob that could move — the
selected checkpoints, the processor and preprocessing identities, the frozen
decoding budget, the evaluation and metric versions, and the exclusion
policy. The manifest's semantic hash is the identity any later confirmatory
phase must bind to.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from dpo.contracts.study_contract import EXPERIMENT_IDS, StudyContract
from dpo.core.identity import semantic_hash

HASH_PREFIX = "sha256:"


class LockError(ValueError):
    """Raised when the lock manifest is violated."""


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
        "exclusion_policy_hash",
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
    selection_report_hash: str,
) -> LockManifest:
    declared = set(contract.tracks)
    for experiment_id in EXPERIMENT_IDS:
        tracks = checkpoint_hashes.get(experiment_id)
        if tracks is None or set(tracks) != declared:
            raise LockError(
                f"lock manifest requires checkpoint hashes for {experiment_id} on"
                f" every declared track {sorted(declared)}"
            )
        for track in sorted(declared):
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
            experiment_id: {track: str(checkpoint_hashes[experiment_id][track]) for track in sorted(declared)}
            for experiment_id in EXPERIMENT_IDS
        },
        "processor_hash": processor_hash,
        "preprocessing_hash": preprocessing_hash,
        "decoding_hash": semantic_hash(dict(contract.validation)),
        "evaluation_version": evaluation_version,
        "metric_versions": dict(sorted(metric_versions.items())),
        "exclusion_policy_hash": semantic_hash(dict(contract.annotation)),
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
