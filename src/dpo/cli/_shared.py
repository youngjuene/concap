"""Cross-command plumbing: emission, operation loading, registry shard access."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from dpo.annotation.raw_annotations import AnnotationError
from dpo.candidates.candidate_records import CandidateError
from dpo.contracts.study_contract import (
    ContractError,
    StudyContract,
    load_contract,
)
from dpo.core.access import AccessCapability, AccessDenied, ProtectedAccessAuthority
from dpo.core.artifacts import (
    ArtifactError,
    ArtifactManifest,
    ArtifactStore,
)
from dpo.core.identity import repo_lock_hash
from dpo.core.safety import CheckpointSafetyError, DestructivePathError
from dpo.data.derive_pairs import ViewError
from dpo.data.leakage_audit import LeakageError
from dpo.data.split import SplitError
from dpo.pipeline.canary import CanaryError
from dpo.pipeline.lock import LockError
from dpo.pipeline.publishing import ArtifactPublisher
from dpo.pipeline.stages import StageError, allowed_contract_ids
from dpo.pipeline.study_stage import StudyError
from dpo.pipeline.training_stage import training_cell_contract_ids

Handler = Callable[[argparse.Namespace], int]

DOMAIN_ERRORS = (
    OSError,
    AccessDenied,
    AnnotationError,
    ArtifactError,
    CandidateError,
    CanaryError,
    ContractError,
    CheckpointSafetyError,
    DestructivePathError,
    LeakageError,
    LockError,
    SplitError,
    StageError,
    StudyError,
    ViewError,
)

DEFERRED_GATES = {
    "evaluate": "live model scoring requires a published backend authority",
}


def _emit(document: Mapping[str, object]) -> None:
    sys.stdout.write(json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def _deferred_gate(command: str, action: str) -> int:
    _emit(
        {
            "status": "blocked_pending_external_operation",
            "command": f"{command} {action}",
            "gate": DEFERRED_GATES[command],
            "side_effects": False,
            "canary_command": "dpo canary run --workspace <owned> --contract configs/study/canary.toml",
        }
    )
    return 3


@dataclass(frozen=True)
class _Operation:
    store: ArtifactStore
    contract: StudyContract
    manifests: tuple[ArtifactManifest, ...]

    def publisher(self) -> ArtifactPublisher:
        return ArtifactPublisher(self.store, self.contract, repo_lock_hash())


def _operation(arguments: argparse.Namespace, *, with_training_cells: bool = False) -> _Operation:
    store = ArtifactStore.create(arguments.workspace)
    contract = load_contract(arguments.contract)
    allowed = allowed_contract_ids(contract.raw, contract.contract_hash)
    if with_training_cells:
        # Matrix cells key on their resolved variant, not on a stage-wide slice;
        # recomputing the legitimate set proves a given cell came from this
        # contract instead of widening what any artifact may carry.
        allowed = allowed | training_cell_contract_ids(contract)
    manifests = []
    for artifact_id in getattr(arguments, "artifact_id", None) or []:
        manifest = store.verify(artifact_id)
        request = manifest.semantic.get("request")
        if not isinstance(request, Mapping) or request.get("contract_id") not in allowed:
            raise ArtifactError(f"artifact {artifact_id} was produced under a different contract")
        manifests.append(manifest)
    return _Operation(store=store, contract=contract, manifests=tuple(manifests))


def _require_types(operation: _Operation, accepted: set[str], *, minimum: int = 1) -> None:
    if len(operation.manifests) < minimum:
        raise ArtifactError(f"this command requires at least {minimum} input artifact id(s)")
    for manifest in operation.manifests:
        if manifest.artifact_type not in accepted:
            raise ArtifactError(
                f"input {manifest.artifact_id} has type {manifest.artifact_type!r};"
                f" expected one of {sorted(accepted)}"
            )


def _find_manifest(operation: _Operation, artifact_type: str) -> ArtifactManifest:
    for manifest in operation.manifests:
        if manifest.artifact_type == artifact_type:
            return manifest
    raise ArtifactError(f"this command requires a {artifact_type!r} input artifact")


def _registry_shard_ids(
    store: ArtifactStore,
    registry_manifest: ArtifactManifest,
    keep: Callable[[Mapping[str, object]], bool],
) -> dict[str, str]:
    """clip_id -> shard artifact id, from verified metadata alone.

    Membership metadata carries the clip id and its role, so a caller can learn
    WHICH clips a protected role holds — and check its own preconditions —
    without opening a payload and therefore without a capability. Reading what
    those clips contain still requires one.
    """
    registry_id = registry_manifest.semantic["request"]["parameters"]["registry_id"]
    found: dict[str, str] = {}
    for shard_id in store.find_by_type("dpo.clip-registry-shard/v1"):
        manifest = store.verify_metadata(shard_id)
        entry = manifest.semantic["registry_membership"][0]
        if entry["registry_id"] != registry_id or not keep(entry):
            continue
        found[str(entry["clip_id"])] = shard_id
    return found


def _registry_shard_rows(
    store: ArtifactStore,
    registry_manifest: ArtifactManifest,
    keep: Callable[[Mapping[str, object]], bool],
    *,
    capability: AccessCapability | None = None,
    authority: ProtectedAccessAuthority | None = None,
    semantic_hash_for_read: str | None = None,
) -> dict[str, tuple[str, dict[str, Any]]]:
    """clip_id -> (shard artifact id, locked registry row) for one registry.

    The full clip-registry payload spans the protected test/study roles and
    stays sealed to capability-free readers; per-clip shards carry exactly one
    role each, so only the shards ``keep`` selects (from their verified
    membership metadata) are ever opened.

    Selecting a protected role additionally requires a fenced capability, which
    the caller reserves and completes around the whole read — passing it here
    only authorizes the shard payloads this call opens.
    """
    registry_id = registry_manifest.semantic["request"]["parameters"]["registry_id"]
    rows: dict[str, tuple[str, dict[str, Any]]] = {}
    for shard_id in store.find_by_type("dpo.clip-registry-shard/v1"):
        manifest = store.verify_metadata(shard_id)
        entry = manifest.semantic["registry_membership"][0]
        if entry["registry_id"] != registry_id or not keep(entry):
            continue
        row = json.loads(
            store.read_payload(
                shard_id,
                capability=capability,
                authority=authority,
                semantic_hash=semantic_hash_for_read,
            )
        )
        rows[str(row["clip_id"])] = (shard_id, row)
    return rows


def _blocked(command: str) -> Handler:
    def handler(arguments: argparse.Namespace) -> int:
        del arguments
        return _deferred_gate(command, "run")

    return handler
