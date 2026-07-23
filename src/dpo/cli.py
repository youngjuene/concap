"""The dpo command line: JSON in, JSON out, fail closed.

Every command prints exactly one JSON document. Exit codes: 0 success, 2
domain/usage error, 3 blocked pending an external operation (real GPU
training, live model scoring). Commands that publish take content-addressed
artifact ids as inputs and reject any input from a different contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from dpo.annotation.raw_annotations import AnnotationError
from dpo.candidates.candidate_records import CandidateError
from dpo.contracts.study_contract import ContractError, StudyContract, load_contract
from dpo.core.access import AccessDenied
from dpo.core.artifacts import (
    GC_ROOT_TYPES,
    ArtifactError,
    ArtifactManifest,
    ArtifactStore,
)
from dpo.core.identity import canonical_bytes, repo_lock_hash
from dpo.core.safety import DestructivePathError
from dpo.data.derive_pairs import ViewError
from dpo.data.leakage_audit import LeakageError
from dpo.data.split import ClipInput, SplitError
from dpo.pipeline.canary import CanaryError, run_canary
from dpo.pipeline.corpus_stage import publish_corpus_ingest, publish_lock_splits
from dpo.pipeline.lock import LockError
from dpo.pipeline.publishing import ArtifactPublisher
from dpo.pipeline.stages import STAGES, StageError, allowed_contract_ids, lineage_edges, stage

Handler = Callable[[argparse.Namespace], int]

DOMAIN_ERRORS = (
    OSError,
    AccessDenied,
    AnnotationError,
    ArtifactError,
    CandidateError,
    CanaryError,
    ContractError,
    DestructivePathError,
    LeakageError,
    LockError,
    SplitError,
    StageError,
    ViewError,
)

DEFERRED_GATES = {
    "train": "real GPU training requires a published backend authority and a CUDA lease",
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


def _operation(arguments: argparse.Namespace) -> _Operation:
    store = ArtifactStore.create(arguments.workspace)
    contract = load_contract(arguments.contract)
    allowed = allowed_contract_ids(contract.raw, contract.contract_hash)
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


# ---------------------------------------------------------------------------
# Handlers.
# ---------------------------------------------------------------------------


def _contract_validate(arguments: argparse.Namespace) -> int:
    contract = load_contract(arguments.contract)
    _emit(
        {
            "status": "ok",
            "contract_hash": contract.contract_hash,
            "execution_class": contract.execution_class,
            "tracks": sorted(contract.tracks),
            "seeds": list(contract.seeds),
        }
    )
    return 0


def _contract_lock(arguments: argparse.Namespace) -> int:
    from dpo.core.atomic import atomic_write_bytes

    contract = load_contract(arguments.contract)
    document = {
        "schema": "dpo.contract-lock/v1",
        "contract_hash": contract.contract_hash,
        "contract": contract.raw,
        "environment_lock": repo_lock_hash(),
    }
    payload = canonical_bytes(document) + b"\n"
    written = atomic_write_bytes(arguments.out, payload)
    _emit(
        {
            "status": "published" if written else "cached",
            "contract_hash": contract.contract_hash,
            "out": str(Path(arguments.out).resolve()),
        }
    )
    return 0


def _canary_run(arguments: argparse.Namespace) -> int:
    result = run_canary(arguments.workspace, arguments.contract)
    _emit(
        {
            "status": result.report.get("status", "unknown"),
            "artifact_id": result.artifact_id,
            "cached": result.cached,
            "provider_calls": result.provider_calls,
            "report": result.report,
        }
    )
    return 0


def _artifact_verify(arguments: argparse.Namespace) -> int:
    store = ArtifactStore.open(arguments.workspace)
    if arguments.all:
        store.rebuild_index()
        verified = sorted(store.indexed_ids())
        for artifact_id in verified:
            store.verify(artifact_id)
        _emit({"status": "ok", "verified": len(verified)})
        return 0
    manifest = store.verify(arguments.artifact_id)
    _emit(
        {
            "status": "ok",
            "artifact_id": manifest.artifact_id,
            "artifact_type": manifest.artifact_type,
            "role_exposure": sorted(manifest.role_exposure),
        }
    )
    return 0


def _artifact_trace(arguments: argparse.Namespace) -> int:
    store = ArtifactStore.open(arguments.workspace)
    lineage = store.trace(arguments.artifact_id)
    _emit(
        {
            "status": "ok",
            "artifact_id": arguments.artifact_id,
            "lineage": [
                {"artifact_id": manifest.artifact_id, "artifact_type": manifest.artifact_type}
                for manifest in lineage
            ],
        }
    )
    return 0


def _artifact_rebuild_index(arguments: argparse.Namespace) -> int:
    store = ArtifactStore.open(arguments.workspace)
    store.rebuild_index()
    _emit({"status": "ok", "indexed": len(store.indexed_ids())})
    return 0


def _artifact_gc(arguments: argparse.Namespace) -> int:
    store = ArtifactStore.open(arguments.workspace)
    locks = set(arguments.lock or [])
    if arguments.execute and not locks:
        locks = {
            artifact_id
            for artifact_type in GC_ROOT_TYPES
            for artifact_id in store.find_by_type(artifact_type)
        }
    candidates = store.gc(locks, execute=arguments.execute)
    _emit(
        {
            "status": "ok",
            "executed": bool(arguments.execute),
            "candidates": candidates,
        }
    )
    return 0


def _stage_list(arguments: argparse.Namespace) -> int:
    del arguments
    _emit(
        {
            "status": "ok",
            "stages": {
                name: {
                    "inputs": list(stage.input_artifact_types),
                    "outputs": list(stage.output_artifact_types),
                    "description": stage.description,
                }
                for name, stage in sorted(STAGES.items())
            },
            "lineage": [list(edge) for edge in lineage_edges()],
        }
    )
    return 0


def _read_clip_rows(path: str) -> list[ClipInput]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SplitError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise SplitError(f"{path}:{line_number}: expected a JSON object")
            allowed = {
                "clip_id",
                "source_video_id",
                "media_hash",
                "start_ms",
                "end_ms",
                "derivative_hashes",
                "link_group",
                "asserted_role",
            }
            unknown = sorted(set(value) - allowed)
            if unknown:
                raise SplitError(f"{path}:{line_number}: unknown field {unknown[0]!r}")
            try:
                rows.append(ClipInput.from_document(value))
            except SplitError as exc:
                raise SplitError(f"{path}:{line_number}: {exc}") from exc
    if not rows:
        raise SplitError(f"{path}: no clip rows")
    return rows


def _corpus_ingest(arguments: argparse.Namespace) -> int:
    operation = _operation(arguments)
    clips = _read_clip_rows(arguments.input)
    artifact_id = publish_corpus_ingest(operation.publisher(), clips)
    _emit(
        {
            "status": "published",
            "operation": "corpus-ingest",
            "artifact_id": artifact_id,
            "artifact_type": "dpo.corpus-ingest/v1",
            "row_count": len(clips),
        }
    )
    return 0


def _corpus_lock_splits(arguments: argparse.Namespace) -> int:
    operation = _operation(arguments)
    _require_types(operation, set(stage("lock-splits").input_artifact_types))
    ingest = operation.manifests[0]
    payload = json.loads(operation.store.read_payload(ingest.artifact_id))
    clips = [ClipInput.from_document(row) for row in payload["rows"]]
    registry_artifact, manifest, shard_ids = publish_lock_splits(
        operation.publisher(),
        operation.contract,
        clips,
        ingest_artifact_id=ingest.artifact_id,
    )
    _emit(
        {
            "status": "published",
            "operation": "lock-splits",
            "artifact_id": registry_artifact,
            "artifact_type": "dpo.clip-registry/v1",
            "split_manifest": manifest.signed_document(),
            "shards": shard_ids,
        }
    )
    return 0


def _blocked(command: str) -> Handler:
    def handler(arguments: argparse.Namespace) -> int:
        del arguments
        return _deferred_gate(command, "run")

    return handler


# ---------------------------------------------------------------------------
# Parser.
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dpo", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    contract = commands.add_parser("contract", help="validate or lock the study contract")
    contract_actions = contract.add_subparsers(dest="action", required=True)
    validate = contract_actions.add_parser("validate")
    validate.add_argument("--contract", required=True)
    validate.set_defaults(handler=_contract_validate)
    lock = contract_actions.add_parser("lock")
    lock.add_argument("--contract", required=True)
    lock.add_argument("--out", required=True)
    lock.set_defaults(handler=_contract_lock)

    canary = commands.add_parser("canary", help="run the offline end-to-end canary")
    canary_actions = canary.add_subparsers(dest="action", required=True)
    canary_run = canary_actions.add_parser("run")
    canary_run.add_argument("--workspace", required=True)
    canary_run.add_argument("--contract", required=True)
    canary_run.set_defaults(handler=_canary_run)

    artifact = commands.add_parser("artifact", help="verify, trace, and collect artifacts")
    artifact_actions = artifact.add_subparsers(dest="action", required=True)
    verify = artifact_actions.add_parser("verify")
    verify.add_argument("--workspace", required=True)
    group = verify.add_mutually_exclusive_group(required=True)
    group.add_argument("--artifact-id")
    group.add_argument("--all", action="store_true")
    verify.set_defaults(handler=_artifact_verify)
    trace = artifact_actions.add_parser("trace")
    trace.add_argument("--workspace", required=True)
    trace.add_argument("--artifact-id", required=True)
    trace.set_defaults(handler=_artifact_trace)
    rebuild = artifact_actions.add_parser("rebuild-index")
    rebuild.add_argument("--workspace", required=True)
    rebuild.set_defaults(handler=_artifact_rebuild_index)
    gc = artifact_actions.add_parser("gc")
    gc.add_argument("--workspace", required=True)
    gc.add_argument("--lock", action="append")
    gc.add_argument("--execute", action="store_true")
    gc.set_defaults(handler=_artifact_gc)

    stage = commands.add_parser("stage", help="show the stage registry and lineage")
    stage_actions = stage.add_subparsers(dest="action", required=True)
    stage_list = stage_actions.add_parser("list")
    stage_list.set_defaults(handler=_stage_list)

    corpus = commands.add_parser("corpus", help="ingest clips and lock the immutable splits")
    corpus_actions = corpus.add_subparsers(dest="action", required=True)
    ingest = corpus_actions.add_parser("ingest")
    ingest.add_argument("--workspace", required=True)
    ingest.add_argument("--contract", required=True)
    ingest.add_argument("--input", required=True)
    ingest.set_defaults(handler=_corpus_ingest)
    lock_splits = corpus_actions.add_parser("lock-splits")
    lock_splits.add_argument("--workspace", required=True)
    lock_splits.add_argument("--contract", required=True)
    lock_splits.add_argument("--artifact-id", action="append", required=True)
    lock_splits.set_defaults(handler=_corpus_lock_splits)

    for command_name in ("train", "evaluate"):
        blocked = commands.add_parser(command_name, help=f"live boundary: {DEFERRED_GATES[command_name]}")
        blocked_actions = blocked.add_subparsers(dest="action", required=True)
        run = blocked_actions.add_parser("run")
        run.add_argument("--workspace", required=True)
        run.add_argument("--contract", required=True)
        run.add_argument("--artifact-id", action="append")
        run.add_argument("--track")
        run.add_argument("--invoke-external", action="store_true")
        run.set_defaults(handler=_blocked(command_name))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    handler: Handler = arguments.handler
    try:
        return handler(arguments)
    except DOMAIN_ERRORS as exc:
        parser.exit(2, f"dpo: error: {exc}\n")
        return 2  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
