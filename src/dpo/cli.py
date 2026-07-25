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
from typing import Any

from dpo.annotation.aggregate import aggregate_all
from dpo.annotation.collection_tasks import (
    annotations_from_responses,
    attention_expectations,
    audio_presentations_from_registry,
    build_collection_tasks,
)
from dpo.annotation.raw_annotations import AnnotationError
from dpo.annotation.reliability import (
    ReliabilityReport,
    parse_reliability_report,
    retained_annotations,
)
from dpo.candidates.audit import audit_candidate, resolve_audits
from dpo.candidates.candidate_records import CandidateError
from dpo.candidates.freeze import parse_frozen_pool
from dpo.candidates.generation import (
    AUDIO_MEDIA_SUFFIXES,
    GEMMA_IMPLEMENTATION,
    TINY_IMPLEMENTATION,
    VIDEO_MEDIA_SUFFIXES,
    generate_c0_candidates,
    generate_c0_candidates_gemma,
    resolve_media_files,
    verify_backend_pin,
)
from dpo.contracts.study_contract import (
    EXPERIMENT_IDS,
    TRACKS,
    ContractError,
    StudyContract,
    load_contract,
)
from dpo.core.access import AccessDenied
from dpo.core.artifacts import (
    GC_ROOT_TYPES,
    ArtifactError,
    ArtifactManifest,
    ArtifactStore,
)
from dpo.core.identity import canonical_bytes, repo_lock_hash, semantic_hash, sha256_file
from dpo.core.safety import DestructivePathError
from dpo.data.derive_pairs import ViewError
from dpo.data.leakage_audit import LeakageError
from dpo.data.split import ClipInput, SplitError, parse_split_manifest
from dpo.pipeline.annotation_stage import ingest_annotations
from dpo.pipeline.canary import CanaryError, run_canary
from dpo.pipeline.candidate_stage import publish_frozen_pool
from dpo.pipeline.corpus_stage import publish_corpus_ingest, publish_lock_splits
from dpo.pipeline.experiments import expand_experiment
from dpo.pipeline.live_runner import (
    LiveMatrixRunner,
    TinyBackend,
    TrainingBackend,
    load_checkpoint_policies,
)
from dpo.pipeline.lock import LockError, parse_lock_manifest
from dpo.pipeline.publishing import ArtifactPublisher
from dpo.pipeline.run_matrix import DEFAULT_MEDIA_DIM
from dpo.pipeline.selection_stage import publish_selection
from dpo.pipeline.stage_inputs import (
    collect_matrix_cells,
    collect_view_inputs,
    parse_annotations,
    request_parameters,
)
from dpo.pipeline.stages import STAGES, StageError, allowed_contract_ids, lineage_edges, stage
from dpo.pipeline.training_stage import publish_training_matrix, training_cell_contract_ids
from dpo.pipeline.view_stage import publish_track_views

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


def _find_manifest(operation: _Operation, artifact_type: str) -> ArtifactManifest:
    for manifest in operation.manifests:
        if manifest.artifact_type == artifact_type:
            return manifest
    raise ArtifactError(f"this command requires a {artifact_type!r} input artifact")


def _registry_shard_rows(
    store: ArtifactStore,
    registry_manifest: ArtifactManifest,
    keep: Callable[[Mapping[str, object]], bool],
) -> dict[str, tuple[str, dict[str, Any]]]:
    """clip_id -> (shard artifact id, locked registry row) for one registry.

    The full clip-registry payload spans the protected test/study roles and
    stays sealed to capability-free readers; per-clip shards carry exactly one
    role each, so only the shards ``keep`` selects (from their verified
    membership metadata) are ever opened.
    """
    registry_id = registry_manifest.semantic["request"]["parameters"]["registry_id"]
    rows: dict[str, tuple[str, dict[str, Any]]] = {}
    for shard_id in store.find_by_type("dpo.clip-registry-shard/v1"):
        manifest = store.verify_metadata(shard_id)
        entry = manifest.semantic["registry_membership"][0]
        if entry["registry_id"] != registry_id or not keep(entry):
            continue
        row = json.loads(store.read_payload(shard_id))
        rows[str(row["clip_id"])] = (shard_id, row)
    return rows


def _annotation_export_tasks(arguments: argparse.Namespace) -> int:
    operation = _operation(arguments)
    _require_types(operation, {"dpo.frozen-candidate-pool/v1", "dpo.clip-registry/v1"}, minimum=2)
    pool_manifest = _find_manifest(operation, "dpo.frozen-candidate-pool/v1")
    registry_manifest = _find_manifest(operation, "dpo.clip-registry/v1")
    pool = parse_frozen_pool(operation.store.read_payload(pool_manifest.artifact_id))
    pool_clips = {candidate.clip_id for candidate in pool.candidates}
    shard_rows = _registry_shard_rows(
        operation.store, registry_manifest, lambda entry: str(entry["clip_id"]) in pool_clips
    )
    presentations = audio_presentations_from_registry([row for _, row in shard_rows.values()])
    tasks_document, answers_document = build_collection_tasks(
        operation.contract, pool, audio_presentations=presentations
    )
    Path(arguments.out_tasks).write_text(
        json.dumps(tasks_document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    Path(arguments.out_answers).write_text(
        json.dumps(answers_document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    task_rows = tasks_document["tasks"]
    assert isinstance(task_rows, list)
    kinds: dict[str, int] = {}
    for task in task_rows:
        kinds[str(task["kind"])] = kinds.get(str(task["kind"]), 0) + 1
    _emit(
        {
            "status": "exported",
            "operation": "annotation-export-tasks",
            "track": pool.track,
            "tasks": len(task_rows),
            "kinds": kinds,
            "out_tasks": str(Path(arguments.out_tasks).resolve()),
            "out_answers": str(Path(arguments.out_answers).resolve()),
        }
    )
    return 0


def _annotation_ingest(arguments: argparse.Namespace) -> int:
    operation = _operation(arguments)
    _require_types(operation, {"dpo.frozen-candidate-pool/v1"})
    pool_manifest = _find_manifest(operation, "dpo.frozen-candidate-pool/v1")
    pool = parse_frozen_pool(operation.store.read_payload(pool_manifest.artifact_id))
    tasks_document = json.loads(Path(arguments.tasks).read_text(encoding="utf-8"))
    answers_document = json.loads(Path(arguments.answers).read_text(encoding="utf-8"))
    annotations = []
    for responses_path in arguments.responses:
        responses_document = json.loads(Path(responses_path).read_text(encoding="utf-8"))
        annotations.extend(annotations_from_responses(tasks_document, responses_document))
    retained, aggregates, artifact_ids = ingest_annotations(
        operation.publisher(),
        operation.contract,
        track=pool.track,
        split=arguments.split,
        pool=pool,
        pool_artifact_id=pool_manifest.artifact_id,
        annotations=annotations,
        attention_expected=attention_expectations(answers_document),
    )
    _emit(
        {
            "status": "published",
            "operation": "annotation-ingest",
            "track": pool.track,
            "split": arguments.split,
            "annotations": len(annotations),
            "retained": len(retained),
            "aggregated_pairs": len(aggregates),
            "artifacts": artifact_ids,
        }
    )
    return 0


def _annotation_serve(arguments: argparse.Namespace) -> int:
    from dpo.annotation.webapp import run_collection_app

    run_collection_app(
        tasks_path=Path(arguments.tasks),
        media_dir=Path(arguments.media_dir),
        out_path=Path(arguments.out),
        host=arguments.host,
        port=arguments.port,
    )
    return 0


def _candidates_generate(arguments: argparse.Namespace) -> int:
    contract = load_contract(arguments.contract)
    implementation = str(contract.raw["models"]["seed"]["implementation"])
    if implementation == TINY_IMPLEMENTATION:
        if arguments.backend_config or arguments.media_dir:
            raise ArtifactError("the tiny seed backend takes no --backend-config/--media-dir")
    elif implementation == GEMMA_IMPLEMENTATION:
        if not arguments.backend_config or not arguments.media_dir:
            raise ArtifactError("the Gemma seed backend requires --backend-config and --media-dir")
        import torch

        if not torch.cuda.is_available():
            _emit(
                {
                    "status": "blocked_pending_external_operation",
                    "command": "candidates generate",
                    "gate": "Gemma seed-model generation requires a CUDA device",
                    "side_effects": False,
                }
            )
            return 3
    else:
        _emit(
            {
                "status": "blocked_pending_external_operation",
                "command": "candidates generate",
                "gate": f"seed-model implementation {implementation!r} has no wired generation backend",
                "side_effects": False,
            }
        )
        return 3
    operation = _operation(arguments)
    _require_types(operation, {"dpo.clip-registry/v1"})
    registry_manifest = _find_manifest(operation, "dpo.clip-registry/v1")
    membership = registry_manifest.semantic["registry_membership"]
    split_clips = sorted(
        str(entry["clip_id"]) for entry in membership if str(entry["role"]) == arguments.split
    )
    if not split_clips:
        raise ArtifactError(f"the clip registry assigns no clips to split {arguments.split!r}")
    if implementation == TINY_IMPLEMENTATION:
        candidates = generate_c0_candidates(operation.contract, track=arguments.track, clip_ids=split_clips)
    else:
        candidates = generate_c0_candidates_gemma(
            operation.contract,
            track=arguments.track,
            clip_ids=split_clips,
            backend_config_path=Path(arguments.backend_config),
            media_dir=Path(arguments.media_dir),
        )
    caption_contract = operation.contract.tracks[arguments.track]
    audits = resolve_audits([audit_candidate(record, contract=caption_contract) for record in candidates], [])
    shard_rows = _registry_shard_rows(
        operation.store, registry_manifest, lambda entry: str(entry["role"]) == arguments.split
    )
    shard_ids = {clip_id: shard_id for clip_id, (shard_id, _) in shard_rows.items()}
    missing = sorted(set(split_clips) - set(shard_ids))
    if missing:
        raise ArtifactError(f"split clip {missing[0]!r} has no locked registry shard")
    pool, artifact_id = publish_frozen_pool(
        operation.publisher(),
        operation.contract,
        track=arguments.track,
        split=arguments.split,
        candidates=candidates,
        audits=audits,
        shard_artifact_ids=shard_ids,
        dataset_version=arguments.dataset_version,
        audit_version="audit/v1",
    )
    _emit(
        {
            "status": "published",
            "operation": "candidates-generate",
            "track": arguments.track,
            "split": arguments.split,
            "clips": len(split_clips),
            "candidates": len(pool.candidates),
            "pairs": len(pool.pairs),
            "artifact_id": artifact_id,
        }
    )
    return 0


def _corpus_clips(store: ArtifactStore, registry: ArtifactManifest) -> list[ClipInput]:
    """The full corpus behind a locked registry, read from its ingest ancestor.

    The leakage audit is a whole-corpus check (one source video, one split), so
    it needs every clip row — including the protected roles, whose registry
    shards stay sealed. The registry's own corpus-ingest parent carries exactly
    the rows the splits were computed from and asserts no role at all.
    """
    for parent in registry.parents:
        manifest = store.verify(parent.artifact_id)
        if manifest.artifact_type != "dpo.corpus-ingest/v1":
            continue
        payload = json.loads(store.read_payload(manifest.artifact_id))
        rows = payload["rows"]
        if not isinstance(rows, list):
            raise ArtifactError("corpus-ingest payload rows must be an array")
        return [ClipInput.from_document(row) for row in rows]
    raise ArtifactError(
        f"clip registry {registry.artifact_id} has no corpus-ingest ancestor;"
        " the whole-corpus leakage audit cannot run"
    )


def _split_of(manifest: ArtifactManifest, parameters: Mapping[str, object]) -> str:
    split = parameters.get("split")
    if not isinstance(split, str) or split not in {"train", "validation"}:
        raise ArtifactError(f"artifact {manifest.artifact_id} does not declare a train/validation split")
    return split


def _reliability_report_for(store: ArtifactStore, annotations_artifact_id: str) -> ReliabilityReport:
    """The published screening decision for one raw-annotations artifact.

    Views must retain exactly the annotations the annotation stage retained, and
    the exclusion decision needs the attention-check expectations that live only
    in the restricted answers document. Reading the published report — the one
    artifact parented on these annotations — reuses the decision instead of
    approximating it.
    """
    matches = [
        artifact_id
        for artifact_id in store.find_by_type("dpo.reliability-report/v1")
        if any(
            parent.artifact_id == annotations_artifact_id
            for parent in store.verify_metadata(artifact_id).parents
        )
    ]
    if len(matches) != 1:
        raise ArtifactError(
            f"expected exactly one reliability report for annotations {annotations_artifact_id};"
            f" found {len(matches)}"
        )
    return parse_reliability_report(store.read_payload(matches[0]))


def _claim(found: dict[str, str], split: str, manifest: ArtifactManifest, *, kind: str) -> bool:
    """Record one (split -> artifact) claim, refusing a conflicting second one."""
    existing = found.get(split)
    if existing is not None and existing != manifest.artifact_id:
        raise ArtifactError(f"two different {kind} artifacts were given for split {split!r}")
    found[split] = manifest.artifact_id
    return existing is None


def _views_derive(arguments: argparse.Namespace) -> int:
    operation = _operation(arguments)
    accepted = set(stage("views").input_artifact_types) | {"dpo.clip-registry/v1"}
    _require_types(operation, accepted, minimum=5)
    track = arguments.track
    registry = _find_manifest(operation, "dpo.clip-registry/v1")
    attributes = registry.semantic["attributes"]
    if not isinstance(attributes, Mapping) or "split_manifest" not in attributes:
        raise ArtifactError(f"clip registry {registry.artifact_id} carries no split manifest")
    split_manifest = parse_split_manifest(attributes["split_manifest"])
    clips = _corpus_clips(operation.store, registry)
    pools = {}
    pool_artifacts: dict[str, str] = {}
    annotations: dict[str, str] = {}
    for manifest in operation.manifests:
        if manifest.artifact_type == "dpo.frozen-candidate-pool/v1":
            attributes = manifest.semantic["attributes"]
            if not isinstance(attributes, Mapping) or str(attributes.get("track")) != track:
                continue
            split = _split_of(manifest, attributes)
            if _claim(pool_artifacts, split, manifest, kind="frozen pool"):
                pools[split] = parse_frozen_pool(operation.store.read_payload(manifest.artifact_id))
        elif manifest.artifact_type == "dpo.raw-annotations/v1":
            parameters = request_parameters(manifest)
            if str(parameters.get("track")) != track:
                continue
            _claim(annotations, _split_of(manifest, parameters), manifest, kind="raw annotations")
    missing = [
        f"{kind} for {split}"
        for kind, found in (("frozen pool", pool_artifacts), ("raw annotations", annotations))
        for split in ("train", "validation")
        if split not in found
    ]
    if missing:
        raise ArtifactError(f"track {track!r} is missing its {missing[0]} artifact")
    aggregates = {}
    retained = {}
    for split, artifact_id in annotations.items():
        rows = parse_annotations(operation.store.read_payload(artifact_id))
        report = _reliability_report_for(operation.store, artifact_id)
        kept = retained_annotations(rows, report)
        retained[split] = kept
        aggregates[split] = aggregate_all(
            kept, minimum_judgments=int(str(operation.contract.annotation["judgments_per_pair"]))
        )
    caption_contract = operation.contract.tracks[track]
    audits = dict(
        resolve_audits(
            [audit_candidate(record, contract=caption_contract) for record in pools["train"].candidates],
            [],
        )
    )
    views = publish_track_views(
        operation.publisher(),
        operation.contract,
        track=track,
        manifest=split_manifest,
        clips=clips,
        train_pool=pools["train"],
        validation_pool=pools["validation"],
        train_pool_artifact_id=pool_artifacts["train"],
        validation_pool_artifact_id=pool_artifacts["validation"],
        train_aggregates=aggregates["train"],
        validation_aggregates=aggregates["validation"],
        train_annotations=retained["train"],
        train_audits=audits,
    )
    _emit(
        {
            "status": "published",
            "operation": "views-derive",
            "track": track,
            "artifacts": views.artifact_ids,
            "rows": {
                "sft": len(views.sft_rows),
                "pair_strict": len(views.strict_pairs),
                "pair_all": len(views.metadata_pairs),
                "validation_pairs": len(views.validation_pairs),
            },
        }
    )
    return 0


@dataclass(frozen=True)
class _BackendChoice:
    """The resolved live-training backend and the inputs it needs."""

    implementation: str
    configs: dict[str, Any]
    media_dir: Path | None

    @property
    def name(self) -> str:
        return "tiny" if self.implementation == TINY_IMPLEMENTATION else "gemma"


def _resolve_backend(
    contract: StudyContract, arguments: argparse.Namespace, *, command: str
) -> _BackendChoice | None:
    """Gate the backend before any store mutation; None means blocked (exit 3)."""
    implementation = str(contract.raw["models"]["seed"]["implementation"])
    backend_configs = [str(path) for path in (arguments.backend_config or [])]
    media_dir = arguments.media_dir
    if implementation == TINY_IMPLEMENTATION:
        if backend_configs or media_dir:
            raise ArtifactError("the tiny seed backend takes no --backend-config/--media-dir")
        return _BackendChoice(implementation=implementation, configs={}, media_dir=None)
    if implementation != GEMMA_IMPLEMENTATION:
        _emit(
            {
                "status": "blocked_pending_external_operation",
                "command": command,
                "gate": f"seed-model implementation {implementation!r} has no wired training backend",
                "side_effects": False,
            }
        )
        return None
    if not backend_configs or not media_dir:
        raise ArtifactError("the Gemma seed backend requires --backend-config and --media-dir")
    from dpo.models.gemma4.backend_config import load_config
    from dpo.models.gemma4.training_backend import resolve_lora_targets

    # A contract that cannot express a Gemma LoRA scope fails here, on any
    # machine, rather than after an operator has found a GPU.
    resolve_lora_targets(contract)
    import torch

    if not torch.cuda.is_available():
        _emit(
            {
                "status": "blocked_pending_external_operation",
                "command": command,
                "gate": "Gemma training and scoring require a CUDA device",
                "side_effects": False,
            }
        )
        return None
    configs: dict[str, Any] = {}
    for path in backend_configs:
        config = load_config(path)
        track = config.model.media_inputs
        verify_backend_pin(contract, track=track, backend_config_path=Path(path))
        if track in configs:
            raise ArtifactError(f"two backend configs serve track {track!r}")
        configs[track] = config
    absent = [track for track in TRACKS if track not in configs]
    if absent:
        raise ArtifactError(f"no --backend-config serves track {absent[0]!r}; the matrix trains every track")
    return _BackendChoice(implementation=implementation, configs=configs, media_dir=Path(str(media_dir)))


def _build_backend(contract: StudyContract, choice: _BackendChoice) -> TrainingBackend:
    if choice.implementation == TINY_IMPLEMENTATION:
        return TinyBackend(contract=contract)
    from dpo.models.gemma4.training_backend import GemmaBackend

    assert choice.media_dir is not None
    return GemmaBackend(contract=contract, configs=choice.configs, media_dir=choice.media_dir)


def _require_media_coverage(choice: _BackendChoice, clips_by_track: Mapping[str, set[str]]) -> None:
    if choice.media_dir is None:
        return
    for track, clip_ids in sorted(clips_by_track.items()):
        resolve_media_files(choice.media_dir, sorted(clip_ids), track=track)


def _selection_identity(contract: StudyContract, choice: _BackendChoice) -> dict[str, str]:
    """The processor/preprocessing identities the lock manifest freezes.

    Tiny: the byte tokenizer and the synthetic media generator, exactly as the
    canary records them. Gemma: the pinned checkpoint plus the hash of the
    backend config that produced the processor, and the media-file convention
    (one directory, one file per clip by suffix) that preprocessing consists of.
    """
    if choice.implementation == TINY_IMPLEMENTATION:
        return {
            "processor_hash": semantic_hash({"processor": "tiny-byte/v1"}),
            "preprocessing_hash": semantic_hash({"media": "synthetic/v1", "media_dim": DEFAULT_MEDIA_DIM}),
        }
    return {
        "processor_hash": semantic_hash(
            {
                "processor": "gemma4-chat-template/v1",
                "tracks": {
                    track: {
                        "model_id": config.model.model_id,
                        "revision": config.model.revision,
                        "backend_config_hash": sha256_file(config.source_path),
                    }
                    for track, config in sorted(choice.configs.items())
                },
            }
        ),
        "preprocessing_hash": semantic_hash(
            {
                "media": "media-directory/v1",
                "audio_suffixes": list(AUDIO_MEDIA_SUFFIXES),
                "video_suffixes": list(VIDEO_MEDIA_SUFFIXES),
            }
        ),
    }


def _train_run(arguments: argparse.Namespace) -> int:
    contract = load_contract(arguments.contract)
    choice = _resolve_backend(contract, arguments, command="train run")
    if choice is None:
        return 3
    operation = _operation(arguments)
    _require_types(operation, set(stage("train").input_artifact_types), minimum=2)
    views = collect_view_inputs(
        operation.store, operation.manifests, required=("sft", "pair_strict", "pair_all")
    )
    _require_media_coverage(
        choice,
        {
            track: {row.clip_id for row in views.sft_rows.get(track, ())}
            | {row.clip_id for row in views.strict_pairs.get(track, ())}
            | {row.clip_id for row in views.metadata_pairs.get(track, ())}
            for track in TRACKS
        },
    )
    canonical_seed = int(str(contract.training["canonical_seed"]))
    runner = LiveMatrixRunner(
        contract=contract,
        backend=_build_backend(contract, choice),
        checkpoint_dir=Path(arguments.checkpoint_dir),
        strict_pairs=views.strict_pairs,
        metadata_pairs=views.metadata_pairs,
        sft_rows=views.sft_rows,
    )
    variants_by_experiment = {
        experiment_id: expand_experiment(contract, experiment_id) for experiment_id in EXPERIMENT_IDS
    }
    cells, cell_artifacts = publish_training_matrix(
        operation.publisher(),
        contract,
        runner=runner,
        variants_by_experiment=variants_by_experiment,
        canonical_seed=canonical_seed,
        view_artifacts=views.artifact_ids,
        strict_pairs=views.strict_pairs,
    )
    _emit(
        {
            "status": "published",
            "operation": "train-run",
            "backend": choice.name,
            "checkpoint_dir": str(Path(arguments.checkpoint_dir).resolve()),
            "cells": len(cells),
            "cells_trained": runner.trained_count,
            "cells_resumed": runner.resumed_count,
            "matrix": [
                {
                    "experiment_id": experiment_id,
                    "variant_id": variant_id,
                    "track": track,
                    "artifact_id": cell_artifacts[(experiment_id, variant_id, track)],
                    "steps": cells[(experiment_id, variant_id, track)].steps,
                    "resumed": runner.was_resumed(experiment_id, variant_id, track, canonical_seed),
                }
                for (experiment_id, variant_id, track) in sorted(cells)
            ],
        }
    )
    return 0


def _select_run(arguments: argparse.Namespace) -> int:
    contract = load_contract(arguments.contract)
    choice = _resolve_backend(contract, arguments, command="select run")
    if choice is None:
        return 3
    operation = _operation(arguments, with_training_cells=True)
    _require_types(
        operation,
        set(stage("validate").input_artifact_types)
        | set(stage("train").input_artifact_types)
        | {"dpo.validation-pairs/v1"},
        minimum=2,
    )
    views = collect_view_inputs(
        operation.store, operation.manifests, required=("pair_strict", "validation_pairs")
    )
    cells, cell_artifacts = collect_matrix_cells(operation.store, operation.manifests)
    _require_media_coverage(
        choice,
        {track: {row.clip_id for row in views.validation_pairs.get(track, ())} for track in TRACKS},
    )
    canonical_seed = int(str(contract.training["canonical_seed"]))
    backend = _build_backend(contract, choice)
    policies, _ = load_checkpoint_policies(backend, arguments.checkpoint_dir)
    variants_by_experiment = {
        experiment_id: expand_experiment(contract, experiment_id) for experiment_id in EXPERIMENT_IDS
    }
    # Selection compares the whole matrix: every variant of every experiment on
    # every track must be both published and materializable, or the ranking
    # would silently be over a subset.
    for experiment_id, variants in sorted(variants_by_experiment.items()):
        for variant in variants:
            for track in TRACKS:
                key = (experiment_id, variant.variant_id, track)
                if key not in cells:
                    raise ArtifactError(f"no matrix-cell artifact was given for cell {key}")
                if (*key, canonical_seed) not in policies:
                    raise ArtifactError(
                        f"checkpoint directory has no policy for cell {key};"
                        " run `dpo train run` with the same --checkpoint-dir first"
                    )
    identity = _selection_identity(contract, choice)
    selection = publish_selection(
        operation.publisher(),
        contract,
        variants_by_experiment=variants_by_experiment,
        canonical_seed=canonical_seed,
        validation_pairs=views.validation_pairs,
        strict_pairs=views.strict_pairs,
        policies=policies,
        seed_adapters={track: backend.seed_adapter(track) for track in TRACKS},
        media_provider=backend.media_batch,
        view_artifacts=views.artifact_ids,
        cells=cells,
        cell_artifacts=cell_artifacts,
        processor_hash=identity["processor_hash"],
        preprocessing_hash=identity["preprocessing_hash"],
        evaluation_version="evaluation/v1",
        metric_versions={"compliance": "v1", "preference": "v1"},
        selection_note=(
            f"{choice.name} backend selection: validation preference accuracy, lexical tie-break"
        ),
    )
    _emit(
        {
            "status": "published",
            "operation": "select-run",
            "backend": choice.name,
            "ranking": selection.ranking,
            "selected_variants": selection.selected_variants,
            "validation_accuracy": selection.validation_reports,
            "artifacts": {
                "validation_report": selection.validation_artifact_id,
                "selection_report": selection.selection_artifact_id,
                "lock_manifest": selection.lock_artifact_id,
            },
            "lock_id": selection.lock_manifest.lock_id,
        }
    )
    return 0


def _report_show(arguments: argparse.Namespace) -> int:
    store = ArtifactStore.open(arguments.workspace)

    def _payloads(artifact_type: str) -> list[tuple[str, dict[str, Any]]]:
        rows: list[tuple[str, dict[str, Any]]] = []
        for artifact_id in store.find_by_type(artifact_type):
            payload = json.loads(store.read_payload(artifact_id))
            rows.append((artifact_id, payload))
        return rows

    _emit(
        {
            "status": "ok",
            "reports": {
                "validation": [
                    {"artifact_id": artifact_id, "accuracy": payload["accuracy"]}
                    for artifact_id, payload in _payloads("dpo.validation-report/v1")
                ],
                "selection": [
                    {
                        "artifact_id": artifact_id,
                        "ranking": payload["ranking"],
                        "selected_variants": payload["selected_variants"],
                        "selected_hyperparameters": payload["selected_hyperparameters"],
                    }
                    for artifact_id, payload in _payloads("dpo.selection-report/v1")
                ],
                "locks": [
                    {"artifact_id": artifact_id, "lock_id": parse_lock_manifest(payload).lock_id}
                    for artifact_id, payload in _payloads("dpo.lock-manifest/v1")
                ],
            },
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

    candidates = commands.add_parser(
        "candidates", help="generate, audit, pair, and freeze the offline C0 candidate pool"
    )
    candidates_actions = candidates.add_subparsers(dest="action", required=True)
    generate = candidates_actions.add_parser("generate")
    generate.add_argument("--workspace", required=True)
    generate.add_argument("--contract", required=True)
    generate.add_argument("--artifact-id", action="append", required=True)
    generate.add_argument("--track", required=True, choices=["visual", "audio"])
    generate.add_argument("--split", required=True, choices=["train", "validation"])
    generate.add_argument("--dataset-version", required=True)
    generate.add_argument("--backend-config")
    generate.add_argument("--media-dir")
    generate.set_defaults(handler=_candidates_generate)

    annotation = commands.add_parser(
        "annotation", help="export collection tasks, serve the UI, ingest responses"
    )
    annotation_actions = annotation.add_subparsers(dest="action", required=True)
    export_tasks = annotation_actions.add_parser("export-tasks")
    export_tasks.add_argument("--workspace", required=True)
    export_tasks.add_argument("--contract", required=True)
    export_tasks.add_argument("--artifact-id", action="append", required=True)
    export_tasks.add_argument("--out-tasks", required=True)
    export_tasks.add_argument("--out-answers", required=True)
    export_tasks.set_defaults(handler=_annotation_export_tasks)
    annotation_ingest = annotation_actions.add_parser("ingest")
    annotation_ingest.add_argument("--workspace", required=True)
    annotation_ingest.add_argument("--contract", required=True)
    annotation_ingest.add_argument("--artifact-id", action="append", required=True)
    annotation_ingest.add_argument("--split", required=True, choices=["train", "validation"])
    annotation_ingest.add_argument("--tasks", required=True)
    annotation_ingest.add_argument("--answers", required=True)
    annotation_ingest.add_argument("--responses", action="append", required=True)
    annotation_ingest.set_defaults(handler=_annotation_ingest)
    serve = annotation_actions.add_parser("serve")
    serve.add_argument("--tasks", required=True)
    serve.add_argument("--media-dir", required=True)
    serve.add_argument("--out", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.set_defaults(handler=_annotation_serve)

    views = commands.add_parser("views", help="derive every training view of one track")
    views_actions = views.add_subparsers(dest="action", required=True)
    derive = views_actions.add_parser("derive")
    derive.add_argument("--workspace", required=True)
    derive.add_argument("--contract", required=True)
    derive.add_argument("--artifact-id", action="append", required=True)
    derive.add_argument("--track", required=True, choices=["visual", "audio"])
    derive.set_defaults(handler=_views_derive)

    train = commands.add_parser("train", help="run and publish every experiment-matrix cell")
    train_actions = train.add_subparsers(dest="action", required=True)
    train_run = train_actions.add_parser("run")
    train_run.add_argument("--workspace", required=True)
    train_run.add_argument("--contract", required=True)
    train_run.add_argument("--artifact-id", action="append", required=True)
    train_run.add_argument("--checkpoint-dir", required=True)
    train_run.add_argument("--backend-config", action="append")
    train_run.add_argument("--media-dir")
    train_run.set_defaults(handler=_train_run)

    select = commands.add_parser("select", help="score every variant, select winners, and lock")
    select_actions = select.add_subparsers(dest="action", required=True)
    select_run = select_actions.add_parser("run")
    select_run.add_argument("--workspace", required=True)
    select_run.add_argument("--contract", required=True)
    select_run.add_argument("--artifact-id", action="append", required=True)
    select_run.add_argument("--checkpoint-dir", required=True)
    select_run.add_argument("--backend-config", action="append")
    select_run.add_argument("--media-dir")
    select_run.set_defaults(handler=_select_run)

    report = commands.add_parser("report", help="show published validation/selection/lock reports")
    report_actions = report.add_subparsers(dest="action", required=True)
    report_show = report_actions.add_parser("show")
    report_show.add_argument("--workspace", required=True)
    report_show.set_defaults(handler=_report_show)

    for command_name in DEFERRED_GATES:
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
