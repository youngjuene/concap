"""`dpo views`: derive every training view of one track."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping

from dpo.annotation.aggregate import aggregate_all
from dpo.annotation.reliability import (
    ReliabilityReport,
    parse_reliability_report,
    retained_annotations,
)
from dpo.candidates.audit import audit_candidate, resolve_audits
from dpo.candidates.freeze import parse_frozen_pool
from dpo.cli._shared import _emit, _find_manifest, _operation, _require_types
from dpo.core.artifacts import (
    ArtifactError,
    ArtifactManifest,
    ArtifactStore,
)
from dpo.data.split import ClipInput, parse_split_manifest
from dpo.pipeline.stage_inputs import (
    parse_annotations,
    request_parameters,
)
from dpo.pipeline.stages import stage
from dpo.pipeline.view_stage import publish_track_views


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
