"""`dpo candidates`: generate and freeze the C0 candidate pools."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from dpo.candidates.audit import audit_candidate, resolve_audits
from dpo.candidates.dedup import cross_pool_dedup
from dpo.candidates.freeze import parse_frozen_pool
from dpo.candidates.generation import (
    GEMMA_IMPLEMENTATION,
    TINY_IMPLEMENTATION,
    generate_c0_candidates,
    generate_c0_candidates_gemma,
)
from dpo.cli._shared import _emit, _find_manifest, _operation, _registry_shard_rows, _require_types
from dpo.contracts.study_contract import (
    load_contract,
)
from dpo.core.artifacts import (
    ArtifactError,
)
from dpo.data.leakage_audit import run_leakage_audit
from dpo.data.split import parse_split_manifest
from dpo.pipeline.candidate_stage import publish_frozen_pool


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


def _candidates_dedup(arguments: argparse.Namespace) -> int:
    """Pre-enforce the views-derive leakage gate across two frozen pools.

    Drops every candidate in a cross-pool near-duplicate collision (both
    sides), re-pairs and re-freezes both pools with dedup-source lineage, and
    proves in-band that the leakage audit now passes — the whole point of
    running this before anyone annotates.
    """
    operation = _operation(arguments)
    _require_types(operation, {"dpo.frozen-candidate-pool/v1", "dpo.clip-registry/v1"}, minimum=3)
    registry_manifest = _find_manifest(operation, "dpo.clip-registry/v1")
    pool_manifests = [
        manifest
        for manifest in operation.manifests
        if manifest.artifact_type == "dpo.frozen-candidate-pool/v1"
    ]
    if len(pool_manifests) != 2:
        raise ArtifactError("candidates dedup takes exactly two frozen pools (train and validation)")
    attributes = registry_manifest.semantic["attributes"]
    split_manifest = parse_split_manifest(attributes["split_manifest"])

    pools = {}
    for manifest in pool_manifests:
        pool = parse_frozen_pool(operation.store.read_payload(manifest.artifact_id))
        roles = {split_manifest.role_of(candidate.clip_id) for candidate in pool.candidates}
        if len(roles) != 1:
            raise ArtifactError(f"pool {manifest.artifact_id} spans splits {sorted(roles)}")
        pools[roles.pop()] = (manifest, pool)
    if set(pools) != {"train", "validation"}:
        raise ArtifactError(f"dedup expects one train and one validation pool; got {sorted(pools)}")

    per_clip_min = int(str(operation.contract.pairs["per_clip_min"]))
    outcome = cross_pool_dedup(pools["train"][1], pools["validation"][1], per_clip_min=per_clip_min)

    published = {}
    kept = {"train": outcome.kept_a, "validation": outcome.kept_b}
    for split in ("train", "validation"):
        source_manifest, source_pool = pools[split]
        caption_contract = operation.contract.tracks[source_pool.track]
        audits = dict(
            resolve_audits([audit_candidate(record, contract=caption_contract) for record in kept[split]], [])
        )
        wanted_clips = frozenset(candidate.clip_id for candidate in kept[split])

        def _keep(entry: Mapping[str, object], wanted: frozenset[str] = wanted_clips) -> bool:
            return str(entry["clip_id"]) in wanted

        shard_rows = _registry_shard_rows(operation.store, registry_manifest, _keep)
        pool, artifact_id = publish_frozen_pool(
            operation.publisher(),
            operation.contract,
            track=source_pool.track,
            split=split,
            candidates=list(kept[split]),
            audits=audits,
            shard_artifact_ids={clip: shard for clip, (shard, _) in shard_rows.items()},
            dataset_version=arguments.dataset_version,
            audit_version=source_pool.evidence_audit_version,
            source_pool_ids=(source_manifest.artifact_id,),
            operation="candidates-dedup",
        )
        published[split] = (pool, artifact_id)

    # In-band proof: the exact combined audit views derive will run must pass now.
    from dpo.candidates.freeze import freeze_pool
    from dpo.cli.views import _corpus_clips

    combined = freeze_pool(
        dataset_version=f"{arguments.dataset_version}-combined-check",
        track=published["train"][0].track,
        evidence_audit_version=published["train"][0].evidence_audit_version,
        candidates=(*published["train"][0].candidates, *published["validation"][0].candidates),
        pairs=(*published["train"][0].pairs, *published["validation"][0].pairs),
    )
    report = run_leakage_audit(
        manifest=split_manifest,
        clips=_corpus_clips(operation.store, registry_manifest),
        pool=combined,
        enforce=True,
    )
    _emit(
        {
            "status": "published",
            "operation": "candidates-dedup",
            "dropped": outcome.document()["dropped"],
            "leakage_audit_passed": report.passed,
            "pools": {
                split: {
                    "artifact_id": artifact_id,
                    "candidates": len(pool.candidates),
                    "pairs": len(pool.pairs),
                    "challenge_fraction": round(
                        sum(1 for c in pool.candidates if c.source_kind == "controlled_error")
                        / len(pool.candidates),
                        4,
                    ),
                }
                for split, (pool, artifact_id) in sorted(published.items())
            },
        }
    )
    return 0
