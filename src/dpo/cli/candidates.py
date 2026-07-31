"""`dpo candidates`: generate and freeze the C0 candidate pools."""

from __future__ import annotations

import argparse
from pathlib import Path

from dpo.candidates.audit import audit_candidate, resolve_audits
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
