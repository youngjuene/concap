"""Shared corpus-stage implementations: ingest and split locking.

The CLI handlers and the offline canary both publish corpus artifacts through
these functions, so the row schema, the registry-id formula, and the shard
parameters have exactly one home and cannot drift between the rehearsal
pipeline and the production commands.
"""

from __future__ import annotations

from collections.abc import Sequence

from dpo.contracts.study_contract import StudyContract
from dpo.core.artifacts import ParentEdge
from dpo.core.identity import semantic_hash
from dpo.data.split import ClipInput, SplitManifest, assign_splits, registry_rows
from dpo.pipeline.publishing import ArtifactPublisher


def publish_corpus_ingest(publisher: ArtifactPublisher, clips: Sequence[ClipInput]) -> str:
    return publisher.publish(
        "dpo.corpus-ingest/v1",
        {
            "schema": "dpo.corpus-ingest/v1",
            "rows": [clip.document() for clip in clips],
        },
        stage="corpus-ingest",
        parameters={"operation": "corpus-ingest"},
        row_count=len(clips),
        clips={clip.clip_id for clip in clips},
    )


def publish_lock_splits(
    publisher: ArtifactPublisher,
    contract: StudyContract,
    clips: Sequence[ClipInput],
    *,
    ingest_artifact_id: str,
) -> tuple[str, SplitManifest, dict[str, str]]:
    """Compute and freeze the immutable splits; returns (registry id, manifest, shard ids)."""
    split_seed = int(str(contract.corpus["split_seed"]))
    manifest = assign_splits(clips, seed=split_seed, fractions=contract.split_fractions)
    rows = registry_rows(manifest, list(clips))
    # The registry id binds to the same contract identity its artifact carries —
    # the corpus slice — so the store's registry verifier can recompute it and a
    # hyperparameter tweak elsewhere in the contract cannot re-key the registry.
    registry_id_value = semantic_hash(
        {
            "schema": "dpo.clip-registry/v1",
            "contract_id": publisher.contract_id_for("lock-splits", None),
            "clips": rows,
        }
    )
    registry_artifact = publisher.publish(
        "dpo.clip-registry/v1",
        {
            "schema": "dpo.clip-registry/v1",
            "registry_id": registry_id_value,
            "clips": rows,
        },
        stage="lock-splits",
        parents=(ParentEdge(ingest_artifact_id, "corpus-ingest"),),
        parameters={"operation": "lock-splits", "registry_id": registry_id_value},
        seed=split_seed,
        row_count=len(rows),
        clips={str(row["clip_id"]) for row in rows},
        attributes={
            "registry_id": registry_id_value,
            "split_manifest": manifest.signed_document(),
        },
    )
    shard_ids: dict[str, str] = {}
    for row in rows:
        shard_ids[str(row["clip_id"])] = publisher.publish(
            "dpo.clip-registry-shard/v1",
            row,
            stage="lock-splits",
            parameters={
                "operation": "lock-splits-shard",
                "registry_id": registry_id_value,
                "clip_id": row["clip_id"],
            },
            seed=split_seed,
            row_count=1,
            clips={str(row["clip_id"])},
        )
    return registry_artifact, manifest, shard_ids
