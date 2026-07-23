"""Shared views-stage implementation: every derived view of one track.

From the frozen train and validation pools and their aggregated preferences,
this stage derives and publishes the SFT view, the strict and metadata pair
views, the validation pairs, the natural-noise calibration, and the synthetic
flip manifests — and runs the enforcing leakage audit over the combined pool
before anything is published. The one frozen preference dataset regenerates
every view deterministically; view-gate changes create new view artifacts,
never a new raw collection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from dpo.annotation.aggregate import PairAggregate
from dpo.annotation.raw_annotations import RawAnnotation
from dpo.candidates.audit import ResolvedCandidateAudit
from dpo.candidates.freeze import FrozenCandidatePool, freeze_pool
from dpo.contracts.study_contract import StudyContract
from dpo.core.artifacts import ParentEdge
from dpo.data.derive_pairs import (
    MetadataPair,
    StrictPair,
    derive_pair_all,
    derive_pair_strict,
    with_weights,
)
from dpo.data.derive_sft import SftExample, derive_sft
from dpo.data.leakage_audit import run_leakage_audit
from dpo.data.noise import estimate_natural_noise, make_flip_manifest
from dpo.data.split import ClipInput, SplitManifest
from dpo.data.weighting import pair_weights
from dpo.pipeline.publishing import ArtifactPublisher


@dataclass(frozen=True)
class TrackViews:
    """One track's derived rows and the ids of its published view artifacts."""

    sft_rows: tuple[SftExample, ...]
    strict_pairs: tuple[StrictPair, ...]
    metadata_pairs: tuple[MetadataPair, ...]
    validation_pairs: tuple[StrictPair, ...]
    artifact_ids: dict[str, str]


def publish_track_views(
    publisher: ArtifactPublisher,
    contract: StudyContract,
    *,
    track: str,
    manifest: SplitManifest,
    clips: Sequence[ClipInput],
    train_pool: FrozenCandidatePool,
    validation_pool: FrozenCandidatePool,
    train_pool_artifact_id: str,
    validation_pool_artifact_id: str,
    train_aggregates: Sequence[PairAggregate],
    validation_aggregates: Sequence[PairAggregate],
    train_annotations: Sequence[RawAnnotation],
    train_audits: Mapping[str, ResolvedCandidateAudit],
) -> TrackViews:
    """Derive, leakage-audit, and publish every view of one track."""
    views_config = contract.views
    strict = derive_pair_strict(
        train_aggregates,
        train_pool,
        manifest,
        split="train",
        min_agreement=float(str(views_config["pair_strict"]["min_agreement"])),
        min_strength=float(str(views_config["pair_strict"]["min_strength"])),
    )
    weights = pair_weights(
        strict,
        strategy=str(views_config["weighting"]["strategy"]),
        cap_per_clip=int(str(views_config["weighting"]["cap_per_clip"])),
    )
    strict = with_weights(strict, weights)
    meta = derive_pair_all(train_aggregates, train_pool, manifest, split="train")
    sft_rows = derive_sft(
        train_aggregates,
        train_annotations,
        train_pool,
        manifest,
        train_audits,
        split="train",
        min_agreement=float(str(views_config["sft"]["min_agreement"])),
        min_confidence=float(str(views_config["sft"]["min_confidence"])),
    )
    validation_pairs = derive_pair_strict(
        validation_aggregates,
        validation_pool,
        manifest,
        split="validation",
        min_agreement=float(str(views_config["pair_strict"]["min_agreement"])),
        min_strength=float(str(views_config["pair_strict"]["min_strength"])),
    )
    combined_pool = freeze_pool(
        dataset_version="canary-combined/v1",
        track=track,
        evidence_audit_version="audit/v1",
        candidates=(*train_pool.candidates, *validation_pool.candidates),
        pairs=(*train_pool.pairs, *validation_pool.pairs),
    )
    run_leakage_audit(
        manifest=manifest,
        clips=clips,
        pool=combined_pool,
        sft_rows=sft_rows,
        enforce=True,
    )
    noise = estimate_natural_noise(train_annotations, train_aggregates)
    artifact_ids: dict[str, str] = {}
    artifact_ids["sft"] = publisher.publish(
        "dpo.sft-view/v1",
        {"schema": "dpo.sft-view/v1", "rows": [row.document() for row in sft_rows]},
        parents=(ParentEdge(train_pool_artifact_id, "frozen-pool"),),
        stage="views",
        parameters={"operation": "derive-sft", "track": track},
        row_count=len(sft_rows),
        clips={row.clip_id for row in sft_rows},
        role_exposure={"train"},
    )
    artifact_ids["pair_strict"] = publisher.publish(
        "dpo.pair-strict-view/v1",
        {"schema": "dpo.pair-strict-view/v1", "rows": [row.document() for row in strict]},
        parents=(ParentEdge(train_pool_artifact_id, "frozen-pool"),),
        stage="views",
        parameters={"operation": "derive-pairs-strict", "track": track},
        row_count=len(strict),
        clips={row.clip_id for row in strict},
        role_exposure={"train"},
    )
    artifact_ids["pair_all"] = publisher.publish(
        "dpo.pair-all-view/v1",
        {"schema": "dpo.pair-all-view/v1", "rows": [row.document() for row in meta]},
        parents=(ParentEdge(train_pool_artifact_id, "frozen-pool"),),
        stage="views",
        parameters={"operation": "derive-pairs-all", "track": track},
        row_count=len(meta),
        clips={row.clip_id for row in meta},
        role_exposure={"train"},
    )
    artifact_ids["validation_pairs"] = publisher.publish(
        "dpo.validation-pairs/v1",
        {
            "schema": "dpo.validation-pairs/v1",
            "rows": [row.document() for row in validation_pairs],
        },
        parents=(ParentEdge(validation_pool_artifact_id, "frozen-pool"),),
        stage="views",
        parameters={"operation": "derive-validation-pairs", "track": track},
        row_count=len(validation_pairs),
        clips={row.clip_id for row in validation_pairs},
        role_exposure={"validation"},
    )
    publisher.publish(
        "dpo.noise-calibration/v1",
        noise.document(),
        parents=(ParentEdge(artifact_ids["pair_strict"], "pair-view"),),
        stage="views",
        parameters={"operation": "noise-calibration", "track": track},
        clips={row.clip_id for row in strict},
        role_exposure={"train"},
    )
    for rate in [float(str(rate)) for rate in contract.robustness["flip_rates"]]:
        flip_manifest = make_flip_manifest(
            strict, flip_rate=rate, seed=int(str(contract.robustness["flip_seed"]))
        )
        publisher.publish(
            "dpo.flip-manifest/v1",
            flip_manifest.document(),
            parents=(ParentEdge(artifact_ids["pair_strict"], "pair-view"),),
            stage="views",
            parameters={"operation": "flip-manifest", "track": track, "rate": rate},
            clips={row.clip_id for row in strict},
            role_exposure={"train"},
        )
    return TrackViews(
        sft_rows=sft_rows,
        strict_pairs=strict,
        metadata_pairs=meta,
        validation_pairs=validation_pairs,
        artifact_ids=artifact_ids,
    )
