"""Shared annotation-stage implementation: ingest, reliability, aggregation.

One (track, split) batch of raw preference annotations is validated against
its frozen pool, published, screened by the preregistered reliability rules
(also published), and aggregated. Where the annotations come from is the
caller's concern — the offline canary synthesizes them, a live runner ingests
the real collection export.
"""

from __future__ import annotations

from collections.abc import Sequence

from dpo.annotation.aggregate import PairAggregate, aggregate_all
from dpo.annotation.raw_annotations import RawAnnotation, validate_against_pool
from dpo.annotation.reliability import build_reliability_report, retained_annotations
from dpo.candidates.freeze import FrozenCandidatePool
from dpo.contracts.study_contract import StudyContract
from dpo.core.artifacts import ParentEdge
from dpo.pipeline.publishing import ArtifactPublisher


def ingest_annotations(
    publisher: ArtifactPublisher,
    contract: StudyContract,
    *,
    track: str,
    split: str,
    pool: FrozenCandidatePool,
    pool_artifact_id: str,
    annotations: Sequence[RawAnnotation],
    attention_expected: dict[str, str],
) -> tuple[tuple[RawAnnotation, ...], tuple[PairAggregate, ...], dict[str, str]]:
    """Validate, publish, screen, and aggregate one (track, split) annotation batch.

    Returns the retained annotations, the per-pair aggregates, and the ids of
    the raw-annotations and reliability-report artifacts.
    """
    validate_against_pool(annotations, pool)
    reliability = build_reliability_report(
        annotations,
        attention_expected=attention_expected,
        min_response_ms=int(str(contract.annotation["min_response_ms"])),
        max_position_bias=float(str(contract.annotation["max_position_bias"])),
        min_attention_pass=float(str(contract.annotation["min_attention_pass"])),
    )
    split_clips = {candidate.clip_id for candidate in pool.candidates}
    annotations_artifact = publisher.publish(
        "dpo.raw-annotations/v1",
        {
            "schema": "dpo.raw-annotations/v1",
            "rows": [annotation.document() for annotation in annotations],
        },
        parents=(ParentEdge(pool_artifact_id, "frozen-pool"),),
        stage="annotation",
        parameters={"operation": "annotation-ingest", "track": track, "split": split},
        row_count=len(annotations),
        clips=split_clips,
        role_exposure={split},
    )
    reliability_artifact = publisher.publish(
        "dpo.reliability-report/v1",
        reliability.document(),
        parents=(ParentEdge(annotations_artifact, "raw-annotations"),),
        stage="annotation",
        parameters={"operation": "reliability", "track": track, "split": split},
        clips=split_clips,
        role_exposure={split},
    )
    retained = retained_annotations(annotations, reliability)
    aggregates = aggregate_all(
        retained, minimum_judgments=int(str(contract.annotation["judgments_per_pair"]))
    )
    artifact_ids = {"raw_annotations": annotations_artifact, "reliability_report": reliability_artifact}
    return retained, aggregates, artifact_ids
