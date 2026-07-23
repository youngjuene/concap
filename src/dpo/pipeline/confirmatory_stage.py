"""Shared test-stage implementation: the normative test-once flow.

Reserve before read, fence the reservation, take exactly one official
protected read of the test shards, generate and publish the test metrics
under the frozen decoding budget, and finalize — in that order, with every
step recorded as an artifact. An optional ``probe`` hook runs between the
reservation and the first read so a rehearsal harness can assert the
resume/reject semantics at exactly the point they matter.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from dpo.contracts.study_contract import EXPERIMENT_IDS, TRACKS, StudyContract
from dpo.core.access import ProtectedAccessAuthority
from dpo.core.artifacts import ParentEdge
from dpo.evaluation.compliance import GeneratedCaption, compute_compliance
from dpo.models.base import MediaBatch, ModelAdapter
from dpo.pipeline.lock import (
    LockManifest,
    TestOnceResolver,
    TestReservation,
    test_split_semantic_hash,
)
from dpo.pipeline.publishing import ArtifactPublisher


@dataclass(frozen=True)
class ConfirmatoryTestResult:
    reservation: TestReservation
    reservation_artifact_id: str
    test_metrics_artifact_id: str
    finalization_artifact_id: str


def run_confirmatory_test(
    publisher: ArtifactPublisher,
    contract: StudyContract,
    *,
    database_path: Path,
    lock_manifest: LockManifest,
    lock_artifact_id: str,
    test_clips: Sequence[str],
    shard_ids: Mapping[str, str],
    policies: Mapping[tuple[str, str, str, int], ModelAdapter],
    selected_variants: Mapping[str, Mapping[str, str]],
    canonical_seed: int,
    media_provider: Callable[[str, Sequence[str]], MediaBatch],
    probe: Callable[[TestOnceResolver, TestReservation], None] | None = None,
) -> ConfirmatoryTestResult:
    """Reserve, fence, read once, measure, and finalize the confirmatory test."""
    resolver = TestOnceResolver(str(database_path))
    test_split_hash = test_split_semantic_hash(test_clips)
    reservation = resolver.reserve(lock_manifest, test_split_hash=test_split_hash)
    if probe is not None:
        probe(resolver, reservation)
    resolver.mark_first_read(reservation)
    authority = ProtectedAccessAuthority(database_path)
    test_capability = authority.reserve(
        scope="confirmatory-test", semantic_hash=reservation.semantic_hash, roles={"test"}
    )
    for clip_id in test_clips:
        publisher.store.read_payload(
            shard_ids[clip_id],
            capability=test_capability,
            authority=authority,
            semantic_hash=reservation.semantic_hash,
        )
    reservation_artifact = publisher.publish(
        "dpo.test-reservation/v1",
        {"schema": "dpo.test-reservation/v1", **resolver.authority_record(reservation)},
        parents=(ParentEdge(lock_artifact_id, "lock-manifest"),)
        + tuple(ParentEdge(shard_ids[clip_id], "test-shard") for clip_id in test_clips),
        stage="test",
        parameters={"operation": "test-reserve"},
        clips=set(test_clips),
        role_exposure={"test"},
        test_exposure=set(test_clips),
    )
    test_metrics: dict[str, dict[str, dict[str, object]]] = {}
    validation_decoding = contract.validation
    for track in TRACKS:
        per_model: dict[str, dict[str, object]] = {}
        media = media_provider(track, test_clips)
        for experiment_id in EXPERIMENT_IDS:
            policy_adapter = policies[
                (experiment_id, selected_variants[track][experiment_id], track, canonical_seed)
            ]
            texts = policy_adapter.generate(
                media,
                temperature=float(str(validation_decoding["temperature"])),
                top_p=float(str(validation_decoding["top_p"])),
                max_new_tokens=int(str(validation_decoding["max_new_tokens"])),
                seed=canonical_seed,
            )
            captions = tuple(
                GeneratedCaption(clip_id=clip_id, text=text or "(empty)")
                for clip_id, text in zip(test_clips, texts, strict=True)
            )
            per_model[experiment_id] = compute_compliance(captions, contract.tracks[track]).document()
        test_metrics[track] = per_model
    test_metrics_artifact = publisher.publish(
        "dpo.test-metrics/v1",
        {"schema": "dpo.test-metrics/v1", "compliance": test_metrics},
        parents=(ParentEdge(reservation_artifact, "test-reservation"),),
        stage="test",
        parameters={"operation": "test-metrics"},
        clips=set(test_clips),
        role_exposure={"test"},
        test_exposure=set(test_clips),
    )
    resolver.complete(reservation)
    authority.complete(test_capability)
    finalization_artifact = publisher.publish(
        "dpo.test-finalization/v1",
        {"schema": "dpo.test-finalization/v1", **resolver.authority_record(reservation)},
        parents=(
            ParentEdge(reservation_artifact, "test-reservation"),
            ParentEdge(test_metrics_artifact, "test-metrics"),
        ),
        stage="test",
        parameters={"operation": "test-finalize"},
        clips=set(test_clips),
        role_exposure={"test"},
    )
    return ConfirmatoryTestResult(
        reservation=reservation,
        reservation_artifact_id=reservation_artifact,
        test_metrics_artifact_id=test_metrics_artifact,
        finalization_artifact_id=finalization_artifact,
    )
