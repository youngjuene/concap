"""Shared candidates-stage implementation: pair sampling and pool freezing.

One (track, split) collection of audited candidate records becomes one frozen
candidate pool here: deterministic pair sampling under the contract's bounds,
the freeze, and the artifact publish parented on every clip's claim ledger.
Candidate generation and audit resolution happen upstream — the offline
canary synthesizes them, a live runner records real generations and human
audit decisions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from dpo.candidates.audit import ResolvedCandidateAudit
from dpo.candidates.candidate_records import CandidateRecord
from dpo.candidates.freeze import FrozenCandidatePool, freeze_pool
from dpo.candidates.pair_sampler import sample_pairs
from dpo.contracts.study_contract import StudyContract
from dpo.core.artifacts import ParentEdge
from dpo.pipeline.publishing import ArtifactPublisher


def publish_frozen_pool(
    publisher: ArtifactPublisher,
    contract: StudyContract,
    *,
    track: str,
    split: str,
    candidates: Sequence[CandidateRecord],
    audits: Mapping[str, ResolvedCandidateAudit],
    ledger_artifact_ids: Mapping[str, str],
    dataset_version: str,
    evidence_audit_version: str,
) -> tuple[FrozenCandidatePool, str]:
    """Sample pairs, freeze the pool for one (track, split), and publish it."""
    pairs = sample_pairs(
        candidates,
        audits,
        per_clip_min=int(str(contract.pairs["per_clip_min"])),
        per_clip_max=int(str(contract.pairs["per_clip_max"])),
        seed=int(str(contract.pairs["seed"])),
        near_duplicate_jaccard=float(str(contract.pairs["near_duplicate_jaccard"])),
    )
    pool = freeze_pool(
        dataset_version=dataset_version,
        track=track,
        evidence_audit_version=evidence_audit_version,
        candidates=candidates,
        pairs=pairs,
    )
    split_clips = {candidate.clip_id for candidate in pool.candidates}
    artifact_id = publisher.publish(
        "dpo.frozen-candidate-pool/v1",
        pool.document(),
        parents=tuple(
            ParentEdge(ledger_artifact_ids[clip_id], "claim-ledger") for clip_id in sorted(split_clips)
        ),
        stage="candidates",
        parameters={"operation": "candidates-freeze", "track": track, "split": split},
        row_count=len(pool.pairs),
        clips=split_clips,
        role_exposure={split},
        attributes={"pool_hash": pool.pool_hash, "track": track, "split": split},
    )
    return pool, artifact_id
