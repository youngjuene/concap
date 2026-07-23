"""Shared study-stage implementation: the blinded human-study export.

Every selected model captions the study clips under the one frozen decoding
budget (contract ``[validation]``), the blinded incomplete-block export is
built, and its public documents are published parented on the lock manifest
and the study shards. The full exports — including the restricted
randomization manifests — are returned to the caller, never published.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from dpo.contracts.study_contract import EXPERIMENT_IDS, TRACKS, StudyContract
from dpo.core.artifacts import ParentEdge
from dpo.data.split import ClipInput
from dpo.evaluation.human_study_export import StudyExport, build_study_export
from dpo.models.base import MediaBatch, ModelAdapter
from dpo.pipeline.publishing import ArtifactPublisher


def publish_study_export(
    publisher: ArtifactPublisher,
    contract: StudyContract,
    *,
    lock_artifact_id: str,
    study_clips: Sequence[str],
    clips: Sequence[ClipInput],
    shard_ids: Mapping[str, str],
    policies: Mapping[tuple[str, str, str, int], ModelAdapter],
    selected_variants: Mapping[str, Mapping[str, str]],
    canonical_seed: int,
    media_provider: Callable[[str, Sequence[str]], MediaBatch],
) -> tuple[dict[str, StudyExport], str]:
    """Build and publish the blinded study export; return the full exports."""
    validation_decoding = contract.validation
    study_documents: dict[str, dict[str, list[dict[str, object]]]] = {}
    exports: dict[str, StudyExport] = {}
    for track in TRACKS:
        media = media_provider(track, study_clips)
        captions_by_model: dict[str, dict[str, str]] = {}
        for experiment_id in EXPERIMENT_IDS:
            policy_adapter = policies[
                (experiment_id, selected_variants[track][experiment_id], track, canonical_seed)
            ]
            # The one frozen decoding budget (contract [validation]) serves every
            # model in every evaluation surface, the study export included.
            texts = policy_adapter.generate(
                media,
                temperature=float(str(validation_decoding["temperature"])),
                top_p=float(str(validation_decoding["top_p"])),
                max_new_tokens=int(str(validation_decoding["max_new_tokens"])),
                seed=canonical_seed,
            )
            captions_by_model[experiment_id] = {
                clip_id: (text or "(empty)") for clip_id, text in zip(study_clips, texts, strict=True)
            }
        export = build_study_export(
            track=track,
            captions=captions_by_model,
            media_hashes={
                clip.clip_id: clip.media_hash for clip in clips if clip.clip_id in set(study_clips)
            },
            exposures_per_pair=int(str(contract.study["exposures_per_model_pair"])),
        )
        exports[track] = export
        study_documents[track] = export.public_documents()
    study_export_artifact = publisher.publish(
        "dpo.study-export/v1",
        {"schema": "dpo.study-export/v1", "tracks": study_documents},
        parents=(ParentEdge(lock_artifact_id, "lock-manifest"),)
        + tuple(ParentEdge(shard_ids[clip_id], "study-shard") for clip_id in study_clips),
        stage="study",
        parameters={"operation": "study-export"},
        clips=set(study_clips),
        role_exposure={"study"},
    )
    return exports, study_export_artifact
