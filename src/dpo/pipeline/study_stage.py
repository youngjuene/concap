"""Study-export stage: the held-out study split's captions, from the locked winner.

This is the boundary between the alignment pipeline and the human study that
consumes it. Everything upstream measures models against held-out preferences;
the artifact published here is what a participant actually reads.

Three properties are load-bearing.

``dpo.study-export/v1`` is in ``PUBLIC_DERIVED_TYPES``, so its payload is exempt
from the protected-role capability gate even though its ancestry is not. That is
what lets a study web process serve these captions without holding a study
capability, while the study clips' own rows stay sealed.

The lock manifest is a required input, not the selection report alone. The study
split is held out exactly as the test split is, and the pipeline's rule is that
configuration freezes before held-out access. Requiring the lock makes that
ordering unrepresentable rather than merely documented.

A caption that byte-matches a frozen training candidate fails the publish. If
participants read back a memorized training string, the study measures
memorization and reports it as caption quality — and unlike most defects, that
one produces perfectly plausible output.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from dpo.candidates.freeze import FrozenCandidatePool
from dpo.contracts.study_contract import StudyContract
from dpo.core.artifacts import ParentEdge
from dpo.evaluation.caption_generation import training_candidate_reuse_rate
from dpo.evaluation.compliance import GeneratedCaption
from dpo.pipeline.publishing import ArtifactPublisher

STUDY_EXPORT_TYPE = "dpo.study-export/v1"


class StudyError(ValueError):
    """Raised when a study export would carry something a participant must not see."""


def publish_study_export(
    publisher: ArtifactPublisher,
    contract: StudyContract,
    *,
    track: str,
    experiment_id: str,
    variant_id: str,
    validation_accuracy: float,
    captions: Sequence[GeneratedCaption],
    training_pool: FrozenCandidatePool,
    lock_artifact_id: str,
    shard_artifact_ids: Mapping[str, str],
    decoding: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    """Freeze one track's study captions and publish them for the human study."""
    if not captions:
        raise StudyError("a study export needs at least one caption")
    clip_ids = [caption.clip_id for caption in captions]
    if len(set(clip_ids)) != len(clip_ids):
        raise StudyError("a study export must hold exactly one caption per clip")
    missing = sorted(set(clip_ids) - set(shard_artifact_ids))
    if missing:
        raise StudyError(f"caption for clip {missing[0]!r} has no registry shard to descend from")

    reuse_rate = training_candidate_reuse_rate(captions, training_pool)
    if reuse_rate > 0.0:
        reused = sorted(
            caption.clip_id
            for caption in captions
            if caption.text in {candidate.text for candidate in training_pool.candidates}
        )
        raise StudyError(
            f"{reuse_rate:.0%} of study captions byte-match a frozen training candidate"
            f" (first: clip {reused[0]!r}); the model is reproducing training text, so the"
            " study would measure memorization rather than generalization"
        )

    document: dict[str, object] = {
        "schema": STUDY_EXPORT_TYPE,
        "track": track,
        "experiment_id": experiment_id,
        "variant_id": variant_id,
        "validation_accuracy": validation_accuracy,
        "decoding": dict(sorted(decoding.items())),
        "training_candidate_reuse_rate": reuse_rate,
        "captions": [
            {"clip_id": caption.clip_id, "text": caption.text}
            for caption in sorted(captions, key=lambda item: item.clip_id)
        ],
    }
    artifact_id = publisher.publish(
        STUDY_EXPORT_TYPE,
        document,
        parents=(ParentEdge(lock_artifact_id, "locked-configuration"),)
        + tuple(ParentEdge(shard_artifact_ids[clip_id], "clip-shard") for clip_id in sorted(clip_ids)),
        stage="study-export",
        parameters={"operation": "study-export", "track": track, "experiment_id": experiment_id},
        row_count=len(captions),
        clips=set(clip_ids),
        role_exposure={"study"},
        attributes={"track": track, "experiment_id": experiment_id, "variant_id": variant_id},
        purpose="human-study",
    )
    return document, artifact_id
