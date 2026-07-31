"""Study-export stage: caption generation and the memorization gate.

``generate_captions`` had no test at all before this — it was defined and never
called from ``src``. The gate it feeds is the one that decides what a human
participant reads, so both are covered here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dpo.contracts.study_contract import StudyContract
from dpo.core.artifacts import ArtifactStore
from dpo.core.identity import repo_lock_hash
from dpo.evaluation.caption_generation import generate_captions
from dpo.evaluation.compliance import GeneratedCaption
from dpo.models.tiny import TinyAdapter, synthetic_media
from dpo.pipeline.corpus_stage import publish_corpus_ingest, publish_lock_splits
from dpo.pipeline.publishing import ArtifactPublisher
from dpo.pipeline.run_matrix import DEFAULT_MEDIA_DIM
from dpo.pipeline.study_stage import STUDY_EXPORT_TYPE, StudyError, publish_study_export
from tests.conftest import PreferenceWorld


def _adapter(contract: StudyContract, track: str) -> TinyAdapter:
    return TinyAdapter(track=track, prompt=contract.tracks[track].prompt, media_dim=DEFAULT_MEDIA_DIM, seed=0)


def _publisher(tmp_path: Path, world: PreferenceWorld) -> tuple[ArtifactPublisher, dict[str, str]]:
    """A publisher plus the shard ids of the STUDY-role clips only.

    The store refuses an artifact whose asserted role disagrees with its clips'
    registry membership, and a study export asserts ``study`` — so the fixture
    hands back exactly the clips such an export may legitimately cover.
    """
    store = ArtifactStore.create(tmp_path / "store")
    publisher = ArtifactPublisher(store, world.contract, repo_lock_hash())
    ingest_id = publish_corpus_ingest(publisher, list(world.clips))
    _, manifest, shard_ids = publish_lock_splits(
        publisher, world.contract, list(world.clips), ingest_artifact_id=ingest_id
    )
    study = {clip_id: shard for clip_id, shard in shard_ids.items() if manifest.role_of(clip_id) == "study"}
    assert study, "the fixture world must hold at least one study-role clip"
    return publisher, study


def test_generate_captions_is_deterministic_and_clip_aligned(
    contract: StudyContract, world: PreferenceWorld
) -> None:
    track = world.pool.track
    clip_ids = sorted({candidate.clip_id for candidate in world.pool.candidates})[:3]
    media = synthetic_media(track, clip_ids, media_dim=DEFAULT_MEDIA_DIM)
    adapter = _adapter(contract, track)

    first = generate_captions(adapter, clip_ids, media, temperature=0.0, top_p=1.0, max_new_tokens=8, seed=1)
    second = generate_captions(adapter, clip_ids, media, temperature=0.0, top_p=1.0, max_new_tokens=8, seed=1)
    assert [caption.clip_id for caption in first] == clip_ids
    assert first == second, "greedy decoding under a frozen config must be reproducible"


def test_generate_captions_rejects_a_media_batch_for_other_clips(
    contract: StudyContract, world: PreferenceWorld
) -> None:
    track = world.pool.track
    clip_ids = sorted({candidate.clip_id for candidate in world.pool.candidates})[:2]
    media = synthetic_media(track, list(reversed(clip_ids)), media_dim=DEFAULT_MEDIA_DIM)
    with pytest.raises(ValueError, match="do not match"):
        generate_captions(
            _adapter(contract, track), clip_ids, media, temperature=0.0, top_p=1.0, max_new_tokens=4, seed=1
        )


def test_study_export_refuses_captions_that_reuse_training_candidates(
    tmp_path: Path, world: PreferenceWorld
) -> None:
    """A memorized caption would make the study measure recall, not generalization."""
    publisher, shard_ids = _publisher(tmp_path, world)
    clip_id = sorted(shard_ids)[0]
    # Text lifted verbatim from the frozen training pool, on a study-role clip.
    memorized = world.pool.candidates[0]
    with pytest.raises(StudyError, match="memorization"):
        publish_study_export(
            publisher,
            world.contract,
            track=world.pool.track,
            experiment_id="DPO",
            variant_id="base",
            validation_accuracy=1.0,
            captions=[GeneratedCaption(clip_id=clip_id, text=memorized.text)],
            training_pool=world.pool,
            lock_artifact_id=shard_ids[clip_id],
            shard_artifact_ids=shard_ids,
            decoding={"temperature": 0.0, "top_p": 1.0},
        )


def test_study_export_publishes_fresh_captions(tmp_path: Path, world: PreferenceWorld) -> None:
    publisher, shard_ids = _publisher(tmp_path, world)
    clip_id = sorted(shard_ids)[0]
    document, artifact_id = publish_study_export(
        publisher,
        world.contract,
        track=world.pool.track,
        experiment_id="DPO",
        variant_id="base",
        validation_accuracy=0.75,
        captions=[GeneratedCaption(clip_id=clip_id, text="A wholly unseen caption for the study split.")],
        training_pool=world.pool,
        lock_artifact_id=shard_ids[clip_id],
        shard_artifact_ids=shard_ids,
        decoding={"temperature": 0.0, "top_p": 1.0},
    )
    assert document["schema"] == STUDY_EXPORT_TYPE
    assert document["training_candidate_reuse_rate"] == 0.0
    assert document["experiment_id"] == "DPO"
    assert artifact_id.startswith("sha256:")


def test_study_export_requires_one_caption_per_clip(tmp_path: Path, world: PreferenceWorld) -> None:
    publisher, shard_ids = _publisher(tmp_path, world)
    clip_id = sorted(shard_ids)[0]
    duplicated = [
        GeneratedCaption(clip_id=clip_id, text="First caption for this clip."),
        GeneratedCaption(clip_id=clip_id, text="Second caption for the very same clip."),
    ]
    with pytest.raises(StudyError, match="exactly one caption per clip"):
        publish_study_export(
            publisher,
            world.contract,
            track=world.pool.track,
            experiment_id="DPO",
            variant_id="base",
            validation_accuracy=0.5,
            captions=duplicated,
            training_pool=world.pool,
            lock_artifact_id=shard_ids[clip_id],
            shard_artifact_ids=shard_ids,
            decoding={"temperature": 0.0, "top_p": 1.0},
        )
