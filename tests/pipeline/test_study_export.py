"""Study-export stage: the memorization gate that decides what a participant reads."""

from __future__ import annotations

from pathlib import Path

import pytest

from dpo.contracts.study_contract import StudyContract
from dpo.evaluation.congruency import ClipLadder, ScoredCaption
from dpo.models.tiny import TinyAdapter
from dpo.pipeline.run_matrix import DEFAULT_MEDIA_DIM
from dpo.pipeline.study_stage import STUDY_EXPORT_TYPE, StudyError, publish_study_export
from tests.conftest import PreferenceWorld, study_publisher


def _ladder(clip_id: str, low: str, high: str) -> ClipLadder:
    """A two-rung ladder: audio-only at the bottom, visually bound at the top."""
    return ClipLadder(
        clip_id=clip_id,
        rungs=(
            ScoredCaption(text=low, congruency=0.0, conditioning="audio"),
            ScoredCaption(text=high, congruency=0.6, conditioning="audio+video"),
        ),
    )


def _adapter(contract: StudyContract, track: str) -> TinyAdapter:
    return TinyAdapter(track=track, prompt=contract.tracks[track].prompt, media_dim=DEFAULT_MEDIA_DIM, seed=0)


def test_study_export_refuses_captions_that_reuse_training_candidates(
    tmp_path: Path, world: PreferenceWorld
) -> None:
    """A memorized caption would make the study measure recall, not generalization."""
    publisher, shard_ids = study_publisher(tmp_path, world)
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
            ladders=[_ladder(clip_id, memorized.text, "A fresh second rung.")],
            training_pool=world.pool,
            lock_artifact_id=shard_ids[clip_id],
            shard_artifact_ids=shard_ids,
            decoding={"temperature": 0.0, "top_p": 1.0},
        )


def test_study_export_publishes_fresh_captions(tmp_path: Path, world: PreferenceWorld) -> None:
    publisher, shard_ids = study_publisher(tmp_path, world)
    clip_id = sorted(shard_ids)[0]
    document, artifact_id = publish_study_export(
        publisher,
        world.contract,
        track=world.pool.track,
        experiment_id="DPO",
        variant_id="base",
        validation_accuracy=0.75,
        ladders=[
            _ladder(clip_id, "A wholly unseen caption for the study split.", "A visibly grounded variant.")
        ],
        training_pool=world.pool,
        lock_artifact_id=shard_ids[clip_id],
        shard_artifact_ids=shard_ids,
        decoding={"temperature": 0.0, "top_p": 1.0},
    )
    assert document["schema"] == STUDY_EXPORT_TYPE
    assert document["training_candidate_reuse_rate"] == 0.0
    assert document["experiment_id"] == "DPO"
    assert artifact_id.startswith("sha256:")
    # The slider reads these: stops placed by the measure, ascending, per clip.
    levels = document["clips"][0]["levels"]
    assert [entry["position"] for entry in levels] == [0.0, 1.0]
    assert [entry["congruency"] for entry in levels] == [0.0, 0.6]
    assert document["ladder_summary"]["negative_congruency_clips"] == []


def test_study_export_requires_one_caption_per_clip(tmp_path: Path, world: PreferenceWorld) -> None:
    publisher, shard_ids = study_publisher(tmp_path, world)
    clip_id = sorted(shard_ids)[0]
    duplicated = [
        _ladder(clip_id, "First rung here.", "Second rung here."),
        _ladder(clip_id, "Another first rung.", "Another second rung."),
    ]
    with pytest.raises(StudyError, match="exactly one ladder per clip"):
        publish_study_export(
            publisher,
            world.contract,
            track=world.pool.track,
            experiment_id="DPO",
            variant_id="base",
            validation_accuracy=0.5,
            ladders=duplicated,
            training_pool=world.pool,
            lock_artifact_id=shard_ids[clip_id],
            shard_artifact_ids=shard_ids,
            decoding={"temperature": 0.0, "top_p": 1.0},
        )


def test_study_export_refuses_a_non_monotone_ladder(tmp_path: Path, world: PreferenceWorld) -> None:
    """An unordered axis is not an independent variable."""
    publisher, shard_ids = study_publisher(tmp_path, world)
    clip_id = sorted(shard_ids)[0]
    backwards = ClipLadder(
        clip_id=clip_id,
        rungs=(
            ScoredCaption(text="more grounded", congruency=0.9, conditioning="audio+video"),
            ScoredCaption(text="less grounded", congruency=0.1, conditioning="audio"),
        ),
    )
    with pytest.raises(StudyError, match="non-monotone"):
        publish_study_export(
            publisher,
            world.contract,
            track=world.pool.track,
            experiment_id="DPO",
            variant_id="base",
            validation_accuracy=0.5,
            ladders=[backwards],
            training_pool=world.pool,
            lock_artifact_id=shard_ids[clip_id],
            shard_artifact_ids=shard_ids,
            decoding={"temperature": 0.0, "top_p": 1.0},
        )
