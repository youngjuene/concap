"""Collection-task export, response joining, and ingest over the web-session documents."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dpo.annotation.collection_tasks import (
    RESPONSES_SCHEMA,
    annotations_from_responses,
    attention_expectations,
    build_collection_tasks,
)
from dpo.core.artifacts import ArtifactStore
from dpo.core.identity import repo_lock_hash
from dpo.pipeline.annotation_stage import ingest_annotations
from dpo.pipeline.candidate_stage import publish_frozen_pool
from dpo.pipeline.corpus_stage import publish_corpus_ingest, publish_lock_splits
from dpo.pipeline.publishing import ArtifactPublisher
from tests.conftest import PreferenceWorld

TASK_KEYS = {"task_id", "kind", "pair_id", "clip_id", "track", "presentation", "display", "repeat_of_task"}


def _documents(world: PreferenceWorld) -> tuple[dict[str, object], dict[str, object]]:
    return build_collection_tasks(world.contract, world.pool, audio_presentations={})


def _simulated_responses(
    tasks_document: Mapping[str, Any],
    answers_document: Mapping[str, Any],
    *,
    annotator: str,
) -> dict[str, object]:
    """One diligent annotator's full session.

    Attention checks are answered with the registered expectation and the
    chosen display side alternates across the remaining tasks: an annotator
    who literally always chose the left slot would (correctly) be excluded by
    the contract's preregistered position-bias screen, emptying the retained
    set that the ingest assertions need.
    """
    expected = attention_expectations(answers_document)
    responses: list[dict[str, object]] = []
    alternating = 0
    for task in tasks_document["tasks"]:
        if task["kind"] == "attention":
            choice = expected[str(task["pair_id"])]
        else:
            choice = "a_better" if alternating % 2 == 0 else "b_better"
            alternating += 1
        responses.append(
            {
                "task_id": task["task_id"],
                "choice": choice,
                "tie_subtype": None,
                "preference_strength": 4,
                "confidence": 4,
                "reason_tags": ["coverage"],
                "response_time_ms": 5200,
                "replay_count": 1,
            }
        )
    return {"schema": RESPONSES_SCHEMA, "annotator": annotator, "responses": responses}


def test_collection_tasks_match_the_contract_fractions(world: PreferenceWorld) -> None:
    tasks_document, answers_document = _documents(world)
    tasks = tasks_document["tasks"]
    assert isinstance(tasks, list)
    pair_count = len(world.pool.pairs)
    repeat_count = math.ceil(float(str(world.contract.annotation["repeat_fraction"])) * pair_count)
    attention_count = math.ceil(float(str(world.contract.annotation["attention_fraction"])) * pair_count)
    kinds = Counter(str(task["kind"]) for task in tasks)
    assert kinds["pair"] == pair_count
    assert kinds["repeat"] == repeat_count
    assert kinds["attention"] == attention_count
    assert len(tasks) == pair_count + repeat_count + attention_count
    # Every pair/repeat display is a permutation of its frozen pair.
    for task in tasks:
        if task["kind"] == "attention":
            continue
        pair = world.pool.pair(str(task["pair_id"]))
        displayed = sorted(str(slot["candidate_id"]) for slot in task["display"])
        assert displayed == sorted([pair.candidate_a, pair.candidate_b])
    # Attention expectations live only in the restricted answers document.
    assert all(set(task) == TASK_KEYS for task in tasks)
    attention_ids = {str(task["pair_id"]) for task in tasks if task["kind"] == "attention"}
    assert set(attention_expectations(answers_document)) == attention_ids
    assert set(answers_document) == {"schema", "track", "dataset_version", "attention_expected"}


def test_responses_roundtrip_and_ingest_publish(world: PreferenceWorld, tmp_path: Path) -> None:
    tasks_document, answers_document = _documents(world)
    tasks = tasks_document["tasks"]
    assert isinstance(tasks, list)
    annotations = []
    for annotator in ("Annotator One", "Annotator Two", "Annotator Three"):
        responses_document = _simulated_responses(tasks_document, answers_document, annotator=annotator)
        joined = annotations_from_responses(tasks_document, responses_document)
        assert len(joined) == len(tasks)
        # Repeat tasks link back to the same annotator's primary annotation.
        by_task = {str(task["task_id"]): annotation for task, annotation in zip(tasks, joined, strict=True)}
        for task, annotation in zip(tasks, joined, strict=True):
            assert annotation.pair_id == str(task["pair_id"])
            assert annotation.is_attention_check == (task["kind"] == "attention")
            if task["kind"] == "repeat":
                assert annotation.repeat_of == by_task[str(task["repeat_of_task"])].annotation_id
            else:
                assert annotation.repeat_of is None
        annotations.extend(joined)

    store = ArtifactStore.create(tmp_path / "store")
    publisher = ArtifactPublisher(store, world.contract, repo_lock_hash())
    ingest_id = publish_corpus_ingest(publisher, list(world.clips))
    _, _, shard_ids = publish_lock_splits(
        publisher, world.contract, list(world.clips), ingest_artifact_id=ingest_id
    )
    pool, pool_artifact_id = publish_frozen_pool(
        publisher,
        world.contract,
        track=world.pool.track,
        split="train",
        candidates=list(world.pool.candidates),
        audits=world.audits,
        shard_artifact_ids=shard_ids,
        dataset_version=world.pool.dataset_version,
        audit_version=world.pool.evidence_audit_version,
    )
    assert pool.pool_hash == world.pool.pool_hash
    retained, aggregates, artifact_ids = ingest_annotations(
        publisher,
        world.contract,
        track=pool.track,
        split="train",
        pool=pool,
        pool_artifact_id=pool_artifact_id,
        annotations=annotations,
        attention_expected=attention_expectations(answers_document),
    )
    assert retained
    assert aggregates
    assert set(artifact_ids) == {"raw_annotations", "reliability_report"}
    for artifact_id in artifact_ids.values():
        assert artifact_id in store.indexed_ids()
        store.verify(artifact_id)
