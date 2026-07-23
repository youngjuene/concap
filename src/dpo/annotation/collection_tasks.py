"""Collection-task export and response ingestion for the preference web UI.

``build_collection_tasks`` turns one frozen candidate pool into a
deterministic annotator session: every pair once (display order randomized by
the contract pair seed), repeat tasks for the contract's repeat fraction, and
attention checks built from mismatched captions of other clips. It returns a
public tasks document (safe to serve to annotators) and a restricted answers
document holding the attention expectations.

``annotations_from_responses`` joins one annotator's saved responses back
against those documents into :class:`RawAnnotation` records, ready for
``pipeline.annotation_stage.ingest_annotations``. Annotation ids derive
deterministically from (task id, annotator), so repeat tasks can reference
their primary judgment without any shared state.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from dpo.annotation.raw_annotations import AnnotationError, RawAnnotation
from dpo.candidates.freeze import FrozenCandidatePool
from dpo.contracts.study_contract import StudyContract
from dpo.core.identity import semantic_hash, sha256_bytes

TASKS_SCHEMA = "dpo.collection-tasks/v1"
ANSWERS_SCHEMA = "dpo.collection-answers/v1"
RESPONSES_SCHEMA = "dpo.collection-responses/v1"

TASK_KINDS = ("pair", "repeat", "attention")


def _flip(seed: int, *labels: object) -> bool:
    digest = semantic_hash({"seed": seed, "labels": list(labels)}).removeprefix("sha256:")
    return int(digest[:2], 16) % 2 == 1


def _presentation_for(track: str, audio_presentation: str | None) -> str:
    if track == "visual":
        return "muted_video"
    return audio_presentation if audio_presentation is not None else "audio_only"


def _annotation_id(task_id: str, annotator_id_hash: str) -> str:
    digest = semantic_hash({"task": task_id, "annotator": annotator_id_hash})
    return "ann-" + digest.removeprefix("sha256:")[:16]


def annotator_hash(name: str) -> str:
    return sha256_bytes(f"annotator:{name.strip()}".encode())


def build_collection_tasks(
    contract: StudyContract,
    pool: FrozenCandidatePool,
    *,
    audio_presentations: Mapping[str, str | None],
) -> tuple[dict[str, object], dict[str, object]]:
    """(public tasks document, restricted answers document) for one pool.

    ``audio_presentations`` maps clip ids to their audio-track presentation
    opt-in (from the clip registry); absent or None means audio_only.
    """
    seed = int(str(contract.pairs["seed"]))
    repeat_fraction = float(str(contract.annotation["repeat_fraction"]))
    attention_fraction = float(str(contract.annotation["attention_fraction"]))

    ordered_pairs = sorted(pool.pairs, key=lambda pair: semantic_hash({"seed": seed, "order": pair.pair_id}))
    tasks: list[dict[str, object]] = []
    for pair in ordered_pairs:
        flip = _flip(seed, "display", pair.pair_id)
        first, second = (pair.candidate_b, pair.candidate_a) if flip else (pair.candidate_a, pair.candidate_b)
        tasks.append(
            {
                "task_id": "task-" + semantic_hash({"pair": pair.pair_id}).removeprefix("sha256:")[:12],
                "kind": "pair",
                "pair_id": pair.pair_id,
                "clip_id": pair.clip_id,
                "track": pair.track,
                "presentation": _presentation_for(pair.track, audio_presentations.get(pair.clip_id)),
                "display": [
                    {"candidate_id": first, "text": pool.candidate(first).text},
                    {"candidate_id": second, "text": pool.candidate(second).text},
                ],
                "repeat_of_task": None,
            }
        )

    # Attention checks: a real candidate against a mismatched caption from a
    # different clip; the expected DISPLAYED choice is recorded only in the
    # restricted answers document.
    attention_expected: dict[str, str] = {}
    attention_tasks: list[dict[str, object]] = []
    attention_count = math.ceil(attention_fraction * len(ordered_pairs))
    for index in range(attention_count):
        pair = ordered_pairs[index % len(ordered_pairs)]
        other = ordered_pairs[(index + len(ordered_pairs) // 2) % len(ordered_pairs)]
        if other.clip_id == pair.clip_id:
            other = ordered_pairs[(index + 1) % len(ordered_pairs)]
        real_id = pair.candidate_a
        real_text = pool.candidate(real_id).text
        mismatch_text = pool.candidate(other.candidate_a).text
        pair_id = f"attn-{pool.track}-{index:03d}"
        mismatch_id = f"zz-attn-mismatch-{index:03d}"
        flip = _flip(seed, "attention", pair_id)
        first_slot, second_slot = (
            ((mismatch_id, mismatch_text), (real_id, real_text))
            if flip
            else ((real_id, real_text), (mismatch_id, mismatch_text))
        )
        attention_expected[pair_id] = "b_better" if flip else "a_better"
        attention_tasks.append(
            {
                "task_id": "task-" + semantic_hash({"attention": pair_id}).removeprefix("sha256:")[:12],
                "kind": "attention",
                "pair_id": pair_id,
                "clip_id": pair.clip_id,
                "track": pair.track,
                "presentation": _presentation_for(pair.track, audio_presentations.get(pair.clip_id)),
                "display": [
                    {"candidate_id": first_slot[0], "text": first_slot[1]},
                    {"candidate_id": second_slot[0], "text": second_slot[1]},
                ],
                "repeat_of_task": None,
            }
        )

    # Repeats re-show the first pairs of the session near the end, with an
    # independently randomized display order; the reliability screen compares
    # canonical outcomes, not displayed slots.
    repeat_tasks: list[dict[str, object]] = []
    repeat_count = math.ceil(repeat_fraction * len(ordered_pairs))
    for pair in ordered_pairs[:repeat_count]:
        primary_task_id = "task-" + semantic_hash({"pair": pair.pair_id}).removeprefix("sha256:")[:12]
        flip = _flip(seed, "repeat", pair.pair_id)
        first, second = (pair.candidate_b, pair.candidate_a) if flip else (pair.candidate_a, pair.candidate_b)
        repeat_tasks.append(
            {
                "task_id": "task-" + semantic_hash({"repeat": pair.pair_id}).removeprefix("sha256:")[:12],
                "kind": "repeat",
                "pair_id": pair.pair_id,
                "clip_id": pair.clip_id,
                "track": pair.track,
                "presentation": _presentation_for(pair.track, audio_presentations.get(pair.clip_id)),
                "display": [
                    {"candidate_id": first, "text": pool.candidate(first).text},
                    {"candidate_id": second, "text": pool.candidate(second).text},
                ],
                "repeat_of_task": primary_task_id,
            }
        )

    # Interleave: attention checks spread through the session at a fixed
    # stride, repeats appended at the end so their primaries come first.
    combined = list(tasks)
    if attention_tasks:
        stride = max(1, len(combined) // (len(attention_tasks) + 1))
        for index, task in enumerate(attention_tasks):
            combined.insert(min(len(combined), (index + 1) * stride + index), task)
    combined.extend(repeat_tasks)

    tasks_document: dict[str, object] = {
        "schema": TASKS_SCHEMA,
        "track": pool.track,
        "dataset_version": pool.dataset_version,
        "collection_version": str(contract.annotation["collection_version"]),
        "tasks": combined,
    }
    answers_document: dict[str, object] = {
        "schema": ANSWERS_SCHEMA,
        "track": pool.track,
        "dataset_version": pool.dataset_version,
        "attention_expected": attention_expected,
    }
    return tasks_document, answers_document


def _require(value: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in value:
        raise AnnotationError(f"{context} is missing {key!r}")
    return value[key]


def annotations_from_responses(
    tasks_document: Mapping[str, Any],
    responses_document: Mapping[str, Any],
) -> list[RawAnnotation]:
    """Join one annotator's saved responses against the tasks document."""
    if tasks_document.get("schema") != TASKS_SCHEMA:
        raise AnnotationError(f"tasks document schema must be {TASKS_SCHEMA!r}")
    if responses_document.get("schema") != RESPONSES_SCHEMA:
        raise AnnotationError(f"responses document schema must be {RESPONSES_SCHEMA!r}")
    annotator = str(_require(responses_document, "annotator", "responses document")).strip()
    if not annotator:
        raise AnnotationError("responses document requires a non-empty annotator name")
    annotator_id = annotator_hash(annotator)
    collection_version = str(_require(tasks_document, "collection_version", "tasks document"))
    tasks_by_id: dict[str, Mapping[str, Any]] = {}
    for task in _require(tasks_document, "tasks", "tasks document"):
        tasks_by_id[str(task["task_id"])] = task

    annotations: list[RawAnnotation] = []
    for response in _require(responses_document, "responses", "responses document"):
        task_id = str(_require(response, "task_id", "response"))
        task = tasks_by_id.get(task_id)
        if task is None:
            raise AnnotationError(f"response references unknown task {task_id!r}")
        display = [str(slot["candidate_id"]) for slot in task["display"]]
        candidate_a, candidate_b = sorted(display)
        kind = str(task["kind"])
        repeat_of_task = task.get("repeat_of_task")
        strength = response.get("preference_strength")
        tie_subtype = response.get("tie_subtype")
        annotations.append(
            RawAnnotation(
                annotation_id=_annotation_id(task_id, annotator_id),
                pair_id=str(task["pair_id"]),
                clip_id=str(task["clip_id"]),
                track=str(task["track"]),
                candidate_a=candidate_a,
                candidate_b=candidate_b,
                display_order=(display[0], display[1]),
                choice=str(_require(response, "choice", "response")),
                tie_subtype=None if tie_subtype is None else str(tie_subtype),
                preference_strength=None if strength is None else int(str(strength)),
                confidence=int(str(_require(response, "confidence", "response"))),
                reason_tags=tuple(str(tag) for tag in response.get("reason_tags", [])),
                annotator_id_hash=annotator_id,
                replay_count=int(str(response.get("replay_count", 0))),
                response_time_ms=int(str(_require(response, "response_time_ms", "response"))),
                collection_version=collection_version,
                presentation=str(task["presentation"]),
                is_attention_check=kind == "attention",
                repeat_of=(
                    None if repeat_of_task is None else _annotation_id(str(repeat_of_task), annotator_id)
                ),
            )
        )
    return annotations


def attention_expectations(answers_document: Mapping[str, Any]) -> dict[str, str]:
    if answers_document.get("schema") != ANSWERS_SCHEMA:
        raise AnnotationError(f"answers document schema must be {ANSWERS_SCHEMA!r}")
    expected = _require(answers_document, "attention_expected", "answers document")
    return {str(pair_id): str(choice) for pair_id, choice in expected.items()}


def audio_presentations_from_registry(rows: Sequence[Mapping[str, Any]]) -> dict[str, str | None]:
    """clip_id -> audio presentation opt-in, from clip-registry payload rows."""
    result: dict[str, str | None] = {}
    for row in rows:
        value = row.get("audio_presentation")
        result[str(row["clip_id"])] = None if value is None else str(value)
    return result
