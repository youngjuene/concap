"""`dpo annotation`: export sessions, serve the UI, ingest responses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dpo.annotation.collection_tasks import (
    annotations_from_responses,
    attention_expectations,
    audio_presentations_from_registry,
    build_collection_tasks,
)
from dpo.candidates.freeze import parse_frozen_pool
from dpo.cli._shared import _emit, _find_manifest, _operation, _registry_shard_rows, _require_types
from dpo.contracts.study_contract import (
    AUDIO_PRESENTATIONS,
)
from dpo.core.artifacts import (
    ArtifactError,
)
from dpo.pipeline.annotation_stage import ingest_annotations


def _annotation_export_tasks(arguments: argparse.Namespace) -> int:
    operation = _operation(arguments)
    _require_types(operation, {"dpo.frozen-candidate-pool/v1", "dpo.clip-registry/v1"}, minimum=2)
    pool_manifest = _find_manifest(operation, "dpo.frozen-candidate-pool/v1")
    registry_manifest = _find_manifest(operation, "dpo.clip-registry/v1")
    pool = parse_frozen_pool(operation.store.read_payload(pool_manifest.artifact_id))
    pool_clips = {candidate.clip_id for candidate in pool.candidates}
    shard_rows = _registry_shard_rows(
        operation.store, registry_manifest, lambda entry: str(entry["clip_id"]) in pool_clips
    )
    presentations = audio_presentations_from_registry([row for _, row in shard_rows.values()])
    if arguments.presentation is not None:
        # Serve every audio clip under one presentation regardless of what its
        # registry row opted into, so a session's presentation can be chosen at
        # export time without rewriting clip rows and re-ingesting the corpus.
        if arguments.presentation not in AUDIO_PRESENTATIONS:
            raise ArtifactError(
                f"--presentation must be one of {sorted(AUDIO_PRESENTATIONS)}; got {arguments.presentation!r}"
            )
        if pool.track != "audio":
            raise ArtifactError("--presentation overrides the audio-track presentation only")
        presentations = dict.fromkeys(presentations, arguments.presentation)
    tasks_document, answers_document = build_collection_tasks(
        operation.contract, pool, audio_presentations=presentations
    )
    Path(arguments.out_tasks).write_text(
        json.dumps(tasks_document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    Path(arguments.out_answers).write_text(
        json.dumps(answers_document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    task_rows = tasks_document["tasks"]
    assert isinstance(task_rows, list)
    kinds: dict[str, int] = {}
    for task in task_rows:
        kinds[str(task["kind"])] = kinds.get(str(task["kind"]), 0) + 1
    _emit(
        {
            "status": "exported",
            "operation": "annotation-export-tasks",
            "track": pool.track,
            "tasks": len(task_rows),
            "kinds": kinds,
            "out_tasks": str(Path(arguments.out_tasks).resolve()),
            "out_answers": str(Path(arguments.out_answers).resolve()),
        }
    )
    return 0


def _annotation_ingest(arguments: argparse.Namespace) -> int:
    operation = _operation(arguments)
    _require_types(operation, {"dpo.frozen-candidate-pool/v1"})
    pool_manifest = _find_manifest(operation, "dpo.frozen-candidate-pool/v1")
    pool = parse_frozen_pool(operation.store.read_payload(pool_manifest.artifact_id))
    tasks_document = json.loads(Path(arguments.tasks).read_text(encoding="utf-8"))
    answers_document = json.loads(Path(arguments.answers).read_text(encoding="utf-8"))
    annotations = []
    for responses_path in arguments.responses:
        responses_document = json.loads(Path(responses_path).read_text(encoding="utf-8"))
        annotations.extend(annotations_from_responses(tasks_document, responses_document))
    retained, aggregates, artifact_ids = ingest_annotations(
        operation.publisher(),
        operation.contract,
        track=pool.track,
        split=arguments.split,
        pool=pool,
        pool_artifact_id=pool_manifest.artifact_id,
        annotations=annotations,
        attention_expected=attention_expectations(answers_document),
    )
    _emit(
        {
            "status": "published",
            "operation": "annotation-ingest",
            "track": pool.track,
            "split": arguments.split,
            "annotations": len(annotations),
            "retained": len(retained),
            "aggregated_pairs": len(aggregates),
            "artifacts": artifact_ids,
        }
    )
    return 0


def _annotation_serve(arguments: argparse.Namespace) -> int:
    from dpo.annotation.webapp import run_collection_app

    run_collection_app(
        tasks_path=Path(arguments.tasks),
        media_dir=Path(arguments.media_dir),
        out_path=Path(arguments.out),
        host=arguments.host,
        port=arguments.port,
    )
    return 0
