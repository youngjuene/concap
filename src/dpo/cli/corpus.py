"""`dpo corpus`: ingest clip rows and lock the immutable splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dpo.cli._shared import _emit, _operation, _require_types
from dpo.data.split import ClipInput, SplitError
from dpo.pipeline.corpus_stage import publish_corpus_ingest, publish_lock_splits
from dpo.pipeline.stages import stage


def _read_clip_rows(path: str) -> list[ClipInput]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SplitError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise SplitError(f"{path}:{line_number}: expected a JSON object")
            allowed = {
                "clip_id",
                "source_video_id",
                "media_hash",
                "start_ms",
                "end_ms",
                "derivative_hashes",
                "link_group",
                "asserted_role",
                "audio_presentation",
            }
            unknown = sorted(set(value) - allowed)
            if unknown:
                raise SplitError(f"{path}:{line_number}: unknown field {unknown[0]!r}")
            try:
                rows.append(ClipInput.from_document(value))
            except SplitError as exc:
                raise SplitError(f"{path}:{line_number}: {exc}") from exc
    if not rows:
        raise SplitError(f"{path}: no clip rows")
    return rows


def _corpus_ingest(arguments: argparse.Namespace) -> int:
    operation = _operation(arguments)
    clips = _read_clip_rows(arguments.input)
    artifact_id = publish_corpus_ingest(operation.publisher(), clips)
    _emit(
        {
            "status": "published",
            "operation": "corpus-ingest",
            "artifact_id": artifact_id,
            "artifact_type": "dpo.corpus-ingest/v1",
            "row_count": len(clips),
        }
    )
    return 0


def _corpus_lock_splits(arguments: argparse.Namespace) -> int:
    operation = _operation(arguments)
    _require_types(operation, set(stage("lock-splits").input_artifact_types))
    ingest = operation.manifests[0]
    payload = json.loads(operation.store.read_payload(ingest.artifact_id))
    clips = [ClipInput.from_document(row) for row in payload["rows"]]
    registry_artifact, manifest, shard_ids = publish_lock_splits(
        operation.publisher(),
        operation.contract,
        clips,
        ingest_artifact_id=ingest.artifact_id,
    )
    _emit(
        {
            "status": "published",
            "operation": "lock-splits",
            "artifact_id": registry_artifact,
            "artifact_type": "dpo.clip-registry/v1",
            "split_manifest": manifest.signed_document(),
            "shards": shard_ids,
        }
    )
    return 0
