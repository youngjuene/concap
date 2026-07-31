"""`dpo artifact`: verify, trace, reindex, and collect the store."""

from __future__ import annotations

import argparse

from dpo.cli._shared import _emit
from dpo.core.artifacts import (
    GC_ROOT_TYPES,
    ArtifactStore,
)


def _artifact_verify(arguments: argparse.Namespace) -> int:
    store = ArtifactStore.open(arguments.workspace)
    if arguments.all:
        store.rebuild_index()
        verified = sorted(store.indexed_ids())
        for artifact_id in verified:
            store.verify(artifact_id)
        _emit({"status": "ok", "verified": len(verified)})
        return 0
    manifest = store.verify(arguments.artifact_id)
    _emit(
        {
            "status": "ok",
            "artifact_id": manifest.artifact_id,
            "artifact_type": manifest.artifact_type,
            "role_exposure": sorted(manifest.role_exposure),
        }
    )
    return 0


def _artifact_trace(arguments: argparse.Namespace) -> int:
    store = ArtifactStore.open(arguments.workspace)
    lineage = store.trace(arguments.artifact_id)
    _emit(
        {
            "status": "ok",
            "artifact_id": arguments.artifact_id,
            "lineage": [
                {"artifact_id": manifest.artifact_id, "artifact_type": manifest.artifact_type}
                for manifest in lineage
            ],
        }
    )
    return 0


def _artifact_rebuild_index(arguments: argparse.Namespace) -> int:
    store = ArtifactStore.open(arguments.workspace)
    store.rebuild_index()
    _emit({"status": "ok", "indexed": len(store.indexed_ids())})
    return 0


def _artifact_gc(arguments: argparse.Namespace) -> int:
    store = ArtifactStore.open(arguments.workspace)
    locks = set(arguments.lock or [])
    if arguments.execute and not locks:
        locks = {
            artifact_id
            for artifact_type in GC_ROOT_TYPES
            for artifact_id in store.find_by_type(artifact_type)
        }
    candidates = store.gc(locks, execute=arguments.execute)
    _emit(
        {
            "status": "ok",
            "executed": bool(arguments.execute),
            "candidates": candidates,
        }
    )
    return 0
