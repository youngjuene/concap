"""Content-addressed store invariants: identity, tamper, roles, fencing, GC."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dpo.core.access import AccessDenied, ProtectedAccessAuthority
from dpo.core.artifacts import (
    ArtifactConflict,
    ArtifactError,
    ArtifactStore,
    ArtifactTampered,
    ParentEdge,
    ProtectedExposure,
    RequestSpec,
)
from dpo.core.identity import canonical_bytes, semantic_hash, sha256_bytes

CONTRACT_ID = sha256_bytes(b"test-contract")
LOCK_ID = sha256_bytes(b"test-lock")


def _request(
    artifact_type: str = "dpo.fixture/v1",
    parents: tuple[ParentEdge, ...] = (),
    parameters: dict[str, object] | None = None,
    seed: int = 17,
) -> RequestSpec:
    return RequestSpec(
        artifact_type=artifact_type,
        parents=parents,
        contract_id=CONTRACT_ID,
        model_pins={},
        prompt_pins={},
        parameters=parameters if parameters is not None else {"operation": artifact_type},
        seed=seed,
        lock_id=LOCK_ID,
    )


def _registry_rows(roles: dict[str, str]) -> list[dict[str, object]]:
    return [
        {
            "clip_id": clip_id,
            "source_video_id": f"src-{clip_id}",
            "media_hash": sha256_bytes(f"media-{clip_id}".encode()),
            "start_ms": 0,
            "end_ms": 5000,
            "group_id": f"grp-{clip_id}",
            "derivative_hashes": [sha256_bytes(f"deriv-{clip_id}".encode())],
            "role": role,
        }
        for clip_id, role in sorted(roles.items())
    ]


def _publish_registry(store: ArtifactStore, roles: dict[str, str]) -> str:
    rows = _registry_rows(roles)
    registry_id = semantic_hash({"schema": "dpo.clip-registry/v1", "contract_id": CONTRACT_ID, "clips": rows})
    manifest = store.publish(
        _request(
            "dpo.clip-registry/v1",
            parameters={"operation": "registry", "registry_id": registry_id},
        ),
        canonical_bytes({"schema": "dpo.clip-registry/v1", "registry_id": registry_id, "clips": rows}),
        row_count=len(rows),
        clips=set(roles),
    )
    return manifest.artifact_id


def test_publish_verify_roundtrip_and_idempotence(tmp_path: Path) -> None:
    store = ArtifactStore.create(tmp_path / "store")
    manifest = store.publish(_request(), canonical_bytes({"hello": 1}))
    again = store.publish(_request(), canonical_bytes({"hello": 1}))
    assert manifest.artifact_id == again.artifact_id
    verified = store.verify(manifest.artifact_id)
    assert verified.artifact_type == "dpo.fixture/v1"


def test_same_request_different_content_is_a_first_writer_conflict(tmp_path: Path) -> None:
    store = ArtifactStore.create(tmp_path / "store")
    store.publish(_request(), canonical_bytes({"hello": 1}))
    with pytest.raises(ArtifactConflict):
        store.publish(_request(), canonical_bytes({"hello": 2}))


def test_payload_tamper_is_detected(tmp_path: Path) -> None:
    store = ArtifactStore.create(tmp_path / "store")
    manifest = store.publish(_request(), canonical_bytes({"hello": 1}))
    payload_path = store.payload_path(manifest.artifact_id)
    payload_path.write_bytes(b"corrupted")
    with pytest.raises(ArtifactTampered):
        store.verify(manifest.artifact_id)


def test_manifest_tamper_is_detected(tmp_path: Path) -> None:
    store = ArtifactStore.create(tmp_path / "store")
    manifest = store.publish(_request(), canonical_bytes({"hello": 1}))
    manifest_path = store.manifest_path(manifest.artifact_id)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["semantic"]["row_count"] = 999
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ArtifactTampered):
        store.verify(manifest.artifact_id)


def test_registry_roles_and_recursive_exposure(tmp_path: Path) -> None:
    store = ArtifactStore.create(tmp_path / "store")
    registry_id = _publish_registry(store, {"clip-a": "train", "clip-b": "test"})
    child = store.publish(
        _request("dpo.derived/v1", parameters={"operation": "derived"}),
        canonical_bytes({"rows": []}),
        clips={"clip-a"},
    )
    assert child.role_exposure == frozenset({"train"})
    grandchild = store.publish(
        _request(
            "dpo.derived-2/v1",
            parents=(ParentEdge(child.artifact_id, "input"),),
            parameters={"operation": "derived-2"},
        ),
        canonical_bytes({"rows": [1]}),
    )
    assert grandchild.role_exposure == frozenset({"train"})
    assert registry_id in {edge.artifact_id for edge in child.registry_dependencies}


def test_unregistered_clip_membership_is_rejected(tmp_path: Path) -> None:
    store = ArtifactStore.create(tmp_path / "store")
    with pytest.raises(ArtifactError, match="registry membership"):
        store.publish(
            _request("dpo.derived/v1", parameters={"operation": "derived"}),
            canonical_bytes({"rows": []}),
            clips={"clip-unknown"},
        )


def test_training_purpose_rejects_protected_ancestry(tmp_path: Path) -> None:
    store = ArtifactStore.create(tmp_path / "store")
    _publish_registry(store, {"clip-a": "train", "clip-b": "test"})
    with pytest.raises(ProtectedExposure):
        store.publish(
            _request("dpo.training-thing/v1", parameters={"operation": "train"}),
            canonical_bytes({"rows": []}),
            clips={"clip-b"},
            purpose="training",
        )


def test_protected_payloads_fail_closed_and_require_fenced_capability(tmp_path: Path) -> None:
    store = ArtifactStore.create(tmp_path / "store")
    roles = {"clip-a": "train", "clip-b": "test", "clip-c": "study"}
    _publish_registry(store, roles)
    registry_id = semantic_hash(
        {"schema": "dpo.clip-registry/v1", "contract_id": CONTRACT_ID, "clips": _registry_rows(roles)}
    )

    def _shard(clip_id: str, role: str) -> str:
        manifest = store.publish(
            _request(
                "dpo.clip-registry-shard/v1",
                parameters={"operation": "shard", "registry_id": registry_id, "clip_id": clip_id},
            ),
            canonical_bytes(_registry_rows({clip_id: role})[0]),
            row_count=1,
            clips={clip_id},
        )
        return manifest.artifact_id

    # Test-role bytes fail closed: with the confirmatory phase out of the
    # default pipeline there is no capability path, so no read is possible.
    test_shard = _shard("clip-b", "test")
    with pytest.raises(AccessDenied):
        store.read_payload(test_shard)
    # Study-role bytes require a fenced capability bound to one semantic hash.
    study_shard = _shard("clip-c", "study")
    with pytest.raises(AccessDenied):
        store.read_payload(study_shard)
    reservation_semantic = sha256_bytes(b"study-read")
    authority = ProtectedAccessAuthority(tmp_path / "store" / "study.sqlite")
    capability = authority.reserve(scope="human-study", semantic_hash=reservation_semantic, roles={"study"})
    payload = store.read_payload(
        study_shard,
        capability=capability,
        authority=authority,
        semantic_hash=reservation_semantic,
    )
    assert json.loads(payload)["clip_id"] == "clip-c"
    with pytest.raises(AccessDenied):
        store.read_payload(
            study_shard,
            capability=capability,
            authority=authority,
            semantic_hash=sha256_bytes(b"other-semantic"),
        )


def test_gc_preserves_lock_reachable_lineage(tmp_path: Path) -> None:
    store = ArtifactStore.create(tmp_path / "store")
    parent = store.publish(_request(), canonical_bytes({"hello": 1}))
    kept = store.publish(
        _request(
            "dpo.lock-manifest/v1",
            parents=(ParentEdge(parent.artifact_id, "input"),),
            parameters={"operation": "lock"},
        ),
        canonical_bytes({"lock": True}),
    )
    orphan = store.publish(
        _request("dpo.orphan/v1", parameters={"operation": "orphan"}),
        canonical_bytes({"orphan": True}),
    )
    candidates = store.gc({kept.artifact_id})
    assert orphan.artifact_id in candidates
    assert parent.artifact_id not in candidates
    assert kept.artifact_id not in candidates
    store.gc({kept.artifact_id}, execute=True)
    assert not store.object_dir(orphan.artifact_id).exists()
    store.verify(parent.artifact_id)
