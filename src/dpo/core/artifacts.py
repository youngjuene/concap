from __future__ import annotations

import fcntl
import functools
import json
import os
import platform
import re
import sqlite3
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dpo
from dpo.core.access import AccessCapability, AccessDenied, ProtectedAccessAuthority
from dpo.core.atomic import atomic_write_bytes
from dpo.core.identity import canonical_bytes, semantic_hash, sha256_bytes
from dpo.core.safety import OwnedWorkspace


class ArtifactError(RuntimeError):
    pass


class ArtifactConflict(ArtifactError):
    pass


class ArtifactTampered(ArtifactError):
    pass


class ProtectedExposure(ArtifactError):
    pass


ID_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
PROTECTED_ROLES = frozenset({"validation", "test", "study"})
CAPABILITY_READ_ROLES = frozenset({"test", "study"})
# These payloads are code-owned aggregates/control records rather than protected row data.
# Everything else with protected ancestry is denied by default.
PUBLIC_DERIVED_TYPES = frozenset(
    {
        "dpo.canary-report/v1",
        "dpo.test-metrics/v1",
        "dpo.test-reservation/v1",
        "dpo.test-resume/v1",
        "dpo.test-finalization/v1",
        "dpo.lock-manifest/v1",
        "dpo.selection-report/v1",
        "dpo.study-export/v1",
        "dpo.study-results/v1",
        "dpo.analysis-report/v1",
    }
)
ALL_ROLES = frozenset({"train", *PROTECTED_ROLES})
REGISTRY_TYPES = frozenset({"dpo.clip-registry/v1", "dpo.clip-registry-shard/v1"})
UNTRUSTED_SOURCE_TYPES = frozenset({"dpo.corpus-ingest/v1", "dpo.fixture/v1"})
GC_ROOT_TYPES = frozenset(
    {
        "dpo.lock-manifest/v1",
        "dpo.canary-report/v1",
        # Active/interruptible authorities are roots, not merely descendants of
        # a future terminal artifact.  Retaining historical completed records is
        # conservative and prevents GC from racing a state transition.
        "dpo.test-reservation/v1",
        "dpo.test-resume/v1",
        "dpo.test-finalization/v1",
        "dpo.study-export/v1",
        "dpo.study-results/v1",
        "dpo.analysis-report/v1",
    }
)
LOCKED_CLIP_FIELDS = frozenset(
    {
        "clip_id",
        "source_video_id",
        "media_hash",
        "start_ms",
        "end_ms",
        "group_id",
        "derivative_hashes",
        "role",
    }
)
SEMANTIC_FIELDS = frozenset(
    {
        "schema",
        "artifact_type",
        "request_id",
        "request",
        "parents",
        "payload_hash",
        "payload_size",
        "row_count",
        "clips",
        "groups",
        "registry_membership",
        "local_role_exposure",
        "role_exposure",
        "local_fit_exposure",
        "local_selection_exposure",
        "local_test_exposure",
        "fit_exposure",
        "selection_exposure",
        "test_exposure",
        "protected_sources",
        "attributes",
    }
)


@dataclass(frozen=True)
class ParentEdge:
    artifact_id: str
    edge_type: str

    def __post_init__(self) -> None:
        if not ID_RE.fullmatch(self.artifact_id) or not self.edge_type.strip():
            raise ArtifactError("parent edges require a valid artifact id and nonempty type")

    def document(self) -> dict[str, str]:
        return {"artifact_id": self.artifact_id, "edge_type": self.edge_type}


@dataclass(frozen=True)
class RequestSpec:
    artifact_type: str
    parents: tuple[ParentEdge, ...]
    contract_id: str
    model_pins: Mapping[str, object]
    prompt_pins: Mapping[str, object]
    parameters: Mapping[str, object]
    seed: int
    lock_id: str

    def __post_init__(self) -> None:
        if not self.artifact_type.startswith("dpo.") or "/v" not in self.artifact_type:
            raise ArtifactError("artifact type must be a namespaced versioned dpo type")
        if not ID_RE.fullmatch(self.contract_id) or not ID_RE.fullmatch(self.lock_id):
            raise ArtifactError("request contract and lock identities must be sha256 artifact ids")
        if type(self.seed) is not int or self.seed < 0:
            raise ArtifactError("request seed must be a non-negative integer")

    def document(self) -> dict[str, object]:
        return {
            "schema": "dpo.request/v1",
            "artifact_type": self.artifact_type,
            "parents": [parent.document() for parent in self.parents],
            "contract_id": self.contract_id,
            "model_pins": dict(self.model_pins),
            "prompt_pins": dict(self.prompt_pins),
            "parameters": dict(self.parameters),
            "seed": self.seed,
            "lock_id": self.lock_id,
        }

    @property
    def request_id(self) -> str:
        return semantic_hash(self.document())


@dataclass(frozen=True)
class ArtifactManifest:
    artifact_id: str
    semantic: dict[str, Any]

    @property
    def request_id(self) -> str:
        return str(self.semantic["request_id"])

    @property
    def parents(self) -> tuple[ParentEdge, ...]:
        return tuple(
            ParentEdge(str(parent["artifact_id"]), str(parent["edge_type"]))
            for parent in self.semantic["parents"]
        )

    @property
    def role_exposure(self) -> frozenset[str]:
        return frozenset(str(role) for role in self.semantic["role_exposure"])

    @property
    def artifact_type(self) -> str:
        return str(self.semantic["artifact_type"])

    @property
    def registry_dependencies(self) -> tuple[ParentEdge, ...]:
        return tuple(
            ParentEdge(
                str(entry["authority_id"]),
                f"registry-membership:{entry['role']}",
            )
            for entry in self.semantic["registry_membership"]
            if entry["authority_id"] != "self"
        )


class ArtifactStore:
    def __init__(self, workspace: OwnedWorkspace) -> None:
        self.workspace = workspace
        self.root = workspace.root
        self.index_path = self.root / "index.sqlite"
        self.receipts_dir = self.root / "_receipts"
        self.publish_lock = self.root / ".publish.lock"
        for path in (self.index_path, self.receipts_dir, self.publish_lock):
            if path.is_symlink():
                raise ArtifactError(f"artifact store control path cannot be a symlink: {path.name}")
        self.receipts_dir.mkdir(exist_ok=True)
        self.publish_lock.touch(exist_ok=True)
        self._initialize_index()

    @classmethod
    def create(cls, root: str | Path) -> ArtifactStore:
        store = cls(OwnedWorkspace.create(root))
        store.rebuild_index()
        return store

    @classmethod
    def open(cls, root: str | Path) -> ArtifactStore:
        """Open a store only when it was explicitly initialized already."""
        return cls(OwnedWorkspace.open(root))

    def _connect(self) -> sqlite3.Connection:
        if self.index_path.is_symlink():
            raise ArtifactError("artifact index cannot be a symlink")
        connection = sqlite3.connect(self.index_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize_index(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL UNIQUE,
                    artifact_type TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    manifest_path TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS parents (
                    child_id TEXT NOT NULL,
                    parent_id TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    PRIMARY KEY(child_id, parent_id, edge_type)
                );
                CREATE TABLE IF NOT EXISTS receipts (
                    receipt_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    receipt_path TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _hex(identifier: str) -> str:
        if not ID_RE.fullmatch(identifier):
            raise ArtifactError(f"invalid artifact id {identifier!r}")
        return identifier.removeprefix("sha256:")

    def object_dir(self, artifact_id: str) -> Path:
        return self.root / self._hex(artifact_id)

    def manifest_path(self, artifact_id: str) -> Path:
        return self.object_dir(artifact_id) / "manifest.json"

    def payload_path(self, artifact_id: str) -> Path:
        return self.object_dir(artifact_id) / "payload.bin"

    def read_payload(
        self,
        artifact_id: str,
        *,
        capability: AccessCapability | None = None,
        authority: ProtectedAccessAuthority | None = None,
        semantic_hash: str | None = None,
    ) -> bytes:
        """Return payload bytes only after any protected ancestry is authorized.

        ``verify`` and index/trace operations remain capability-free because they expose
        manifests and integrity facts, not payload bytes.  The payload API itself is
        fail-closed for test/human-study ancestry unless the artifact is a code-owned
        aggregate/control type.
        """
        manifest = self.verify_metadata(artifact_id)
        if not self.payload_requires_authority(manifest):
            # Public control/aggregate payloads may have protected parents.  Verify
            # their own bytes and all unprotected metadata lineage without opening
            # any protected ancestor payload.
            self.verify(artifact_id)
            return self._validate_payload(manifest)
        manifest, payloads = self._verify_full_tree(
            artifact_id,
            capability=capability,
            authority=authority,
            reservation_semantic_hash=semantic_hash,
        )
        return payloads[manifest.artifact_id]

    def _read_manifest(self, artifact_id: str) -> ArtifactManifest:
        object_dir = self.object_dir(artifact_id)
        manifest_path = self.manifest_path(artifact_id)
        payload_path = self.payload_path(artifact_id)
        if object_dir.is_symlink() or manifest_path.is_symlink() or payload_path.is_symlink():
            raise ArtifactTampered(f"artifact {artifact_id}: symlinked object paths are forbidden")
        try:
            document = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactTampered(f"artifact {artifact_id}: manifest is missing or invalid") from exc
        if not isinstance(document, dict) or set(document) != {"schema", "artifact_id", "semantic"}:
            raise ArtifactTampered(f"artifact {artifact_id}: manifest shape is invalid")
        if document["schema"] != "dpo.artifact-manifest/v1" or document["artifact_id"] != artifact_id:
            raise ArtifactTampered(f"artifact {artifact_id}: manifest identity is invalid")
        semantic = document["semantic"]
        if not isinstance(semantic, dict) or semantic_hash(semantic) != artifact_id:
            raise ArtifactTampered(f"artifact {artifact_id}: artifact id does not match semantic manifest")
        if set(semantic) != SEMANTIC_FIELDS or semantic.get("schema") != "dpo.artifact-semantic/v1":
            raise ArtifactTampered(f"artifact {artifact_id}: semantic manifest shape is invalid")
        request = semantic.get("request")
        if not isinstance(request, dict) or semantic_hash(request) != semantic.get("request_id"):
            raise ArtifactTampered(f"artifact {artifact_id}: request identity is invalid")
        try:
            if set(request) != {
                "schema",
                "artifact_type",
                "parents",
                "contract_id",
                "model_pins",
                "prompt_pins",
                "parameters",
                "seed",
                "lock_id",
            }:
                raise ArtifactError("request shape is invalid")
            parent_values = request["parents"]
            if not isinstance(parent_values, list):
                raise ArtifactError("request parents are invalid")
            request_parents = tuple(
                ParentEdge(str(parent["artifact_id"]), str(parent["edge_type"]))
                for parent in parent_values
                if isinstance(parent, dict) and set(parent) == {"artifact_id", "edge_type"}
            )
            if len(request_parents) != len(parent_values):
                raise ArtifactError("request parents are invalid")
            for key in ("model_pins", "prompt_pins", "parameters"):
                if not isinstance(request[key], dict):
                    raise ArtifactError(f"request {key} is invalid")
            resolved_request = RequestSpec(
                artifact_type=str(request["artifact_type"]),
                parents=request_parents,
                contract_id=str(request["contract_id"]),
                model_pins=request["model_pins"],
                prompt_pins=request["prompt_pins"],
                parameters=request["parameters"],
                seed=request["seed"],
                lock_id=str(request["lock_id"]),
            )
        except (ArtifactError, KeyError, TypeError) as exc:
            raise ArtifactTampered(f"artifact {artifact_id}: request contract is invalid") from exc
        if (
            resolved_request.document() != request
            or semantic["artifact_type"] != resolved_request.artifact_type
            or semantic["parents"] != request["parents"]
        ):
            raise ArtifactTampered(f"artifact {artifact_id}: request lineage is inconsistent")
        if (
            not isinstance(semantic["payload_hash"], str)
            or not ID_RE.fullmatch(semantic["payload_hash"])
            or type(semantic["payload_size"]) is not int
            or semantic["payload_size"] < 0
            or type(semantic["row_count"]) is not int
            or semantic["row_count"] < 0
        ):
            raise ArtifactTampered(f"artifact {artifact_id}: payload metadata is invalid")
        return ArtifactManifest(artifact_id=artifact_id, semantic=semantic)

    @staticmethod
    def payload_requires_authority(manifest: ArtifactManifest) -> bool:
        """Whether opening this payload requires a fenced protected-role capability."""
        return bool(manifest.role_exposure & CAPABILITY_READ_ROLES) and (
            manifest.artifact_type not in PUBLIC_DERIVED_TYPES
        )

    def _metadata_tree(self, artifact_id: str) -> tuple[ArtifactManifest, dict[str, ArtifactManifest]]:
        verified: dict[str, ArtifactManifest] = {}
        manifest = self._verify_recursive_metadata(artifact_id, active=set(), verified=verified)
        return manifest, verified

    def verify_metadata(self, artifact_id: str) -> ArtifactManifest:
        """Verify the content-addressed manifest and recursive metadata without payload I/O."""
        return self._metadata_tree(artifact_id)[0]

    def verify(self, artifact_id: str) -> ArtifactManifest:
        """Verify metadata recursively and payloads that do not require protected authority.

        Protected payloads are deliberately manifest-only here.  Call ``verify_full`` or
        ``read_payload`` with a fenced role capability after the semantic reservation to
        hash and structurally validate protected bytes.
        """
        manifest, manifests = self._metadata_tree(artifact_id)
        for candidate in manifests.values():
            if not self.payload_requires_authority(candidate):
                self._validate_payload(candidate)
        return manifest

    def verify_full(
        self,
        artifact_id: str,
        *,
        capability: AccessCapability | None = None,
        authority: ProtectedAccessAuthority | None = None,
        semantic_hash: str | None = None,
    ) -> ArtifactManifest:
        """Verify every payload in a lineage, fencing before protected byte access."""
        return self._verify_full_tree(
            artifact_id,
            capability=capability,
            authority=authority,
            reservation_semantic_hash=semantic_hash,
        )[0]

    def _payload_checked(self, manifest: ArtifactManifest) -> bytes:
        artifact_id = manifest.artifact_id
        payload_path = self.payload_path(artifact_id)
        try:
            payload = payload_path.read_bytes()
        except OSError as exc:
            raise ArtifactTampered(f"artifact {artifact_id}: payload is missing") from exc
        if sha256_bytes(payload) != manifest.semantic["payload_hash"]:
            raise ArtifactTampered(f"artifact {artifact_id}: payload hash mismatch")
        if len(payload) != manifest.semantic["payload_size"]:
            raise ArtifactTampered(f"artifact {artifact_id}: payload size mismatch")
        return payload

    def _validate_payload(self, manifest: ArtifactManifest) -> bytes:
        payload = self._payload_checked(manifest)
        if manifest.artifact_type in REGISTRY_TYPES:
            registry_id, rows = self._registry_rows(manifest, payload)
            expected_membership = [
                {
                    "authority_id": "self",
                    "registry_id": registry_id,
                    "clip_id": clip_id,
                    "group_id": row["group_id"],
                    "role": row["role"],
                }
                for clip_id, row in sorted(rows.items())
            ]
            if manifest.semantic["registry_membership"] != expected_membership:
                raise ArtifactTampered(
                    f"artifact {manifest.artifact_id}: registry membership is not payload-derived"
                )
        return payload

    def _verify_full_tree(
        self,
        artifact_id: str,
        *,
        capability: AccessCapability | None,
        authority: ProtectedAccessAuthority | None,
        reservation_semantic_hash: str | None,
    ) -> tuple[ArtifactManifest, dict[str, bytes]]:
        manifest, manifests = self._metadata_tree(artifact_id)
        protected = [item for item in manifests.values() if self.payload_requires_authority(item)]
        if protected:
            roles = {role for item in protected for role in item.role_exposure & CAPABILITY_READ_ROLES}
            if len(roles) != 1:
                raise AccessDenied(
                    "mixed protected-role payloads must be consumed through role-specific shards"
                )
            role = next(iter(roles))
            if capability is None or authority is None or reservation_semantic_hash is None:
                raise AccessDenied(f"role {role!r} payload requires a fenced capability")
            expected_path = self.root / f"{role}.sqlite"
            if authority.path.resolve(strict=False) != expected_path.resolve(strict=False):
                raise AccessDenied(f"role {role!r} payload used the wrong access authority")
            # This validation is intentionally the last action before any payload is
            # opened.  Metadata parsing above only reads content-addressed manifests.
            authority.validate_and_mark_read(
                capability,
                semantic_hash=reservation_semantic_hash,
                role=role,
            )
        payloads = {item.artifact_id: self._validate_payload(item) for item in manifests.values()}
        return manifest, payloads

    @staticmethod
    def _validated_clip_row(value: object) -> dict[str, object]:
        if not isinstance(value, dict) or set(value) != LOCKED_CLIP_FIELDS:
            raise ArtifactTampered("clip registry row shape is invalid")
        row = dict(value)
        for key in ("clip_id", "source_video_id", "group_id"):
            if not isinstance(row[key], str) or not str(row[key]).strip():
                raise ArtifactTampered(f"clip registry row {key} is invalid")
        for key in ("media_hash",):
            if not isinstance(row[key], str) or not ID_RE.fullmatch(str(row[key])):
                raise ArtifactTampered(f"clip registry row {key} is invalid")
        derivatives = row["derivative_hashes"]
        if not isinstance(derivatives, list) or any(
            not isinstance(item, str) or not ID_RE.fullmatch(item) for item in derivatives
        ):
            raise ArtifactTampered("clip registry derivative hashes are invalid")
        if (
            type(row["start_ms"]) is not int
            or type(row["end_ms"]) is not int
            or int(row["start_ms"]) >= int(row["end_ms"])
        ):
            raise ArtifactTampered("clip registry interval is invalid")
        if row["role"] not in ALL_ROLES:
            raise ArtifactTampered("clip registry role is invalid")
        return row

    def _registry_rows(
        self, manifest: ArtifactManifest, payload: bytes
    ) -> tuple[str, dict[str, dict[str, object]]]:
        try:
            document = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ArtifactTampered("clip registry payload is invalid JSON") from exc
        parameters = manifest.semantic["request"].get("parameters")
        attributes = manifest.semantic.get("attributes")
        if not isinstance(parameters, dict) or not isinstance(attributes, dict):
            raise ArtifactTampered("clip registry request metadata is invalid")
        if manifest.artifact_type == "dpo.clip-registry-shard/v1":
            rows = [self._validated_clip_row(document)]
            registry_id = parameters.get("registry_id", attributes.get("registry_id"))
        elif manifest.artifact_type == "dpo.clip-registry/v1":
            if (
                not isinstance(document, dict)
                or set(document) != {"schema", "registry_id", "clips"}
                or document["schema"] != "dpo.clip-registry/v1"
                or not isinstance(document["clips"], list)
                or not document["clips"]
            ):
                raise ArtifactTampered("clip registry payload shape is invalid")
            rows = [self._validated_clip_row(row) for row in document["clips"]]
            registry_id = document["registry_id"]
            expected_registry = semantic_hash(
                {
                    "schema": "dpo.clip-registry/v1",
                    "contract_id": manifest.semantic["request"]["contract_id"],
                    "clips": rows,
                }
            )
            if registry_id != expected_registry:
                raise ArtifactTampered("clip registry id does not match its locked rows")
        else:
            raise ArtifactTampered("artifact is not a clip registry authority")
        if not isinstance(registry_id, str) or not ID_RE.fullmatch(registry_id):
            raise ArtifactTampered("clip registry id is invalid")
        by_id: dict[str, dict[str, object]] = {}
        for row in rows:
            clip_id = str(row["clip_id"])
            if clip_id in by_id:
                raise ArtifactTampered(f"clip registry has duplicate clip {clip_id!r}")
            by_id[clip_id] = row
        return registry_id, by_id

    def _registry_catalog(self) -> dict[str, dict[str, object]]:
        catalog: dict[str, dict[str, object]] = {}
        for artifact_type in sorted(REGISTRY_TYPES):
            for artifact_id in self.find_by_type(artifact_type):
                manifest = self.verify_metadata(artifact_id)
                for membership in manifest.semantic["registry_membership"]:
                    clip_id = str(membership["clip_id"])
                    entry = {
                        "authority_id": artifact_id,
                        "registry_id": membership["registry_id"],
                        "clip_id": clip_id,
                        "group_id": membership["group_id"],
                        "role": membership["role"],
                    }
                    previous = catalog.get(clip_id)
                    if previous is not None and any(
                        previous[key] != entry[key] for key in ("registry_id", "group_id", "role")
                    ):
                        raise ArtifactTampered(f"conflicting registry authority for clip {clip_id!r}")
                    if previous is None or artifact_type == "dpo.clip-registry/v1":
                        catalog[clip_id] = entry
        return catalog

    @staticmethod
    def _expected_protected_sources(
        membership: list[dict[str, object]], parents: list[ArtifactManifest]
    ) -> list[dict[str, str]]:
        sources = {
            (str(entry["authority_id"]), str(entry["role"]), "registry")
            for entry in membership
            if entry["role"] in PROTECTED_ROLES
        }
        for parent in parents:
            sources.update(
                (parent.artifact_id, role, "parent") for role in parent.role_exposure & PROTECTED_ROLES
            )
        return [
            {"artifact_id": artifact_id, "role": role, "source_type": source_type}
            for artifact_id, role, source_type in sorted(sources)
        ]

    def _verify_membership_metadata(
        self,
        manifest: ArtifactManifest,
        active: set[str],
        verified: dict[str, ArtifactManifest],
    ) -> list[dict[str, object]]:
        membership_value = manifest.semantic.get("registry_membership")
        if not isinstance(membership_value, list) or any(
            not isinstance(entry, dict)
            or set(entry) != {"authority_id", "registry_id", "clip_id", "group_id", "role"}
            for entry in membership_value
        ):
            raise ArtifactTampered(f"artifact {manifest.artifact_id}: registry membership is invalid")
        membership = [dict(entry) for entry in membership_value]
        seen_clips: set[str] = set()
        for entry in membership:
            authority_id = entry["authority_id"]
            clip_id = entry["clip_id"]
            if (
                not isinstance(authority_id, str)
                or not isinstance(entry["registry_id"], str)
                or not ID_RE.fullmatch(entry["registry_id"])
                or not isinstance(clip_id, str)
                or not clip_id.strip()
                or not isinstance(entry["group_id"], str)
                or not entry["group_id"].strip()
                or entry["role"] not in ALL_ROLES
                or clip_id in seen_clips
            ):
                raise ArtifactTampered(
                    f"artifact {manifest.artifact_id}: registry membership values are invalid"
                )
            seen_clips.add(clip_id)
        if manifest.artifact_type in REGISTRY_TYPES:
            if not membership or any(entry["authority_id"] != "self" for entry in membership):
                raise ArtifactTampered(
                    f"artifact {manifest.artifact_id}: registry authority metadata is invalid"
                )
            registry_ids = {str(entry["registry_id"]) for entry in membership}
            parameters = manifest.semantic["request"]["parameters"]
            attributes = manifest.semantic["attributes"]
            declared = parameters.get("registry_id", attributes.get("registry_id"))
            if len(registry_ids) != 1 or declared not in registry_ids:
                raise ArtifactTampered(
                    f"artifact {manifest.artifact_id}: registry identity metadata is inconsistent"
                )
            if manifest.artifact_type == "dpo.clip-registry-shard/v1" and len(membership) != 1:
                raise ArtifactTampered(
                    f"artifact {manifest.artifact_id}: registry shard must contain one clip"
                )
            return membership
        for entry in membership:
            authority_id = str(entry["authority_id"])
            if not ID_RE.fullmatch(authority_id):
                raise ArtifactTampered(f"artifact {manifest.artifact_id}: membership authority id is invalid")
            authority = self._verify_recursive_metadata(authority_id, active=active, verified=verified)
            if authority.artifact_type not in REGISTRY_TYPES:
                raise ArtifactTampered(
                    f"artifact {manifest.artifact_id}: membership authority is not a registry"
                )
            authoritative = next(
                (
                    item
                    for item in authority.semantic["registry_membership"]
                    if item["clip_id"] == entry["clip_id"]
                ),
                None,
            )
            expected_entry = (
                None if authoritative is None else {**dict(authoritative), "authority_id": authority_id}
            )
            if entry != expected_entry:
                raise ArtifactTampered(
                    f"artifact {manifest.artifact_id}: membership disagrees with registry authority"
                )
        return membership

    def _verify_recursive_metadata(
        self,
        artifact_id: str,
        *,
        active: set[str],
        verified: dict[str, ArtifactManifest],
    ) -> ArtifactManifest:
        if artifact_id in active:
            raise ArtifactTampered(f"artifact lineage cycle at {artifact_id}")
        if artifact_id in verified:
            return verified[artifact_id]
        active.add(artifact_id)
        try:
            manifest = self._read_manifest(artifact_id)
            parents = [
                self._verify_recursive_metadata(parent.artifact_id, active=active, verified=verified)
                for parent in manifest.parents
            ]
            membership = self._verify_membership_metadata(manifest, active, verified)
            local_clips = {str(entry["clip_id"]) for entry in membership}
            declared_clips = set(map(str, manifest.semantic.get("clips", [])))
            if manifest.artifact_type not in UNTRUSTED_SOURCE_TYPES and declared_clips != local_clips:
                raise ArtifactTampered(f"artifact {artifact_id}: clip membership is not registry-derived")
            expected_groups = sorted({str(entry["group_id"]) for entry in membership})
            if (
                manifest.artifact_type not in UNTRUSTED_SOURCE_TYPES
                and manifest.semantic.get("groups") != expected_groups
            ):
                raise ArtifactTampered(f"artifact {artifact_id}: group membership is not registry-derived")
            local_roles = {str(entry["role"]) for entry in membership}
            if manifest.semantic.get("local_role_exposure") != sorted(local_roles):
                raise ArtifactTampered(f"artifact {artifact_id}: local role exposure is invalid")
            expected_roles = local_roles | {role for parent in parents for role in parent.role_exposure}
            if manifest.semantic.get("role_exposure") != sorted(expected_roles):
                raise ArtifactTampered(f"artifact {artifact_id}: recursive role exposure is invalid")
            for name in ("fit", "selection", "test"):
                local_key = f"local_{name}_exposure"
                key = f"{name}_exposure"
                local = set(map(str, manifest.semantic.get(local_key, [])))
                if not local <= declared_clips:
                    raise ArtifactTampered(f"artifact {artifact_id}: {name} exposure is outside membership")
                inherited = {item for parent in parents for item in map(str, parent.semantic.get(key, []))}
                if manifest.semantic.get(key) != sorted(local | inherited):
                    raise ArtifactTampered(f"artifact {artifact_id}: recursive {name} exposure is invalid")
            expected_sources = self._expected_protected_sources(membership, parents)
            if manifest.semantic.get("protected_sources") != expected_sources:
                raise ArtifactTampered(f"artifact {artifact_id}: protected parent typing is invalid")
            verified[artifact_id] = manifest
            return manifest
        finally:
            active.remove(artifact_id)

    def _index_manifest(self, connection: sqlite3.Connection, manifest: ArtifactManifest) -> None:
        existing = connection.execute(
            "SELECT artifact_id FROM artifacts WHERE request_id=?", (manifest.request_id,)
        ).fetchone()
        if existing is not None and str(existing["artifact_id"]) != manifest.artifact_id:
            raise ArtifactConflict(
                f"request {manifest.request_id} has multiple content results (first-writer conflict)"
            )
        connection.execute(
            """INSERT OR IGNORE INTO artifacts
               (artifact_id, request_id, artifact_type, payload_hash, manifest_path)
               VALUES (?, ?, ?, ?, ?)""",
            (
                manifest.artifact_id,
                manifest.request_id,
                str(manifest.semantic["artifact_type"]),
                str(manifest.semantic["payload_hash"]),
                str(self.manifest_path(manifest.artifact_id).relative_to(self.root)),
            ),
        )
        for parent in manifest.parents:
            connection.execute(
                "INSERT OR IGNORE INTO parents(child_id, parent_id, edge_type) VALUES (?, ?, ?)",
                (manifest.artifact_id, parent.artifact_id, parent.edge_type),
            )
        for dependency in manifest.registry_dependencies:
            connection.execute(
                "INSERT OR IGNORE INTO parents(child_id, parent_id, edge_type) VALUES (?, ?, ?)",
                (manifest.artifact_id, dependency.artifact_id, dependency.edge_type),
            )

    def rebuild_index(self) -> None:
        manifests: list[ArtifactManifest] = []
        for path in sorted(self.root.iterdir(), key=lambda item: item.name):
            if not path.is_dir() or not re.fullmatch(r"[0-9a-f]{64}", path.name):
                continue
            manifests.append(self.verify(f"sha256:{path.name}"))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM parents")
            connection.execute("DELETE FROM receipts")
            connection.execute("DELETE FROM artifacts")
            for manifest in manifests:
                self._index_manifest(connection, manifest)
            for receipt_path in sorted(self.receipts_dir.glob("*.json")):
                try:
                    if receipt_path.is_symlink():
                        raise ArtifactTampered(f"execution receipt {receipt_path.name} cannot be a symlink")
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                    receipt_id = str(receipt["receipt_id"])
                    if sha256_bytes(canonical_bytes(receipt["receipt"])) != receipt_id:
                        raise ArtifactTampered(f"execution receipt {receipt_path.name} is corrupted")
                    connection.execute(
                        "INSERT INTO receipts VALUES (?, ?, ?, ?)",
                        (
                            receipt_id,
                            str(receipt["receipt"]["request_id"]),
                            str(receipt["receipt"]["artifact_id"]),
                            str(receipt_path.relative_to(self.root)),
                        ),
                    )
                except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    raise ArtifactTampered(f"execution receipt {receipt_path.name} is invalid") from exc

    def indexed_ids(self) -> set[str]:
        with self._connect() as connection:
            return {
                str(row["artifact_id"]) for row in connection.execute("SELECT artifact_id FROM artifacts")
            }

    def find_by_request_id(self, request_id: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT artifact_id FROM artifacts WHERE request_id=?", (request_id,)
            ).fetchone()
        return None if row is None else str(row["artifact_id"])

    def find_by_type(self, artifact_type: str) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT artifact_id FROM artifacts WHERE artifact_type=? ORDER BY artifact_id",
                (artifact_type,),
            ).fetchall()
        return [str(row["artifact_id"]) for row in rows]

    def publish(
        self,
        request: RequestSpec,
        payload: bytes,
        *,
        row_count: int = 0,
        clips: set[str] | None = None,
        groups: set[str] | None = None,
        role_exposure: set[str] | None = None,
        fit_exposure: set[str] | None = None,
        selection_exposure: set[str] | None = None,
        test_exposure: set[str] | None = None,
        attributes: Mapping[str, object] | None = None,
        purpose: str = "research",
        fault_hook: Callable[[str], None] | None = None,
    ) -> ArtifactManifest:
        if type(row_count) is not int or row_count < 0:
            raise ArtifactError("artifact row_count must be a non-negative integer")
        parent_manifests = [self.verify(parent.artifact_id) for parent in request.parents]
        attributes_document = dict(attributes or {})
        requested_clips = set(clips or ())
        requested_groups = set(groups or ())
        requested_roles = set(role_exposure or ())
        if request.artifact_type in REGISTRY_TYPES:
            provisional = ArtifactManifest(
                artifact_id="sha256:" + "0" * 64,
                semantic={
                    "artifact_type": request.artifact_type,
                    "request": request.document(),
                    "attributes": attributes_document,
                },
            )
            registry_id, rows = self._registry_rows(provisional, payload)
            registry_membership = [
                {
                    "authority_id": "self",
                    "registry_id": registry_id,
                    "clip_id": clip_id,
                    "group_id": row["group_id"],
                    "role": row["role"],
                }
                for clip_id, row in sorted(rows.items())
            ]
            if row_count != len(rows):
                raise ArtifactError("clip registry row_count must match its payload")
        elif request.artifact_type in UNTRUSTED_SOURCE_TYPES:
            registry_membership = []
        else:
            catalog = self._registry_catalog()
            missing = sorted(requested_clips - set(catalog))
            if missing:
                raise ArtifactError(f"artifact clip {missing[0]!r} has no registry membership authority")
            registry_membership = [dict(catalog[clip_id]) for clip_id in sorted(requested_clips)]
        derived_clips = {str(entry["clip_id"]) for entry in registry_membership}
        derived_groups = {str(entry["group_id"]) for entry in registry_membership}
        local_roles = {str(entry["role"]) for entry in registry_membership}
        if request.artifact_type not in UNTRUSTED_SOURCE_TYPES:
            if requested_clips != derived_clips:
                raise ArtifactError("artifact clip membership must be registry-derived")
            if requested_groups and requested_groups != derived_groups:
                raise ArtifactError("artifact group assertion disagrees with registry membership")
            if requested_roles and requested_roles != local_roles:
                raise ArtifactError("artifact role assertion disagrees with registry membership")
        elif requested_roles:
            raise ArtifactError("untrusted source artifacts cannot assert role exposure")
        inherited_roles = local_roles | {role for parent in parent_manifests for role in parent.role_exposure}
        if purpose == "training" and inherited_roles & PROTECTED_ROLES:
            raise ProtectedExposure(
                f"training artifact has protected exposure {sorted(inherited_roles & PROTECTED_ROLES)}"
            )
        local_exposures = {
            "fit": set(fit_exposure or ()),
            "selection": set(selection_exposure or ()),
            "test": set(test_exposure or ()),
        }
        for name, exposure in local_exposures.items():
            if not exposure <= requested_clips:
                raise ArtifactError(f"{name} exposure must be a subset of registry clip membership")
        recursive_exposures = {
            name: exposure
            | {
                item
                for parent in parent_manifests
                for item in map(str, parent.semantic.get(f"{name}_exposure", []))
            }
            for name, exposure in local_exposures.items()
        }
        semantic: dict[str, object] = {
            "schema": "dpo.artifact-semantic/v1",
            "artifact_type": request.artifact_type,
            "request_id": request.request_id,
            "request": request.document(),
            "parents": [parent.document() for parent in request.parents],
            "payload_hash": sha256_bytes(payload),
            "payload_size": len(payload),
            "row_count": row_count,
            "clips": sorted(requested_clips),
            "groups": sorted(
                requested_groups if request.artifact_type in UNTRUSTED_SOURCE_TYPES else derived_groups
            ),
            "registry_membership": registry_membership,
            "local_role_exposure": sorted(local_roles),
            "role_exposure": sorted(inherited_roles),
            "local_fit_exposure": sorted(local_exposures["fit"]),
            "local_selection_exposure": sorted(local_exposures["selection"]),
            "local_test_exposure": sorted(local_exposures["test"]),
            "fit_exposure": sorted(recursive_exposures["fit"]),
            "selection_exposure": sorted(recursive_exposures["selection"]),
            "test_exposure": sorted(recursive_exposures["test"]),
            "protected_sources": self._expected_protected_sources(registry_membership, parent_manifests),
            "attributes": attributes_document,
        }
        artifact_id = semantic_hash(semantic)
        document = {
            "schema": "dpo.artifact-manifest/v1",
            "artifact_id": artifact_id,
            "semantic": semantic,
        }
        if self.publish_lock.is_symlink():
            raise ArtifactError("artifact publication lock cannot be a symlink")
        with self.publish_lock.open("rb") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                with self._connect() as connection:
                    existing = connection.execute(
                        "SELECT artifact_id FROM artifacts WHERE request_id=?", (request.request_id,)
                    ).fetchone()
                if existing is not None:
                    existing_id = str(existing["artifact_id"])
                    if existing_id != artifact_id:
                        raise ArtifactConflict(
                            f"request {request.request_id} already resolved to different content"
                        )
                    manifest = self.verify(existing_id)
                    self._record_default_receipt(manifest, request)
                    return manifest
                destination = self.object_dir(artifact_id)
                if destination.exists():
                    manifest = self.verify(artifact_id)
                    if manifest.request_id != request.request_id:
                        raise ArtifactConflict("content collision has a different request identity")
                else:
                    stage = self.workspace.claim_child(f".stage-{uuid.uuid4().hex}")
                    try:
                        (stage / "payload.bin").write_bytes(payload)
                        if fault_hook is not None:
                            fault_hook("after-payload-write")
                        (stage / "manifest.json").write_bytes(canonical_bytes(document) + b"\n")
                        if fault_hook is not None:
                            fault_hook("after-manifest-write")
                        for name in ("payload.bin", "manifest.json"):
                            descriptor = os.open(stage / name, os.O_RDONLY)
                            try:
                                os.fsync(descriptor)
                            finally:
                                os.close(descriptor)
                        if fault_hook is not None:
                            fault_hook("before-rename")
                        os.rename(stage, destination)
                        directory_descriptor = os.open(self.root, os.O_RDONLY)
                        try:
                            os.fsync(directory_descriptor)
                        finally:
                            os.close(directory_descriptor)
                        if fault_hook is not None:
                            fault_hook("after-rename-before-index")
                    finally:
                        if stage.exists():
                            self.workspace.safe_delete(stage)
                manifest = self.verify(artifact_id)
                with self._connect() as connection:
                    self._index_manifest(connection, manifest)
                self._record_default_receipt(manifest, request)
                return manifest
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _package_source_root() -> Path:
        package_paths = list(getattr(dpo, "__path__", ()))
        if not package_paths:
            raise ArtifactError("dpo package source path is unavailable")
        raw_root = Path(str(package_paths[0]))
        if raw_root.is_symlink():
            raise ArtifactError("dpo package source path is unsafe")
        root = raw_root.resolve()
        if not root.is_dir() or root.is_symlink():
            raise ArtifactError("dpo package source path is unsafe")
        return root

    @staticmethod
    def _package_source_hash() -> str:
        root = ArtifactStore._package_source_root()
        digest_input = b"dpo-package-source-v1\0"
        for path in sorted(root.rglob("*.py")):
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(root)
            except ValueError as exc:
                raise ArtifactError("dpo package source escaped its root") from exc
            if path.is_symlink() or not path.is_file():
                raise ArtifactError("dpo package source contains unsafe paths")
            digest_input += str(relative).encode() + b"\0" + path.read_bytes() + b"\0"
        return sha256_bytes(digest_input)

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _git_facts() -> tuple[str, str]:
        # Source identity is a per-process constant by design: every receipt in
        # one run must carry the same identity, and recomputing it means four
        # git subprocesses plus reading every untracked file's bytes per publish.
        package_root = ArtifactStore._package_source_root()
        fallback = ("unavailable", ArtifactStore._package_source_hash())

        def run_git(args: list[str], *, cwd: Path, text: bool = False) -> subprocess.CompletedProcess[Any]:
            try:
                return subprocess.run(
                    args,
                    cwd=cwd,
                    check=False,
                    capture_output=True,
                    text=text,
                )
            except (OSError, subprocess.SubprocessError):
                return subprocess.CompletedProcess(args=args, returncode=1, stdout="" if text else b"")

        top_level_result = run_git(["git", "rev-parse", "--show-toplevel"], cwd=package_root, text=True)
        if top_level_result.returncode != 0 or not isinstance(top_level_result.stdout, str):
            return fallback
        root = Path(top_level_result.stdout.strip()).resolve()
        if not root.is_dir():
            return fallback
        try:
            package_root.relative_to(root)
        except ValueError:
            return fallback
        commit_result = run_git(["git", "rev-parse", "HEAD"], cwd=root)
        diff_result = run_git(["git", "diff", "--binary", "HEAD"], cwd=root)
        untracked_result = run_git(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=root,
        )
        if commit_result.returncode != 0 or diff_result.returncode != 0 or untracked_result.returncode != 0:
            return fallback
        if (
            not isinstance(commit_result.stdout, bytes)
            or not isinstance(diff_result.stdout, bytes)
            or not isinstance(untracked_result.stdout, bytes)
        ):
            return fallback
        commit = commit_result.stdout.strip()
        diff = diff_result.stdout
        untracked = [name for name in untracked_result.stdout.split(b"\0") if name]
        for relative_bytes in sorted(untracked):
            try:
                relative = relative_bytes.decode("utf-8")
            except UnicodeDecodeError:
                return fallback
            raw_path = root / relative
            if raw_path.is_symlink():
                continue
            path = raw_path.resolve()
            try:
                path.relative_to(root)
            except ValueError:
                continue
            if path.is_file() and not path.is_symlink():
                content = path.read_bytes()
                diff += (
                    b"untracked\0"
                    + len(relative_bytes).to_bytes(8, "big")
                    + relative_bytes
                    + len(content).to_bytes(8, "big")
                    + content
                )
        return commit.decode() or "unavailable", sha256_bytes(diff)

    @classmethod
    def source_facts(cls) -> dict[str, object]:
        commit, dirty_patch_hash = cls._git_facts()
        return {
            "schema": "dpo.source-facts/v1",
            "git_commit": commit,
            "dirty_patch_hash": dirty_patch_hash,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
        }

    @classmethod
    def source_identity(cls) -> str:
        return semantic_hash(cls.source_facts())

    def _record_default_receipt(self, artifact: ArtifactManifest, request: RequestSpec) -> None:
        source_facts = self.source_facts()
        self.record_receipt(
            artifact,
            {
                **source_facts,
                "source_identity": semantic_hash(source_facts),
                "command": " ".join(sys.argv),
                "package_lock_hash": request.lock_id,
                "hardware": platform.platform(),
                "gpu_uuid": os.environ.get("DPO_GPU_UUID", "not-allocated"),
                "world_size": os.environ.get("WORLD_SIZE", "1"),
                "pid": os.getpid(),
            },
        )

    def record_receipt(self, artifact: ArtifactManifest, metadata: Mapping[str, object]) -> str:
        receipt = {
            "schema": "dpo.execution-receipt/v1",
            "request_id": artifact.request_id,
            "artifact_id": artifact.artifact_id,
            "metadata": dict(metadata),
        }
        receipt_id = sha256_bytes(canonical_bytes(receipt))
        document = {"receipt_id": receipt_id, "receipt": receipt}
        path = self.receipts_dir / f"{self._hex(receipt_id)}.json"
        if self.receipts_dir.is_symlink() or path.is_symlink():
            raise ArtifactError("execution receipt path cannot be a symlink")
        atomic_write_bytes(path, canonical_bytes(document) + b"\n")
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO receipts VALUES (?, ?, ?, ?)",
                (
                    receipt_id,
                    artifact.request_id,
                    artifact.artifact_id,
                    str(path.relative_to(self.root)),
                ),
            )
        return receipt_id

    def trace(self, artifact_id: str) -> list[ArtifactManifest]:
        ordered: list[ArtifactManifest] = []
        seen: set[str] = set()
        active: set[str] = set()

        def visit(identifier: str) -> None:
            if identifier in active:
                raise ArtifactTampered(f"artifact lineage cycle at {identifier}")
            if identifier in seen:
                return
            active.add(identifier)
            manifest = self.verify(identifier)
            ordered.append(manifest)
            for dependency in (*manifest.parents, *manifest.registry_dependencies):
                visit(dependency.artifact_id)
            active.remove(identifier)
            seen.add(identifier)

        visit(artifact_id)
        return ordered

    def gc(self, lock_reachable_ids: set[str], *, execute: bool = False) -> list[str]:
        if execute:
            if self.publish_lock.is_symlink():
                raise ArtifactError("artifact publication lock cannot be a symlink")
            with self.publish_lock.open("rb") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    self.rebuild_index()
                    return self._gc_locked(lock_reachable_ids, execute=True)
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        return self._gc_locked(lock_reachable_ids, execute=False)

    def _gc_locked(self, lock_reachable_ids: set[str], *, execute: bool) -> list[str]:
        authoritative_roots = {
            artifact_id for artifact_type in GC_ROOT_TYPES for artifact_id in self.find_by_type(artifact_type)
        }
        if execute and lock_reachable_ids != authoritative_roots:
            raise ArtifactError(
                "executable GC requires the complete set of code-owned authoritative lock roots"
            )
        reachable: set[str] = set()
        for artifact_id in lock_reachable_ids:
            reachable.update(manifest.artifact_id for manifest in self.trace(artifact_id))
        candidates = sorted(self.indexed_ids() - reachable)
        if execute:
            for artifact_id in candidates:
                self.verify(artifact_id)
                self.workspace.safe_delete(self.object_dir(artifact_id))
            self.rebuild_index()
        return candidates
