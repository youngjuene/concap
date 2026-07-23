"""Group-aware immutable splits, computed before any construction stage.

All clips from one source video stay in one split; near-duplicate or
continuity-linked clips (asserted by an external detector, with provenance)
merge groups across sources. Assignment is a pure function of the corpus, the
contract seed, and the fractions, so the manifest regenerates bit-identically
and any drift changes its hash.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from dpo.contracts.study_contract import SPLITS
from dpo.core.identity import semantic_hash

HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
MANIFEST_VERSION = "1.0"


class SplitError(ValueError):
    """Raised when the corpus or a split manifest violates the grouping rules."""


@dataclass(frozen=True)
class ClipInput:
    """One corpus clip as ingested, before any role exists."""

    clip_id: str
    source_video_id: str
    media_hash: str
    start_ms: int
    end_ms: int
    derivative_hashes: tuple[str, ...]
    link_group: str | None = None
    asserted_role: str | None = None

    def __post_init__(self) -> None:
        if not self.clip_id.strip() or not self.source_video_id.strip():
            raise SplitError("clip requires clip_id and source_video_id")
        if not HASH_RE.fullmatch(self.media_hash):
            raise SplitError(f"clip {self.clip_id!r}: media_hash must be a sha256 hash")
        for derivative in self.derivative_hashes:
            if not HASH_RE.fullmatch(derivative):
                raise SplitError(f"clip {self.clip_id!r}: derivative hashes must be sha256 hashes")
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise SplitError(f"clip {self.clip_id!r}: interval must satisfy 0 <= start < end")
        if self.asserted_role is not None and self.asserted_role not in SPLITS:
            raise SplitError(f"clip {self.clip_id!r}: asserted role must be one of {sorted(SPLITS)}")

    def document(self) -> dict[str, object]:
        """The one serialized clip-row shape every producer publishes."""
        return {
            "clip_id": self.clip_id,
            "source_video_id": self.source_video_id,
            "media_hash": self.media_hash,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "derivative_hashes": list(self.derivative_hashes),
            "link_group": self.link_group,
            "asserted_role": self.asserted_role,
        }

    @classmethod
    def from_document(cls, value: Mapping[str, object]) -> ClipInput:
        derivatives = value.get("derivative_hashes", [])
        if not isinstance(derivatives, list):
            raise SplitError("clip row derivative_hashes must be an array")
        link_group = value.get("link_group")
        asserted_role = value.get("asserted_role")
        return cls(
            clip_id=str(value.get("clip_id", "")),
            source_video_id=str(value.get("source_video_id", "")),
            media_hash=str(value.get("media_hash", "")),
            start_ms=int(str(value.get("start_ms", -1))),
            end_ms=int(str(value.get("end_ms", -1))),
            derivative_hashes=tuple(str(item) for item in derivatives),
            link_group=None if link_group is None else str(link_group),
            asserted_role=None if asserted_role is None else str(asserted_role),
        )


@dataclass(frozen=True)
class SplitManifest:
    manifest_version: str
    group_key: str
    seed: int
    assignments: Mapping[str, tuple[str, ...]]  # split -> sorted clip ids
    groups: Mapping[str, str]  # clip id -> group id

    def role_of(self, clip_id: str) -> str:
        for split in SPLITS:
            if clip_id in set(self.assignments[split]):
                return split
        raise SplitError(f"clip {clip_id!r} has no split assignment")

    def document(self) -> dict[str, object]:
        body: dict[str, object] = {
            "manifest_version": self.manifest_version,
            "group_key": self.group_key,
            "seed": self.seed,
            "groups": dict(sorted(self.groups.items())),
        }
        for split in SPLITS:
            body[split] = list(self.assignments[split])
        return body

    @property
    def sha256(self) -> str:
        return semantic_hash(self.document())

    def signed_document(self) -> dict[str, object]:
        return {**self.document(), "sha256": self.sha256}


def _union_groups(clips: Sequence[ClipInput]) -> dict[str, str]:
    """Union-find over source ids and link labels; returns clip -> group id."""
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        root = node
        while parent.setdefault(root, root) != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for clip in clips:
        source_node = f"source::{clip.source_video_id}"
        find(source_node)
        if clip.link_group is not None:
            union(source_node, f"link::{clip.link_group}")
    members: dict[str, list[str]] = {}
    for clip in clips:
        root = find(f"source::{clip.source_video_id}")
        members.setdefault(root, []).append(clip.source_video_id)
    group_ids = {
        root: "grp-" + semantic_hash({"members": sorted(set(sources))}).removeprefix("sha256:")[:12]
        for root, sources in members.items()
    }
    return {clip.clip_id: group_ids[find(f"source::{clip.source_video_id}")] for clip in clips}


def assign_splits(
    clips: Sequence[ClipInput],
    *,
    seed: int,
    fractions: Mapping[str, float],
) -> SplitManifest:
    """Deterministic group-atomic assignment approximating the contract fractions."""
    if not clips:
        raise SplitError("cannot split an empty corpus")
    if set(fractions) != set(SPLITS):
        raise SplitError(f"split fractions must cover exactly {sorted(SPLITS)}")
    seen: set[str] = set()
    for clip in clips:
        if clip.clip_id in seen:
            raise SplitError(f"duplicate clip id {clip.clip_id!r}")
        seen.add(clip.clip_id)
    groups = _union_groups(clips)
    clips_by_group: dict[str, list[ClipInput]] = {}
    for clip in clips:
        clips_by_group.setdefault(groups[clip.clip_id], []).append(clip)
    total = len(clips)
    targets = {split: fractions[split] * total for split in SPLITS}
    assigned: dict[str, float] = dict.fromkeys(SPLITS, 0.0)
    assignment: dict[str, list[str]] = {split: [] for split in SPLITS}
    ordered_groups = sorted(
        clips_by_group,
        key=lambda group_id: semantic_hash({"seed": seed, "group": group_id}),
    )
    for group_id in ordered_groups:
        group_clips = clips_by_group[group_id]
        deficits = {split: targets[split] - assigned[split] for split in SPLITS}
        best = max(SPLITS, key=lambda split: (deficits[split], -SPLITS.index(split)))
        assignment[best].extend(sorted(clip.clip_id for clip in group_clips))
        assigned[best] += len(group_clips)
    empty = [split for split in SPLITS if not assignment[split]]
    if empty:
        raise SplitError(
            f"split {empty[0]!r} received no groups; the corpus is too small for these fractions"
        )
    manifest = SplitManifest(
        manifest_version=MANIFEST_VERSION,
        group_key="source_video_id",
        seed=seed,
        assignments={split: tuple(sorted(assignment[split])) for split in SPLITS},
        groups=groups,
    )
    for clip in clips:
        if clip.asserted_role is not None and manifest.role_of(clip.clip_id) != clip.asserted_role:
            raise SplitError(
                f"clip {clip.clip_id!r} asserts role {clip.asserted_role!r} but the seeded"
                f" assignment is {manifest.role_of(clip.clip_id)!r}; asserted roles exist only"
                " to catch drift and can never override the computed manifest"
            )
    return manifest


def registry_rows(manifest: SplitManifest, clips: Sequence[ClipInput]) -> list[dict[str, object]]:
    """Rows for the artifact-store clip registry, locking role and group per clip."""
    by_id = {clip.clip_id: clip for clip in clips}
    missing = sorted(set(manifest.groups) - set(by_id))
    if missing:
        raise SplitError(f"manifest references unknown clip {missing[0]!r}")
    rows = []
    for split in SPLITS:
        for clip_id in manifest.assignments[split]:
            clip = by_id[clip_id]
            rows.append(
                {
                    "clip_id": clip.clip_id,
                    "source_video_id": clip.source_video_id,
                    "media_hash": clip.media_hash,
                    "start_ms": clip.start_ms,
                    "end_ms": clip.end_ms,
                    "group_id": manifest.groups[clip.clip_id],
                    "derivative_hashes": list(clip.derivative_hashes),
                    "role": split,
                }
            )
    return sorted(rows, key=lambda row: str(row["clip_id"]))


def parse_split_manifest(payload: bytes | str | Mapping[str, object]) -> SplitManifest:
    if isinstance(payload, (bytes, str)):
        try:
            document = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise SplitError("split manifest payload is not valid JSON") from exc
    else:
        document = dict(payload)
    if not isinstance(document, dict) or set(document) != {
        "manifest_version",
        "group_key",
        "seed",
        "groups",
        *SPLITS,
        "sha256",
    }:
        raise SplitError("split manifest payload has an unexpected field set")
    groups_value = document["groups"]
    if not isinstance(groups_value, Mapping):
        raise SplitError("split manifest groups must be an object")
    seed = document["seed"]
    if type(seed) is not int:
        raise SplitError("split manifest seed must be an integer")
    assignments = {}
    for split in SPLITS:
        value = document[split]
        if not isinstance(value, list):
            raise SplitError(f"split manifest {split} must be an array")
        assignments[split] = tuple(str(item) for item in value)
    manifest = SplitManifest(
        manifest_version=str(document["manifest_version"]),
        group_key=str(document["group_key"]),
        seed=int(seed),
        assignments=assignments,
        groups={str(key): str(value) for key, value in groups_value.items()},
    )
    if manifest.manifest_version != MANIFEST_VERSION or manifest.group_key != "source_video_id":
        raise SplitError("split manifest version/group_key are not supported")
    if document["sha256"] != manifest.sha256:
        raise SplitError("split manifest sha256 does not match its content; the manifest is immutable")
    all_ids = [clip_id for split in SPLITS for clip_id in assignments[split]]
    if len(all_ids) != len(set(all_ids)):
        raise SplitError("split manifest assigns a clip to more than one split")
    if set(all_ids) != set(manifest.groups):
        raise SplitError("split manifest groups and assignments disagree")
    return manifest
