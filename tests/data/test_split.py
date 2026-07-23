"""Immutable group-level split tests (PRD sections 5 and 32.2)."""

from __future__ import annotations

import pytest

from dpo.contracts.study_contract import SPLITS, StudyContract
from dpo.core.identity import sha256_bytes
from dpo.data.split import (
    ClipInput,
    SplitError,
    assign_splits,
    parse_split_manifest,
    registry_rows,
)

FRACTIONS = {"train": 0.55, "validation": 0.15, "test": 0.15, "study": 0.15}


def _clips(count: int = 24, *, per_source: int = 2) -> list[ClipInput]:
    return [
        ClipInput(
            clip_id=f"clip-{index:03d}",
            source_video_id=f"src-{index // per_source:02d}",
            media_hash=sha256_bytes(f"media-{index}".encode()),
            start_ms=0,
            end_ms=6000,
            derivative_hashes=(),
        )
        for index in range(count)
    ]


def test_assignment_is_deterministic_and_group_atomic() -> None:
    clips = _clips()
    first = assign_splits(clips, seed=7, fractions=FRACTIONS)
    second = assign_splits(list(reversed(clips)), seed=7, fractions=FRACTIONS)
    assert first.signed_document() == second.signed_document()
    for clip in clips:
        sibling = next(
            other
            for other in clips
            if other.source_video_id == clip.source_video_id and other.clip_id != clip.clip_id
        )
        assert first.role_of(clip.clip_id) == first.role_of(sibling.clip_id)


def test_seed_changes_the_assignment_and_the_hash() -> None:
    clips = _clips()
    first = assign_splits(clips, seed=7, fractions=FRACTIONS)
    second = assign_splits(clips, seed=8, fractions=FRACTIONS)
    assert first.sha256 != second.sha256


def test_link_group_merges_across_sources() -> None:
    clips = [
        ClipInput(
            clip_id="clip-a",
            source_video_id="src-a",
            media_hash=sha256_bytes(b"a"),
            start_ms=0,
            end_ms=5000,
            derivative_hashes=(),
            link_group="near-dup-1",
        ),
        ClipInput(
            clip_id="clip-b",
            source_video_id="src-b",
            media_hash=sha256_bytes(b"b"),
            start_ms=0,
            end_ms=5000,
            derivative_hashes=(),
            link_group="near-dup-1",
        ),
        *_clips(20),
    ]
    manifest = assign_splits(clips, seed=3, fractions=FRACTIONS)
    assert manifest.groups["clip-a"] == manifest.groups["clip-b"]
    assert manifest.role_of("clip-a") == manifest.role_of("clip-b")


def test_manifest_roundtrip_and_immutability() -> None:
    manifest = assign_splits(_clips(), seed=7, fractions=FRACTIONS)
    parsed = parse_split_manifest(manifest.signed_document())
    assert parsed.sha256 == manifest.sha256
    tampered = manifest.signed_document()
    train = tampered["train"]
    assert isinstance(train, list)
    validation = tampered["validation"]
    assert isinstance(validation, list)
    validation.append(train.pop())
    with pytest.raises(SplitError, match="immutable"):
        parse_split_manifest(tampered)


def test_asserted_role_mismatch_is_rejected() -> None:
    clips = _clips()
    manifest = assign_splits(clips, seed=7, fractions=FRACTIONS)
    victim = clips[0]
    actual = manifest.role_of(victim.clip_id)
    wrong = next(split for split in SPLITS if split != actual)
    clips[0] = ClipInput(
        clip_id=victim.clip_id,
        source_video_id=victim.source_video_id,
        media_hash=victim.media_hash,
        start_ms=victim.start_ms,
        end_ms=victim.end_ms,
        derivative_hashes=victim.derivative_hashes,
        asserted_role=wrong,
    )
    with pytest.raises(SplitError, match="asserted"):
        assign_splits(clips, seed=7, fractions=FRACTIONS)


def test_registry_rows_lock_role_and_group(contract: StudyContract) -> None:
    del contract
    clips = _clips()
    manifest = assign_splits(clips, seed=7, fractions=FRACTIONS)
    rows = registry_rows(manifest, clips)
    assert len(rows) == len(clips)
    by_id = {str(row["clip_id"]): row for row in rows}
    for clip in clips:
        row = by_id[clip.clip_id]
        assert row["role"] == manifest.role_of(clip.clip_id)
        assert row["group_id"] == manifest.groups[clip.clip_id]
        assert row["source_video_id"] == clip.source_video_id
