"""Leakage-rule tests (PRD sections 9.7 and 32.2)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from dpo.candidates.candidate_records import CandidateError
from dpo.candidates.freeze import assert_pool_unchanged, freeze_pool, parse_frozen_pool
from dpo.data.leakage_audit import (
    LeakageError,
    audit_text_leakage,
    run_leakage_audit,
)
from tests.conftest import PreferenceWorld


def test_clean_world_passes_the_audit(world: PreferenceWorld) -> None:
    report = run_leakage_audit(manifest=world.manifest, clips=world.clips, pool=world.pool, enforce=True)
    assert report.passed


def test_near_duplicate_text_across_splits_is_flagged(world: PreferenceWorld) -> None:
    validation_clip = world.manifest.assignments["validation"][0]
    train_candidate = world.pool.candidates[0]
    duplicate = replace(
        train_candidate,
        candidate_id="cand-leaked00001",
        clip_id=validation_clip,
    )
    leaky_pool = freeze_pool(
        dataset_version="leaky/v1",
        track=world.pool.track,
        evidence_audit_version="audit/v1",
        candidates=(*world.pool.candidates, duplicate),
        pairs=(),
    )
    violations = audit_text_leakage(leaky_pool, world.manifest)
    assert violations
    with pytest.raises(LeakageError):
        run_leakage_audit(manifest=world.manifest, clips=world.clips, pool=leaky_pool, enforce=True)


def test_candidate_hashes_are_stable_and_drift_is_detected(world: PreferenceWorld) -> None:
    document = world.pool.document()
    parsed = parse_frozen_pool(document)
    assert parsed.pool_hash == world.pool.pool_hash
    assert_pool_unchanged(parsed, expected_pool_hash=world.pool.pool_hash)
    candidates_value = document["candidates"]
    assert isinstance(candidates_value, list)
    first = dict(candidates_value[0])
    first["text"] = first["text"] + " tampered"
    candidates_value[0] = first
    with pytest.raises(CandidateError, match="hash"):
        parse_frozen_pool(document)


def test_sft_rows_from_a_protected_split_are_flagged(world: PreferenceWorld) -> None:
    from dpo.data.derive_sft import SftExample

    validation_clip = world.manifest.assignments["validation"][0]
    rows = (
        SftExample(
            example_id="sft-leak",
            clip_id=validation_clip,
            track="visual",
            candidate_id="cand-whatever",
            completion="A caption endorsed on the validation split.",
            consensus_weight=1.0,
        ),
    )
    report = run_leakage_audit(manifest=world.manifest, clips=world.clips, sft_rows=rows)
    assert not report.passed
    assert any("validation" in violation for violation in report.violations)
