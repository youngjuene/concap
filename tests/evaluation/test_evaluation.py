"""Evaluation tests: compliance, preference accuracy, factuality, study export."""

from __future__ import annotations

import math

import pytest

from dpo.contracts.study_contract import EXPERIMENT_IDS, StudyContract
from dpo.core.identity import sha256_bytes
from dpo.evaluation.caption_generation import training_candidate_reuse_rate
from dpo.evaluation.compliance import GeneratedCaption, compute_compliance
from dpo.evaluation.factuality import evaluate_caption_claims, summarize_factuality
from dpo.evaluation.human_study_export import (
    StudyExportError,
    build_study_export,
)
from dpo.evaluation.preference_accuracy import ScoredPair, evaluate_preferences
from tests.conftest import PreferenceWorld


def _scored(margin: float, *, difficulty: float = 0.2, agreement: float = 1.0) -> ScoredPair:
    return ScoredPair(
        pair_id=f"pair-{margin}",
        clip_id="clip-1",
        track="visual",
        policy_chosen_logp=margin,
        policy_rejected_logp=0.0,
        ref_chosen_logp=0.0,
        ref_rejected_logp=0.0,
        difficulty=difficulty,
        agreement=agreement,
    )


def test_preference_report_accuracy_and_logloss() -> None:
    report = evaluate_preferences([_scored(2.0), _scored(-1.0)], beta=1.0)
    assert report.accuracy == 0.5
    expected = (-math.log(1 / (1 + math.exp(-2.0))) - math.log(1 / (1 + math.exp(1.0)))) / 2
    assert report.log_loss == pytest.approx(expected, rel=1e-6)
    assert "easy" in report.accuracy_by_difficulty


def test_compliance_metrics_over_generated_captions(contract: StudyContract) -> None:
    captions = [
        GeneratedCaption("clip-1", "A cyclist crosses the junction toward the market."),
        GeneratedCaption("clip-2", "A cyclist crosses the junction toward the market."),
        GeneratedCaption("clip-3", ""),
        GeneratedCaption("clip-4", "Loud music plays while a cyclist rides away quickly."),
    ]
    metrics = compute_compliance(captions, contract.tracks["visual"])
    assert metrics.caption_count == 4
    assert metrics.empty_rate == pytest.approx(0.25)
    assert metrics.duplicate_output_rate == pytest.approx(0.25)
    assert metrics.modality_violation_rate == pytest.approx(0.25)
    assert metrics.grammar == "blocked_pending_external_operation"


def test_factuality_matches_full_claim_forms_only(world: PreferenceWorld) -> None:
    clip_id = world.pool.candidates[0].clip_id
    ledger = world.ledgers[clip_id]
    supported = ledger.claims[0].canonical_form
    verdict = evaluate_caption_claims("caption-1", f"The scene shows {supported} in daylight.", ledger)
    assert verdict.supported_mention_count == 1
    assert verdict.contradicted_mention_count == 0
    report = summarize_factuality([verdict], {clip_id: ledger})
    assert report.caption_count == 1
    assert 0.0 < report.ledger_audit_coverage <= 1.0


def test_generated_captions_must_not_reuse_training_candidates(
    world: PreferenceWorld,
) -> None:
    reused = GeneratedCaption(clip_id=world.pool.candidates[0].clip_id, text=world.pool.candidates[0].text)
    fresh = GeneratedCaption(clip_id="clip-000", text="An entirely fresh generated caption.")
    assert training_candidate_reuse_rate([reused, fresh], world.pool) == pytest.approx(0.5)


def _study_captions() -> dict[str, dict[str, str]]:
    clips = [f"study-clip-{index}" for index in range(4)]
    return {
        experiment_id: {
            clip: f"caption {index} for {clip} by policy {sum(ord(char) for char in experiment_id)}"
            for index, clip in enumerate(clips)
        }
        for experiment_id in EXPERIMENT_IDS
    }


def test_study_export_covers_all_36_pairs_blind_and_balanced() -> None:
    captions = _study_captions()
    media = {f"study-clip-{index}": sha256_bytes(f"media-{index}".encode()) for index in range(4)}
    export = build_study_export(track="visual", captions=captions, media_hashes=media, exposures_per_pair=2)
    assert len(export.tasks) == 36 * 2
    pair_counts: dict[tuple[str, str], int] = {}
    position_counts: dict[str, int] = {}
    for task in export.tasks:
        identity = export.randomization[task.task_id]
        key = tuple(sorted((identity["model_a"], identity["model_b"])))
        pair_counts[key] = pair_counts.get(key, 0) + 1
        position_counts[identity["model_a"]] = position_counts.get(identity["model_a"], 0) + 1
    assert all(count == 2 for count in pair_counts.values())
    assert len(pair_counts) == 36
    # Blind: no client-visible field contains a model identity.
    documents = export.public_documents()
    for row in documents["study_candidate_text"]:
        assert "DPO" not in str(row["stimulus_id"])
    # Blocks are clip-disjoint.
    seen: dict[int, set[str]] = {}
    for task in export.tasks:
        block_clips = seen.setdefault(task.block, set())
        assert task.clip_id not in block_clips
        block_clips.add(task.clip_id)


def test_study_export_rejects_identity_leaks() -> None:
    captions = _study_captions()
    captions["DPO"]["study-clip-0"] = "the dpo model wrote this caption"
    media = {f"study-clip-{index}": sha256_bytes(f"media-{index}".encode()) for index in range(4)}
    with pytest.raises(StudyExportError, match="blinding"):
        build_study_export(track="visual", captions=captions, media_hashes=media, exposures_per_pair=1)


def test_study_export_requires_the_full_nine_condition_suite() -> None:
    captions = _study_captions()
    del captions["RDPO"]
    media = {f"study-clip-{index}": sha256_bytes(f"media-{index}".encode()) for index in range(4)}
    with pytest.raises(StudyExportError, match="nine-condition"):
        build_study_export(track="visual", captions=captions, media_hashes=media, exposures_per_pair=1)
