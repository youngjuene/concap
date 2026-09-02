"""Evaluation tests: preference accuracy and the candidate-reuse gate."""

from __future__ import annotations

import math

import pytest

from dpo.evaluation.caption_reuse import GeneratedCaption, training_candidate_reuse_rate
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


def test_generated_captions_must_not_reuse_training_candidates(
    world: PreferenceWorld,
) -> None:
    reused = GeneratedCaption(clip_id=world.pool.candidates[0].clip_id, text=world.pool.candidates[0].text)
    fresh = GeneratedCaption(clip_id="clip-000", text="An entirely fresh generated caption.")
    assert training_candidate_reuse_rate([reused, fresh], world.pool) == pytest.approx(0.5)
