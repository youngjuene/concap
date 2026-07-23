"""Bradley-Terry, clip-cluster bootstrap, and robustness tests."""

from __future__ import annotations

import pytest

from dpo.analysis.bootstrap import benjamini_hochberg, clip_cluster_bootstrap
from dpo.analysis.bradley_terry import AnalysisError, PairwiseOutcome, fit_bradley_terry
from dpo.analysis.robustness import flip_curve, sliced_preference_reports
from dpo.evaluation.preference_accuracy import ScoredPair


def _outcomes() -> list[PairwiseOutcome]:
    outcomes = []
    # A beats B 9/1, B beats C 8/2, A beats C 9/1: strengths must order A>B>C.
    for index in range(10):
        outcomes.append(PairwiseOutcome("A", "B", "A" if index < 9 else "B", clip_id=f"clip-{index % 3}"))
        outcomes.append(PairwiseOutcome("B", "C", "B" if index < 8 else "C", clip_id=f"clip-{index % 3}"))
        outcomes.append(PairwiseOutcome("A", "C", "A" if index < 9 else None, clip_id=f"clip-{index % 3}"))
    return outcomes


def test_bradley_terry_recovers_the_ordering() -> None:
    fit = fit_bradley_terry(_outcomes())
    assert fit.converged
    assert fit.ranking() == ["A", "B", "C"]
    strengths = fit.log_strengths
    assert strengths["A"] > strengths["B"] > strengths["C"]


def test_bradley_terry_needs_two_models() -> None:
    with pytest.raises(AnalysisError):
        fit_bradley_terry([])


def test_clip_cluster_bootstrap_interval_contains_the_point() -> None:
    outcomes = _outcomes()
    point, (low, high) = clip_cluster_bootstrap(
        outcomes,
        lambda outcome: outcome.clip_id,
        lambda rows: sum(1.0 for row in rows if row.winner == "A") / len(rows),
        samples=200,
        seed=11,
    )
    assert low <= point <= high


def test_benjamini_hochberg_orders_rejections() -> None:
    decisions = benjamini_hochberg({"a": 0.001, "b": 0.02, "c": 0.4, "d": 0.9}, alpha=0.05)
    assert decisions["a"] is True
    assert decisions["d"] is False


def test_natural_noise_slices_partition_by_metadata() -> None:
    pairs = [
        ScoredPair("p1", "c1", "visual", 1.0, 0.0, 0.0, 0.0, difficulty=0.9, agreement=0.5),
        ScoredPair("p2", "c2", "visual", -1.0, 0.0, 0.0, 0.0, difficulty=0.1, agreement=1.0),
    ]
    reports = sliced_preference_reports(pairs, beta=1.0)
    assert reports["near_tie"]["pair_count"] == 1
    assert reports["easy"]["pair_count"] == 1
    assert reports["low_agreement"]["pair_count"] == 1


def test_flip_curve_orders_points_and_separates_epsilon_modes() -> None:
    pairs = [ScoredPair("p1", "c1", "visual", 1.0, 0.0, 0.0, 0.0, difficulty=0.5, agreement=1.0)]
    curve = flip_curve(
        [(0.2, "estimated", pairs), (0.0, "estimated", pairs), (0.1, "known_synthetic", pairs)],
        beta=1.0,
    )
    assert [(point.epsilon_mode, point.flip_rate) for point in curve] == [
        ("estimated", 0.0),
        ("estimated", 0.2),
        ("known_synthetic", 0.1),
    ]
