"""The inferential comparison: deterministic, and honest about separation."""

from __future__ import annotations

import pytest

from dpo.analysis.bradley_terry import AnalysisError
from dpo.analysis.compare import _exact_binomial_two_sided, compare_experiments


def _row(pair: int, clip: str, margin: float) -> dict[str, object]:
    return {
        "pair_id": f"pair-{pair:03d}",
        "clip_id": clip,
        "policy_chosen_logp": margin,
        "policy_rejected_logp": 0.0,
        "ref_chosen_logp": 0.0,
        "ref_rejected_logp": 0.0,
        "difficulty": 0.2,
        "agreement": 1.0,
    }


def _payloads(dpo_margins: list[float], seed_margins: list[float]):
    clips = [f"clip-{index % 3}" for index in range(len(dpo_margins))]
    validation = {
        "schema": "dpo.validation-report/v1",
        "accuracy": {},
        "scores": {
            "audio": {
                "DPO": {"base": [_row(i, clips[i], m) for i, m in enumerate(dpo_margins)]},
                "SEED": {"base": [_row(i, clips[i], m) for i, m in enumerate(seed_margins)]},
            }
        },
    }
    selection = {
        "ranking": {"audio": ["DPO", "SEED"]},
        "selected_variants": {"audio": {"DPO": "base", "SEED": "base"}},
        "selected_hyperparameters": {"audio": {"DPO": {"beta": 0.1}, "SEED": {}}},
    }
    return validation, selection


def test_compare_separates_a_clean_winner() -> None:
    # DPO right on every pair, SEED wrong on every pair: six discordant pairs.
    validation, selection = _payloads([1.0] * 6, [-1.0] * 6)
    document = compare_experiments(validation, selection, bootstrap_samples=64, seed=7)
    track = document["tracks"]["audio"]
    dpo = track["experiments"]["DPO"]
    assert dpo["accuracy"] == 1.0
    assert dpo["accuracy_ci95"] == [1.0, 1.0], "a unanimous statistic has a degenerate CI"
    assert dpo["vs_seed"]["wins"] == 6 and dpo["vs_seed"]["losses"] == 0
    assert dpo["vs_seed"]["p_value"] == pytest.approx(2 * 0.5**6)
    assert dpo["vs_seed"]["bh_significant"] is True
    ranking = track["bradley_terry"]["ranking"]
    assert ranking[0] == "DPO"
    assert track["top_experiment"] == "DPO"
    assert track["experiments"]["SEED"]["comparison_modes"] == ["primary"]


def test_compare_reports_a_tie_as_insignificant() -> None:
    # Identical margins: zero discordant pairs, p = 1, nothing significant.
    validation, selection = _payloads([1.0, -1.0, 1.0], [1.0, -1.0, 1.0])
    document = compare_experiments(validation, selection, bootstrap_samples=32, seed=1)
    dpo = document["tracks"]["audio"]["experiments"]["DPO"]
    assert dpo["vs_seed"] == {"wins": 0, "losses": 0, "p_value": 1.0, "bh_significant": False}


def test_compare_is_deterministic() -> None:
    validation, selection = _payloads([1.0, -0.5, 0.3, -0.2, 0.9, 0.1], [0.5, -1.0, -0.3, 0.2, -0.9, 0.4])
    first = compare_experiments(validation, selection, bootstrap_samples=50, seed=3)
    second = compare_experiments(validation, selection, bootstrap_samples=50, seed=3)
    assert first == second


def test_compare_refuses_a_scoreless_report() -> None:
    with pytest.raises(AnalysisError, match="no per-pair scores"):
        compare_experiments({"accuracy": {}}, {"ranking": {}}, bootstrap_samples=10)


def test_exact_binomial_matches_known_values() -> None:
    assert _exact_binomial_two_sided(0, 0) == 1.0
    assert _exact_binomial_two_sided(5, 5) == pytest.approx(2 * 0.5**5)
    assert _exact_binomial_two_sided(3, 6) == 1.0  # perfectly balanced
