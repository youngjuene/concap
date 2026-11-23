"""The chain's analysis (docs/v4-diff/chain-spec.md §5) on simulated exports with known effects."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from diff.chain_analysis import correlation, fit, regression, simulate


def test_strong_effects_come_back_with_their_signs_and_no_cross_terms(tmp_path: Path) -> None:
    simulate(tmp_path, participants=200, follow=0.4, gamma_context=0.6, gamma_texture=0.6, seed=7)
    report = fit(tmp_path)
    for test in ("H1", "H2", "H3", "R"):
        assert report[test]["estimate"] > 0 and report[test]["p_one_sided"] < 0.01, test
    assert report["H1"]["estimate"] == pytest.approx(0.4, abs=0.1) and report["H1"]["n_participants"] == 200
    assert (report["H1"]["role"], report["H2"]["role"], report["R"]["role"]) == (
        "confirmatory",
        "estimate",
        "confirmatory",
    )
    for cross in ("context_Phi", "texture_B"):
        low, high = report["D"][cross]["interval_90"]
        assert low < 0 < high, cross
    assert sum(report["H2"]["time_at_stop"].values()) == pytest.approx(1, abs=1e-3)


def test_without_effects_nothing_but_the_trait_s_stability_is_found(tmp_path: Path) -> None:
    simulate(tmp_path, participants=48, follow=0.0, gamma_context=0.0, gamma_texture=0.0, seed=11)
    report = fit(tmp_path)
    for test in ("H1", "H2", "H3"):
        low, high = report[test]["interval_90"]
        assert low < 0 < high, test
    assert report["R"]["estimate"] > 0  # R's link is in the data whatever the sliders do


def test_the_slider_table_is_what_the_export_writes(tmp_path: Path) -> None:
    from diff.export_bundle import SLIDER_COLUMNS

    simulate(tmp_path, participants=6, seed=3)
    with (tmp_path / "phase2_sliders.csv").open(encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    used = {"participant", "video_id", "axis", "start", "settled", "binding_B", "fascination_Phi"}
    assert used | {f"share_{k}" for k in range(3)} <= set(header) <= set(SLIDER_COLUMNS)


def test_the_standard_errors_are_the_textbook_ones() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=40)
    rows = [{"participant": f"p{i}", "x": v, "y": 2 * v + rng.normal()} for i, v in enumerate(x)]
    plain = regression(rows, ("x",), (), cluster=False)["x"]
    slope, intercept = np.polyfit(x, [r["y"] for r in rows], 1)
    residual = np.array([r["y"] for r in rows]) - (slope * x + intercept)
    se = np.sqrt(residual @ residual / 38 / ((x - x.mean()) @ (x - x.mean())))
    assert plain["estimate"] == pytest.approx(slope, abs=1e-4) and plain["se"] == pytest.approx(se, abs=1e-4)
    # With one row per participant, the cluster-robust error is the heteroskedasticity-robust one: close here.
    assert regression(rows, ("x",), (), cluster=True)["x"]["se"] == pytest.approx(se, rel=0.3)
    r = correlation([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 6, 5])
    assert 0.9 < r["estimate"] < 1 and r["interval_90"][0] < r["estimate"] < r["interval_90"][1]
