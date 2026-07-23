"""One deterministic training step per trained objective on the shared fixture."""

from __future__ import annotations

from dpo.contracts.study_contract import EXPERIMENT_IDS
from tests.conftest import PreferenceWorld, build_offline_runner


def golden_training_steps(world: PreferenceWorld) -> dict[str, dict[str, float]]:
    runner = build_offline_runner(world)
    snapshot: dict[str, dict[str, float]] = {}
    for experiment_id in EXPERIMENT_IDS:
        cell = runner.run_cell(experiment_id, track="visual", seed=1)
        values: dict[str, float] = {"steps": float(cell.steps)}
        if cell.final_loss is not None:
            values["final_loss"] = round(cell.final_loss, 6)
        summary = cell.diagnostics_summary
        means = summary.get("means")
        if isinstance(means, dict) and "margin_mean" in means:
            values["margin_mean"] = round(float(str(means["margin_mean"])), 6)
        snapshot[experiment_id] = values
    return snapshot
