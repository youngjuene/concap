"""The offline end-to-end milestone: cold run, cached warm run, zero providers."""

from __future__ import annotations

from pathlib import Path

from dpo.pipeline.canary import run_canary
from tests.conftest import CANARY_CONTRACT


def test_canary_cold_then_warm_is_cached_with_zero_provider_calls(tmp_path: Path) -> None:
    cold = run_canary(tmp_path / "store", CANARY_CONTRACT)
    assert cold.report["status"] == "offline_milestone_complete"
    assert not cold.cached
    assert cold.provider_calls > 0
    assert cold.report["oracles"] == {
        "leakage": "protected_exposure_rejected",
        "tamper": "payload_corruption_detected",
    }
    assert cold.report["release"] == "blocked_pending_external_operation"
    warm = run_canary(tmp_path / "store", CANARY_CONTRACT)
    assert warm.cached
    assert warm.provider_calls == 0
    assert warm.artifact_id == cold.artifact_id
