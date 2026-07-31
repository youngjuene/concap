"""`dpo canary`: the offline end-to-end canary."""

from __future__ import annotations

import argparse

from dpo.cli._shared import _emit
from dpo.pipeline.canary import run_canary


def _canary_run(arguments: argparse.Namespace) -> int:
    result = run_canary(arguments.workspace, arguments.contract)
    _emit(
        {
            "status": result.report.get("status", "unknown"),
            "artifact_id": result.artifact_id,
            "cached": result.cached,
            "provider_calls": result.provider_calls,
            "report": result.report,
        }
    )
    return 0
