"""`dpo report`: show published validation/selection/lock reports."""

from __future__ import annotations

import argparse
import json
from typing import Any

from dpo.cli._shared import _emit
from dpo.core.artifacts import (
    ArtifactStore,
)
from dpo.pipeline.lock import parse_lock_manifest


def _report_show(arguments: argparse.Namespace) -> int:
    store = ArtifactStore.open(arguments.workspace)

    def _payloads(artifact_type: str) -> list[tuple[str, dict[str, Any]]]:
        rows: list[tuple[str, dict[str, Any]]] = []
        for artifact_id in store.find_by_type(artifact_type):
            payload = json.loads(store.read_payload(artifact_id))
            rows.append((artifact_id, payload))
        return rows

    _emit(
        {
            "status": "ok",
            "reports": {
                "validation": [
                    {"artifact_id": artifact_id, "accuracy": payload["accuracy"]}
                    for artifact_id, payload in _payloads("dpo.validation-report/v1")
                ],
                "selection": [
                    {
                        "artifact_id": artifact_id,
                        "ranking": payload["ranking"],
                        "selected_variants": payload["selected_variants"],
                        "selected_hyperparameters": payload["selected_hyperparameters"],
                    }
                    for artifact_id, payload in _payloads("dpo.selection-report/v1")
                ],
                "locks": [
                    {"artifact_id": artifact_id, "lock_id": parse_lock_manifest(payload).lock_id}
                    for artifact_id, payload in _payloads("dpo.lock-manifest/v1")
                ],
            },
        }
    )
    return 0
