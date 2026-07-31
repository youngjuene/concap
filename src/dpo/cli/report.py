"""`dpo report`: show published validation/selection/lock reports."""

from __future__ import annotations

import argparse
import json
from typing import Any

from dpo.analysis.compare import DEFAULT_BOOTSTRAP_SAMPLES, compare_experiments
from dpo.cli._shared import _emit
from dpo.contracts.study_contract import load_contract
from dpo.core.artifacts import (
    ArtifactError,
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


def _report_analyze(arguments: argparse.Namespace) -> int:
    """Inferential comparison over the published validation + selection reports.

    Read-only: consumes payloads, publishes nothing. Promotion to a published
    dpo.analysis-report/v1 artifact is deliberate follow-up work once the
    authors have seen the shape on real data (docs/TODO.md).
    """
    store = ArtifactStore.open(arguments.workspace)
    contract = load_contract(arguments.contract)

    def _single(artifact_type: str) -> dict[str, Any]:
        ids = store.find_by_type(artifact_type)
        if len(ids) != 1:
            raise ArtifactError(
                f"expected exactly one {artifact_type} in the workspace, found {len(ids)};"
                " pass a workspace holding one select run"
            )
        return dict(json.loads(store.read_payload(ids[0])))

    samples = int(str(contract.validation.get("bootstrap_samples", DEFAULT_BOOTSTRAP_SAMPLES)))
    document = compare_experiments(
        _single("dpo.validation-report/v1"),
        _single("dpo.selection-report/v1"),
        bootstrap_samples=samples,
    )
    _emit({"status": "ok", "analysis": document})
    return 0
