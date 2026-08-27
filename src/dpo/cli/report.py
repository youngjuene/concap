"""`dpo report`: show published reports, and publish the inferential comparison."""

from __future__ import annotations

import argparse
import json
from typing import Any

from dpo.analysis.compare import compare_experiments
from dpo.cli._shared import _emit, _find_manifest, _operation, _require_types
from dpo.core.artifacts import ArtifactError, ArtifactStore, ParentEdge
from dpo.pipeline.lock import parse_lock_manifest

ANALYSIS_REPORT_TYPE = "dpo.analysis-report/v1"


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
                "analyses": [
                    {
                        "artifact_id": artifact_id,
                        "top_experiment": {
                            track: entry["top_experiment"] for track, entry in payload["tracks"].items()
                        },
                    }
                    for artifact_id, payload in _payloads(ANALYSIS_REPORT_TYPE)
                ],
            },
        }
    )
    return 0


def _report_analyze(arguments: argparse.Namespace) -> int:
    """Publish the inferential comparison over one validation + selection report pair.

    Re-scores nothing: every number derives from the per-pair scores the
    validation report persists, so the published ``dpo.analysis-report/v1`` is
    a pure function of its two parents and the contract's resample count, and
    a rerun republishes to the same id.
    """
    operation = _operation(arguments)
    _require_types(operation, {"dpo.validation-report/v1", "dpo.selection-report/v1"}, minimum=2)
    validation_manifest = _find_manifest(operation, "dpo.validation-report/v1")
    selection_manifest = _find_manifest(operation, "dpo.selection-report/v1")
    # The selection report ranks exactly one validation report; an analysis
    # over any other pairing would compare a ranking against scores it never
    # saw.
    if all(parent.artifact_id != validation_manifest.artifact_id for parent in selection_manifest.parents):
        raise ArtifactError(
            f"selection report {selection_manifest.artifact_id} was not derived from validation"
            f" report {validation_manifest.artifact_id}"
        )
    validation = json.loads(operation.store.read_payload(validation_manifest.artifact_id))
    selection = json.loads(operation.store.read_payload(selection_manifest.artifact_id))
    document = compare_experiments(
        validation,
        selection,
        bootstrap_samples=int(str(operation.contract.validation["bootstrap_samples"])),
    )
    artifact_id = operation.publisher().publish(
        ANALYSIS_REPORT_TYPE,
        document,
        parents=(
            ParentEdge(validation_manifest.artifact_id, "validation-report"),
            ParentEdge(selection_manifest.artifact_id, "selection-report"),
        ),
        stage="analyze",
        parameters={"operation": "analyze"},
    )
    _emit(
        {
            "status": "published",
            "operation": "report-analyze",
            "artifact_id": artifact_id,
            "analysis": document,
        }
    )
    return 0
