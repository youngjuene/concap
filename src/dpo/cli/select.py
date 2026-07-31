"""`dpo select`: score variants, select winners, and lock."""

from __future__ import annotations

import argparse

from dpo.cli._backend import _build_backend, _require_media_coverage, _resolve_backend, _selection_identity
from dpo.cli._shared import _emit, _operation, _require_types
from dpo.contracts.study_contract import (
    EXPERIMENT_IDS,
    load_contract,
)
from dpo.core.artifacts import (
    ArtifactError,
)
from dpo.pipeline.experiments import expand_experiment
from dpo.pipeline.live_runner import (
    load_checkpoint_policies,
)
from dpo.pipeline.selection_stage import publish_selection
from dpo.pipeline.stage_inputs import (
    collect_matrix_cells,
    collect_view_inputs,
)
from dpo.pipeline.stages import stage


def _select_run(arguments: argparse.Namespace) -> int:
    contract = load_contract(arguments.contract)
    choice = _resolve_backend(contract, arguments, command="select run")
    if choice is None:
        return 3
    operation = _operation(arguments, with_training_cells=True)
    _require_types(
        operation,
        set(stage("validate").input_artifact_types)
        | set(stage("train").input_artifact_types)
        | {"dpo.validation-pairs/v1"},
        minimum=2,
    )
    views = collect_view_inputs(
        operation.store,
        operation.manifests,
        required=("pair_strict", "validation_pairs"),
        tracks=sorted(contract.tracks),
    )
    cells, cell_artifacts = collect_matrix_cells(operation.store, operation.manifests)
    _require_media_coverage(
        choice,
        {track: {row.clip_id for row in views.validation_pairs.get(track, ())} for track in contract.tracks},
    )
    canonical_seed = int(str(contract.training["canonical_seed"]))
    backend = _build_backend(contract, choice)
    policies, _ = load_checkpoint_policies(backend, arguments.checkpoint_dir)
    variants_by_experiment = {
        experiment_id: expand_experiment(contract, experiment_id) for experiment_id in EXPERIMENT_IDS
    }
    # Selection compares the whole matrix: every variant of every experiment on
    # every track must be both published and materializable, or the ranking
    # would silently be over a subset.
    for experiment_id, variants in sorted(variants_by_experiment.items()):
        for variant in variants:
            for track in contract.tracks:
                key = (experiment_id, variant.variant_id, track)
                if key not in cells:
                    raise ArtifactError(f"no matrix-cell artifact was given for cell {key}")
                if (*key, canonical_seed) not in policies:
                    raise ArtifactError(
                        f"checkpoint directory has no policy for cell {key};"
                        " run `dpo train run` with the same --checkpoint-dir first"
                    )
    identity = _selection_identity(contract, choice)
    selection = publish_selection(
        operation.publisher(),
        contract,
        variants_by_experiment=variants_by_experiment,
        canonical_seed=canonical_seed,
        validation_pairs=views.validation_pairs,
        strict_pairs=views.strict_pairs,
        policies=policies,
        seed_adapters={track: backend.seed_adapter(track) for track in contract.tracks},
        media_provider=backend.media_batch,
        view_artifacts=views.artifact_ids,
        cells=cells,
        cell_artifacts=cell_artifacts,
        processor_hash=identity["processor_hash"],
        preprocessing_hash=identity["preprocessing_hash"],
        evaluation_version="evaluation/v1",
        metric_versions={"compliance": "v1", "preference": "v1"},
        selection_note=(
            f"{choice.name} backend selection: validation preference accuracy, lexical tie-break"
        ),
    )
    _emit(
        {
            "status": "published",
            "operation": "select-run",
            "backend": choice.name,
            "ranking": selection.ranking,
            "selected_variants": selection.selected_variants,
            "validation_accuracy": selection.validation_reports,
            "artifacts": {
                "validation_report": selection.validation_artifact_id,
                "selection_report": selection.selection_artifact_id,
                "lock_manifest": selection.lock_artifact_id,
            },
            "lock_id": selection.lock_manifest.lock_id,
        }
    )
    return 0
