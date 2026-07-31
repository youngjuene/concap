"""`dpo train`: run and publish every experiment-matrix cell."""

from __future__ import annotations

import argparse
from pathlib import Path

from dpo.cli._backend import _build_backend, _require_media_coverage, _resolve_backend
from dpo.cli._shared import _emit, _operation, _require_types
from dpo.contracts.study_contract import (
    EXPERIMENT_IDS,
    load_contract,
)
from dpo.pipeline.experiments import expand_experiment
from dpo.pipeline.live_runner import (
    LiveMatrixRunner,
)
from dpo.pipeline.stage_inputs import (
    collect_view_inputs,
)
from dpo.pipeline.stages import stage
from dpo.pipeline.training_stage import publish_training_matrix


def _train_run(arguments: argparse.Namespace) -> int:
    contract = load_contract(arguments.contract)
    choice = _resolve_backend(contract, arguments, command="train run")
    if choice is None:
        return 3
    operation = _operation(arguments)
    _require_types(operation, set(stage("train").input_artifact_types), minimum=2)
    views = collect_view_inputs(
        operation.store,
        operation.manifests,
        required=("sft", "pair_strict", "pair_all"),
        tracks=sorted(contract.tracks),
    )
    _require_media_coverage(
        choice,
        {
            track: {row.clip_id for row in views.sft_rows.get(track, ())}
            | {row.clip_id for row in views.strict_pairs.get(track, ())}
            | {row.clip_id for row in views.metadata_pairs.get(track, ())}
            for track in contract.tracks
        },
    )
    canonical_seed = int(str(contract.training["canonical_seed"]))
    runner = LiveMatrixRunner(
        contract=contract,
        backend=_build_backend(contract, choice),
        checkpoint_dir=Path(arguments.checkpoint_dir),
        strict_pairs=views.strict_pairs,
        metadata_pairs=views.metadata_pairs,
        sft_rows=views.sft_rows,
    )
    variants_by_experiment = {
        experiment_id: expand_experiment(contract, experiment_id) for experiment_id in EXPERIMENT_IDS
    }
    cells, cell_artifacts = publish_training_matrix(
        operation.publisher(),
        contract,
        runner=runner,
        variants_by_experiment=variants_by_experiment,
        canonical_seed=canonical_seed,
        view_artifacts=views.artifact_ids,
        strict_pairs=views.strict_pairs,
    )
    _emit(
        {
            "status": "published",
            "operation": "train-run",
            "backend": choice.name,
            "checkpoint_dir": str(Path(arguments.checkpoint_dir).resolve()),
            "cells": len(cells),
            "cells_trained": runner.trained_count,
            "cells_resumed": runner.resumed_count,
            "matrix": [
                {
                    "experiment_id": experiment_id,
                    "variant_id": variant_id,
                    "track": track,
                    "artifact_id": cell_artifacts[(experiment_id, variant_id, track)],
                    "steps": cells[(experiment_id, variant_id, track)].steps,
                    "resumed": runner.was_resumed(experiment_id, variant_id, track, canonical_seed),
                }
                for (experiment_id, variant_id, track) in sorted(cells)
            ],
        }
    )
    return 0
