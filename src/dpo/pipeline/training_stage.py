"""Shared train-stage implementation: run and publish every matrix cell.

Every experiment variant trains on both tracks at the canonical seed through
one runner interface, and each cell is published with per-variant cache
identity: the ``slice_override`` keys the cell on exactly what it trained
from, so extending a sweep axis recomputes only the new cells and leaves
every sibling variant's artifacts untouched.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from dpo.contracts.study_contract import EXPERIMENT_IDS, TRACKS, StudyContract
from dpo.core.artifacts import ParentEdge
from dpo.core.identity import semantic_hash
from dpo.data.derive_pairs import StrictPair
from dpo.pipeline.experiments import ExperimentVariant, expand_experiment, resolve_experiment
from dpo.pipeline.publishing import ArtifactPublisher
from dpo.pipeline.run_matrix import CellResult


class MatrixCellRunner(Protocol):
    """Any backend that can execute one experiment-matrix cell."""

    def run_cell(
        self,
        experiment_id: str,
        *,
        track: str,
        seed: int,
        variant: ExperimentVariant | None = None,
    ) -> CellResult: ...


def training_cell_slice(
    contract: StudyContract,
    *,
    experiment_id: str,
    variant_id: str,
    track: str,
    hyperparameters: Mapping[str, object],
    training_view: str | None,
) -> dict[str, object]:
    """The per-variant cache identity of one training cell.

    Published cells key on exactly this, so extending a sweep axis leaves every
    sibling variant's request identity untouched. It has one construction site
    so a consumer can recompute the identity of a cell it is given.
    """
    return {
        "training": dict(contract.raw["training"]),
        "models_seed": dict(contract.raw["models"]["seed"]),
        "track_contract": dict(contract.raw["tracks"][track]),
        "experiment": {
            "experiment_id": experiment_id,
            "variant_id": variant_id,
            "hyperparameters": dict(hyperparameters),
            "training_view": training_view,
        },
    }


def training_cell_contract_ids(contract: StudyContract) -> frozenset[str]:
    """Every contract identity a matrix cell of this contract may legitimately carry.

    Cells key on their resolved variant rather than on a stage-wide contract
    slice, so a consumer of published cells (validation, selection) recomputes
    the whole legitimate set here instead of widening the stage registry.
    """
    canonical_seed = int(str(contract.training["canonical_seed"]))
    identities = set()
    for experiment_id in EXPERIMENT_IDS:
        for variant in expand_experiment(contract, experiment_id):
            for track in TRACKS:
                resolved = resolve_experiment(
                    contract, experiment_id, track=track, seed=canonical_seed, variant=variant
                )
                identities.add(
                    semantic_hash(
                        training_cell_slice(
                            contract,
                            experiment_id=experiment_id,
                            variant_id=variant.variant_id,
                            track=track,
                            hyperparameters=resolved.hyperparameters,
                            training_view=resolved.training_view,
                        )
                    )
                )
    return frozenset(identities)


def publish_training_matrix(
    publisher: ArtifactPublisher,
    contract: StudyContract,
    *,
    runner: MatrixCellRunner,
    variants_by_experiment: Mapping[str, tuple[ExperimentVariant, ...]],
    canonical_seed: int,
    view_artifacts: Mapping[str, Mapping[str, str]],
    strict_pairs: Mapping[str, tuple[StrictPair, ...]],
) -> tuple[dict[tuple[str, str, str], CellResult], dict[tuple[str, str, str], str]]:
    """Run every (experiment, variant, track) cell and publish each result."""
    cells: dict[tuple[str, str, str], CellResult] = {}
    cell_artifacts: dict[tuple[str, str, str], str] = {}
    for experiment_id in EXPERIMENT_IDS:
        for variant in variants_by_experiment[experiment_id]:
            for track in TRACKS:
                cell = runner.run_cell(experiment_id, track=track, seed=canonical_seed, variant=variant)
                cells[(experiment_id, variant.variant_id, track)] = cell
                parent_edges: list[ParentEdge] = []
                views = view_artifacts[track]
                if experiment_id == "SFT":
                    parent_edges.append(ParentEdge(views["sft"], "training-view"))
                elif experiment_id == "SFT_DPO":
                    parent_edges.append(ParentEdge(views["sft"], "warm-start-view"))
                    parent_edges.append(ParentEdge(views["pair_strict"], "training-view"))
                elif cell.trained:
                    view_key = "pair_all" if cell.training_view == "pair_all" else "pair_strict"
                    parent_edges.append(ParentEdge(views[view_key], "training-view"))
                cell_clips = {row.clip_id for row in strict_pairs[track]} if cell.trained else set()
                cell_artifacts[(experiment_id, variant.variant_id, track)] = publisher.publish(
                    "dpo.matrix-cell/v1",
                    cell.document(),
                    parents=tuple(parent_edges),
                    stage="train",
                    # Per-variant cache identity: the cell keys on exactly what it
                    # trained from, so extending a sweep axis leaves every sibling
                    # variant's request identity untouched.
                    slice_override=training_cell_slice(
                        contract,
                        experiment_id=experiment_id,
                        variant_id=variant.variant_id,
                        track=track,
                        hyperparameters=cell.hyperparameters,
                        training_view=cell.training_view,
                    ),
                    parameters={
                        "operation": "train-cell",
                        "experiment_id": experiment_id,
                        "variant": variant.variant_id,
                        "track": track,
                        "seed": canonical_seed,
                    },
                    seed=canonical_seed,
                    clips=cell_clips,
                    role_exposure={"train"} if cell_clips else None,
                    fit_exposure=cell_clips or None,
                    purpose="training" if cell.trained else "research",
                )
    return cells, cell_artifacts
