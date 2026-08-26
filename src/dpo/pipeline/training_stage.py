"""Shared train-stage implementation: run and publish every matrix cell.

Every experiment variant trains on each track the contract declares, at the
canonical seed, through one runner interface, and each cell is published with
per-variant cache identity: the ``slice_override`` keys the cell on exactly
what it trained from, so extending a sweep axis recomputes only the new cells
and leaves every sibling variant's artifacts untouched.

The matrix has one more axis than the experiment table: the contract's
``[robustness].flip_rates``. Every preference cell trained on the strict pair
view is retrained once per positive rate on the shared flip manifest at that
rate — the same corrupted labels for every method — and each retraining is a
matrix cell of its own, keyed by its rate. ``matrix_cells`` is the one place
that enumerates the matrix, so training, selection, and the contract-identity
check can never disagree about which cells exist.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from dpo.contracts.study_contract import EXPERIMENT_IDS, StudyContract
from dpo.core.artifacts import ArtifactError, ParentEdge
from dpo.core.identity import semantic_hash
from dpo.data.derive_pairs import StrictPair
from dpo.data.noise import FlipManifest
from dpo.pipeline.experiments import ExperimentVariant, expand_experiment, resolve_experiment
from dpo.pipeline.publishing import ArtifactPublisher
from dpo.pipeline.run_matrix import CellResult
from dpo.pipeline.stage_inputs import MATRIX_KEY


class MatrixCellRunner(Protocol):
    """Any backend that can execute one experiment-matrix cell."""

    def run_cell(
        self,
        experiment_id: str,
        *,
        track: str,
        seed: int,
        variant: ExperimentVariant | None = None,
        flip: FlipManifest | None = None,
    ) -> CellResult: ...


@dataclass(frozen=True)
class MatrixCell:
    """One cell of the matrix as the contract declares it, before it runs."""

    experiment_id: str
    variant: ExperimentVariant
    track: str
    flip_rate: float
    training_view: str | None
    hyperparameters: Mapping[str, object]

    @property
    def key(self) -> MATRIX_KEY:
        return (self.experiment_id, self.variant.variant_id, self.track, self.flip_rate)


def flip_rates_of(contract: StudyContract) -> tuple[float, ...]:
    """The contract's synthetic flip rates above zero, ascending."""
    rates = [float(str(rate)) for rate in contract.robustness["flip_rates"]]
    return tuple(sorted(rate for rate in rates if rate > 0.0))


def matrix_cells(contract: StudyContract) -> tuple[MatrixCell, ...]:
    """Every cell this contract declares, in training order.

    The base matrix (every experiment variant on every declared track), plus
    one flipped-label retraining per positive flip rate for each preference
    cell that trains on ``pair_strict`` — the view the shared manifests are
    built over. A WDPO variant on ``pair_all`` therefore has no flip series,
    and the analysis reports none for it rather than a curve over a different
    pair set.
    """
    canonical_seed = int(str(contract.training["canonical_seed"]))
    cells: list[MatrixCell] = []
    for experiment_id in EXPERIMENT_IDS:
        for variant in expand_experiment(contract, experiment_id):
            for track in contract.tracks:
                resolved = resolve_experiment(
                    contract, experiment_id, track=track, seed=canonical_seed, variant=variant
                )
                rates: tuple[float, ...] = (0.0,)
                if resolved.is_preference and resolved.training_view == "pair_strict":
                    rates = (0.0, *flip_rates_of(contract))
                for rate in rates:
                    cells.append(
                        MatrixCell(
                            experiment_id=experiment_id,
                            variant=variant,
                            track=track,
                            flip_rate=rate,
                            training_view=resolved.training_view,
                            hyperparameters=resolved.hyperparameters,
                        )
                    )
    return tuple(cells)


def training_cell_slice(
    contract: StudyContract,
    *,
    experiment_id: str,
    variant_id: str,
    track: str,
    hyperparameters: Mapping[str, object],
    training_view: str | None,
    flip_rate: float,
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
            "flip_rate": flip_rate,
        },
    }


def training_cell_contract_ids(contract: StudyContract) -> frozenset[str]:
    """Every contract identity a matrix cell of this contract may legitimately carry.

    Cells key on their resolved variant rather than on a stage-wide contract
    slice, so a consumer of published cells (validation, selection) recomputes
    the whole legitimate set here instead of widening the stage registry.
    """
    return frozenset(
        semantic_hash(
            training_cell_slice(
                contract,
                experiment_id=cell.experiment_id,
                variant_id=cell.variant.variant_id,
                track=cell.track,
                hyperparameters=cell.hyperparameters,
                training_view=cell.training_view,
                flip_rate=cell.flip_rate,
            )
        )
        for cell in matrix_cells(contract)
    )


def publish_training_matrix(
    publisher: ArtifactPublisher,
    contract: StudyContract,
    *,
    runner: MatrixCellRunner,
    canonical_seed: int,
    view_artifacts: Mapping[str, Mapping[str, str]],
    strict_pairs: Mapping[str, tuple[StrictPair, ...]],
    flip_manifests: Mapping[str, Mapping[float, tuple[str, FlipManifest]]],
) -> tuple[dict[MATRIX_KEY, CellResult], dict[MATRIX_KEY, str]]:
    """Run every cell ``matrix_cells`` declares and publish each result."""
    cells: dict[MATRIX_KEY, CellResult] = {}
    cell_artifacts: dict[MATRIX_KEY, str] = {}
    for spec in matrix_cells(contract):
        experiment_id, track = spec.experiment_id, spec.track
        flip: FlipManifest | None = None
        parent_edges: list[ParentEdge] = []
        if spec.flip_rate > 0.0:
            found = flip_manifests.get(track, {}).get(spec.flip_rate)
            if found is None:
                raise ArtifactError(
                    f"track {track!r} has no flip manifest at rate {spec.flip_rate:g};"
                    " the contract's [robustness] rates need one manifest each"
                )
            flip_artifact_id, flip = found
            parent_edges.append(ParentEdge(flip_artifact_id, "flip-manifest"))
        cell = runner.run_cell(
            experiment_id, track=track, seed=canonical_seed, variant=spec.variant, flip=flip
        )
        cells[spec.key] = cell
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
        cell_artifacts[spec.key] = publisher.publish(
            "dpo.matrix-cell/v1",
            cell.document(),
            parents=tuple(parent_edges),
            stage="train",
            # Per-variant cache identity: the cell keys on exactly what it
            # trained from, so extending a sweep axis or a flip rate leaves
            # every sibling cell's request identity untouched.
            slice_override=training_cell_slice(
                contract,
                experiment_id=experiment_id,
                variant_id=spec.variant.variant_id,
                track=track,
                hyperparameters=cell.hyperparameters,
                training_view=cell.training_view,
                flip_rate=cell.flip_rate,
            ),
            parameters={
                "operation": "train-cell",
                "experiment_id": experiment_id,
                "variant": spec.variant.variant_id,
                "track": track,
                "seed": canonical_seed,
                "flip_rate": cell.flip_rate,
            },
            seed=canonical_seed,
            clips=cell_clips,
            role_exposure={"train"} if cell_clips else None,
            fit_exposure=cell_clips or None,
            purpose="training" if cell.trained else "research",
        )
    return cells, cell_artifacts
