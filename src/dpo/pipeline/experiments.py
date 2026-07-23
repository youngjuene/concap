"""Resolution of the code-owned nine-condition matrix against one study contract.

The initialization rules are structural and non-configurable: DPO, IPO, CDPO,
RDPO, DRDPO, and WDPO start from SEED with frozen-SEED references, SFT_DPO
starts from SFT with a frozen-SFT reference and the same pair view as DPO. A
contract chooses hyperparameters and (for WDPO only) the pair view; it can
never move an experiment's initialization, reference, or objective family.

Sweep axes: a contract may give a SWEEPABLE hyperparameter a list of values
(``beta = [0.1, 0.3]``). ``expand_experiment`` turns each experiment table
into its cartesian product of fully-scalar variants; every variant trains as
its own matrix cell, and validation-time selection picks one winner per
experiment and track before anything is locked or studied.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product

from dpo.contracts.study_contract import (
    EXPERIMENT_MATRIX,
    SWEEPABLE_HYPERPARAMETERS,
    ContractError,
    ExperimentSpec,
    StudyContract,
)
from dpo.objectives import build_objective
from dpo.objectives.base import PreferenceObjective


@dataclass(frozen=True)
class ExperimentVariant:
    """One fully-scalar hyperparameter assignment of one experiment."""

    experiment_id: str
    variant_id: str  # "base" without axes; else "beta=0.3" / "beta=0.3,epsilon=0.2"
    hyperparameters: Mapping[str, object]


def expand_experiment(contract: StudyContract, experiment_id: str) -> tuple[ExperimentVariant, ...]:
    """The experiment's sweep variants in contract author order."""
    if experiment_id not in EXPERIMENT_MATRIX:
        raise ContractError(f"unknown experiment {experiment_id!r}")
    table = dict(contract.experiment(experiment_id))
    axes: list[tuple[str, list[float]]] = []
    for key in SWEEPABLE_HYPERPARAMETERS:
        value = table.get(key)
        if isinstance(value, list):
            axes.append((key, [float(str(item)) for item in value]))
    if not axes:
        return (ExperimentVariant(experiment_id, "base", table),)
    variants = []
    for combo in product(*[values for _, values in axes]):
        assignment = dict(table)
        parts = []
        for (key, _), value in zip(axes, combo, strict=True):
            assignment[key] = value
            parts.append(f"{key}={value:g}")
        variants.append(ExperimentVariant(experiment_id, ",".join(parts), assignment))
    return tuple(variants)


@dataclass(frozen=True)
class ResolvedExperiment:
    experiment_id: str
    variant_id: str
    track: str
    seed: int
    spec: ExperimentSpec
    training_view: str | None
    hyperparameters: dict[str, object]

    @property
    def is_trained(self) -> bool:
        return self.spec.objective is not None

    @property
    def is_preference(self) -> bool:
        return self.spec.is_preference

    def build_preference_objective(self) -> PreferenceObjective:
        if not self.is_preference:
            raise ContractError(f"{self.experiment_id} has no preference objective")
        return build_objective(self.hyperparameters)


def resolve_experiment(
    contract: StudyContract,
    experiment_id: str,
    *,
    track: str,
    seed: int,
    variant: ExperimentVariant | None = None,
) -> ResolvedExperiment:
    spec = EXPERIMENT_MATRIX.get(experiment_id)
    if spec is None:
        raise ContractError(f"unknown experiment {experiment_id!r}")
    if track not in contract.tracks:
        raise ContractError(f"unknown track {track!r}")
    if seed not in contract.seeds:
        raise ContractError(f"seed {seed} is not one of the contract training seeds")
    if variant is None:
        variants = expand_experiment(contract, experiment_id)
        if len(variants) != 1:
            raise ContractError(
                f"{experiment_id} declares sweep axes; resolve one of its {len(variants)} variants"
            )
        variant = variants[0]
    elif variant.experiment_id != experiment_id:
        raise ContractError(
            f"variant {variant.variant_id!r} belongs to {variant.experiment_id}, not {experiment_id}"
        )
    table = dict(variant.hyperparameters)
    view: str | None
    if spec.objective is None or spec.objective == "sft":
        view = spec.allowed_views[0] if spec.allowed_views else None
    elif experiment_id == "WDPO":
        view = str(table["view"])
    else:
        view = "pair_strict"
    if view is not None and spec.allowed_views and view not in spec.allowed_views:
        raise ContractError(f"{experiment_id} may only train on {sorted(spec.allowed_views)}, not {view!r}")
    return ResolvedExperiment(
        experiment_id=experiment_id,
        variant_id=variant.variant_id,
        track=track,
        seed=seed,
        spec=spec,
        training_view=view,
        hyperparameters=table,
    )


def initialization_of(experiment_id: str) -> tuple[str | None, str | None]:
    """(policy initialization, frozen reference) for one experiment."""
    spec = EXPERIMENT_MATRIX.get(experiment_id)
    if spec is None:
        raise ContractError(f"unknown experiment {experiment_id!r}")
    return spec.policy_init, spec.reference


def validate_matrix_invariants() -> None:
    """The comparison constraints of PRD section 10, asserted structurally."""
    for experiment_id in ("DPO", "IPO", "CDPO", "RDPO", "DRDPO", "WDPO"):
        spec = EXPERIMENT_MATRIX[experiment_id]
        if spec.policy_init != "SEED" or spec.reference != "SEED":
            raise ContractError(f"{experiment_id} must start from SEED with a frozen SEED reference")
    sft_dpo = EXPERIMENT_MATRIX["SFT_DPO"]
    if sft_dpo.policy_init != "SFT" or sft_dpo.reference != "SFT" or sft_dpo.objective != "dpo":
        raise ContractError("SFT_DPO must start from SFT, reference frozen SFT, standard DPO")
    if sft_dpo.allowed_views != EXPERIMENT_MATRIX["DPO"].allowed_views:
        raise ContractError("SFT_DPO must train on the same pair view as DPO")
    sft = EXPERIMENT_MATRIX["SFT"]
    if sft.policy_init != "SEED" or sft.reference is not None or sft.objective != "sft":
        raise ContractError("SFT must be chosen-only SFT initialized from SEED with no reference")
    seed = EXPERIMENT_MATRIX["SEED"]
    if seed.policy_init is not None or seed.reference is not None or seed.objective is not None:
        raise ContractError("SEED is never trained")


def compute_report_fields(experiment_id: str) -> dict[str, object]:
    """Report scaffolding for the two SFT_DPO comparison modes (PRD section 20.2)."""
    if experiment_id != "SFT_DPO":
        return {"comparison_modes": ["primary"]}
    return {
        "comparison_modes": ["pipeline", "compute_matched"],
        "note": (
            "SFT_DPO consumes the SFT budget plus its own DPO budget; both totals must be"
            " visible in the report and the compute-matched comparison must match updates,"
            " tokens, or accelerator hours against DPO, IPO, CDPO, RDPO, DRDPO, and WDPO"
        ),
    }
