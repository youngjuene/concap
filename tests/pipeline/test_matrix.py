"""Matrix-runner tests: initialization rules made operational."""

from __future__ import annotations

import copy
import tomllib

import pytest

from dpo.contracts.study_contract import ContractError, StudyContract, validate_contract
from dpo.pipeline.experiments import expand_experiment
from dpo.pipeline.run_matrix import OfflineMatrixRunner
from tests.conftest import CANARY_CONTRACT, PreferenceWorld, build_offline_runner


@pytest.fixture(scope="module")
def runner(world: PreferenceWorld) -> OfflineMatrixRunner:
    return build_offline_runner(world)


def test_seed_is_never_trained(runner: OfflineMatrixRunner) -> None:
    cell = runner.run_cell("SEED", track="visual", seed=1)
    assert not cell.trained
    assert cell.steps == 0
    assert cell.checkpoint_signature == runner.seed_adapter("visual").state_signature()


def test_dpo_starts_from_seed_and_references_frozen_seed(runner: OfflineMatrixRunner) -> None:
    cell = runner.run_cell("DPO", track="visual", seed=1)
    assert cell.reference_signature == runner.seed_adapter("visual").state_signature()
    assert cell.checkpoint_signature != cell.reference_signature


def test_sft_dpo_starts_from_sft_and_references_frozen_sft(runner: OfflineMatrixRunner) -> None:
    sft = runner.run_cell("SFT", track="visual", seed=1)
    sft_dpo = runner.run_cell("SFT_DPO", track="visual", seed=1)
    assert sft_dpo.reference_signature == sft.checkpoint_signature
    assert sft_dpo.compute["sft_steps"] == sft.steps
    assert sft_dpo.compute["preference_steps"] > 0
    dpo = runner.run_cell("DPO", track="visual", seed=1)
    assert dpo.compute["sft_steps"] == 0
    assert sft_dpo.training_view == dpo.training_view == "pair_strict"


def test_seeds_produce_distinct_trained_checkpoints(runner: OfflineMatrixRunner) -> None:
    first = runner.run_cell("DPO", track="visual", seed=1)
    second = runner.run_cell("DPO", track="visual", seed=2)
    assert first.checkpoint_signature != second.checkpoint_signature
    assert first.reference_signature == second.reference_signature


def test_all_preference_cells_share_the_seed_reference(runner: OfflineMatrixRunner) -> None:
    seed_signature = runner.seed_adapter("visual").state_signature()
    for experiment_id in ("DPO", "IPO", "CDPO", "RDPO", "DRDPO", "WDPO"):
        cell = runner.run_cell(experiment_id, track="visual", seed=1)
        assert cell.reference_signature == seed_signature, experiment_id


def _sweep_contract(base_document: dict[str, object]) -> StudyContract:
    mutated = copy.deepcopy(base_document)
    experiments = mutated["experiments"]
    assert isinstance(experiments, dict)
    experiments["DPO"] = {"objective": "dpo", "beta": [0.1, 0.3]}
    return validate_contract(mutated)


def test_sweep_variants_train_distinct_cells(world: PreferenceWorld) -> None:
    with CANARY_CONTRACT.open("rb") as handle:
        document = tomllib.load(handle)
    sweep = _sweep_contract(document)
    sweep_runner = build_offline_runner(world, contract=sweep)
    variants = expand_experiment(sweep, "DPO")
    assert [variant.variant_id for variant in variants] == ["beta=0.1", "beta=0.3"]
    with pytest.raises(ContractError, match="sweep axes"):
        sweep_runner.run_cell("DPO", track="visual", seed=1)
    cells = [sweep_runner.run_cell("DPO", track="visual", seed=1, variant=variant) for variant in variants]
    assert cells[0].checkpoint_signature != cells[1].checkpoint_signature
    assert cells[0].reference_signature == cells[1].reference_signature
    assert cells[0].hyperparameters["beta"] == 0.1
    assert cells[1].hyperparameters["beta"] == 0.3
    run_results = sweep_runner.run(experiment_ids=["DPO"], tracks=["visual"], seeds=[1])
    assert [(cell.experiment_id, cell.variant_id) for cell in run_results] == [
        ("DPO", "beta=0.1"),
        ("DPO", "beta=0.3"),
    ]
