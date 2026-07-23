"""Study-contract validator tests: normative constraints stay code-owned."""

from __future__ import annotations

import copy
import tomllib

import pytest

from dpo.contracts.study_contract import (
    EXPERIMENT_MATRIX,
    ContractError,
    StudyContract,
    validate_contract,
)
from dpo.pipeline.experiments import (
    expand_experiment,
    initialization_of,
    resolve_experiment,
    validate_matrix_invariants,
)
from tests.conftest import CANARY_CONTRACT


@pytest.fixture()
def document() -> dict[str, object]:
    with CANARY_CONTRACT.open("rb") as handle:
        return tomllib.load(handle)


def test_canary_contract_validates(document: dict[str, object]) -> None:
    contract = validate_contract(document)
    assert contract.execution_class == "synthetic_canary"
    assert set(contract.tracks) == {"visual", "audio"}


def test_matrix_initialization_rules_are_structural() -> None:
    validate_matrix_invariants()
    for experiment_id in ("DPO", "IPO", "CDPO", "RDPO", "DRDPO", "WDPO"):
        assert initialization_of(experiment_id) == ("SEED", "SEED")
    assert initialization_of("SFT_DPO") == ("SFT", "SFT")
    assert initialization_of("SFT") == ("SEED", None)
    assert initialization_of("SEED") == (None, None)


def test_contract_cannot_reassign_an_objective(document: dict[str, object]) -> None:
    mutated = copy.deepcopy(document)
    experiments = mutated["experiments"]
    assert isinstance(experiments, dict)
    experiments["DPO"] = {"objective": "ipo", "beta": 0.1}
    with pytest.raises(ContractError, match="code-owned"):
        validate_contract(mutated)


def test_sft_dpo_objective_is_pinned_to_standard_dpo(document: dict[str, object]) -> None:
    mutated = copy.deepcopy(document)
    experiments = mutated["experiments"]
    assert isinstance(experiments, dict)
    experiments["SFT_DPO"] = {"objective": "wdpo", "beta": 0.1}
    with pytest.raises(ContractError):
        validate_contract(mutated)


def test_epsilon_bound_is_enforced(document: dict[str, object]) -> None:
    mutated = copy.deepcopy(document)
    experiments = mutated["experiments"]
    assert isinstance(experiments, dict)
    experiments["RDPO"] = {
        "objective": "rdpo",
        "beta": 0.1,
        "epsilon": 0.6,
        "epsilon_mode": "estimated",
    }
    with pytest.raises(ContractError, match="epsilon"):
        validate_contract(mutated)


def test_split_fractions_must_sum_to_one(document: dict[str, object]) -> None:
    mutated = copy.deepcopy(document)
    corpus = mutated["corpus"]
    assert isinstance(corpus, dict)
    fractions = dict(corpus["split_fractions"])
    fractions["train"] = 0.9
    corpus["split_fractions"] = fractions
    with pytest.raises(ContractError, match="sum to 1.0"):
        validate_contract(mutated)


def test_track_prompts_cannot_reference_the_other_modality(document: dict[str, object]) -> None:
    import hashlib

    mutated = copy.deepcopy(document)
    tracks = mutated["tracks"]
    assert isinstance(tracks, dict)
    visual = dict(tracks["visual"])
    bad_prompt = "Describe what you can hear in the clip."
    visual["prompt"] = bad_prompt
    visual["prompt_hash"] = "sha256:" + hashlib.sha256(bad_prompt.encode()).hexdigest()
    tracks["visual"] = visual
    with pytest.raises(ContractError, match="audio evidence"):
        validate_contract(mutated)


def test_wdpo_view_choice_is_bounded(contract: StudyContract) -> None:
    resolved = resolve_experiment(contract, "WDPO", track="visual", seed=1)
    assert resolved.training_view in EXPERIMENT_MATRIX["WDPO"].allowed_views
    dpo = resolve_experiment(contract, "DPO", track="visual", seed=1)
    assert dpo.training_view == "pair_strict"


def test_unknown_keys_are_rejected(document: dict[str, object]) -> None:
    mutated = copy.deepcopy(document)
    mutated["surprise"] = {"x": 1}
    with pytest.raises(ContractError, match="unknown key"):
        validate_contract(mutated)


def _with_beta_axis(document: dict[str, object], values: list[float]) -> dict[str, object]:
    mutated = copy.deepcopy(document)
    experiments = mutated["experiments"]
    assert isinstance(experiments, dict)
    experiments["DPO"] = {"objective": "dpo", "beta": values}
    return mutated


def test_sweep_axis_expands_to_scalar_variants(document: dict[str, object]) -> None:
    contract = validate_contract(_with_beta_axis(document, [0.1, 0.3]))
    variants = expand_experiment(contract, "DPO")
    assert [variant.variant_id for variant in variants] == ["beta=0.1", "beta=0.3"]
    assert [variant.hyperparameters["beta"] for variant in variants] == [0.1, 0.3]
    with pytest.raises(ContractError, match="sweep axes"):
        resolve_experiment(contract, "DPO", track="visual", seed=1)
    resolved = resolve_experiment(contract, "DPO", track="visual", seed=1, variant=variants[1])
    assert resolved.variant_id == "beta=0.3"
    assert resolved.hyperparameters["beta"] == 0.3


def test_sweep_axis_rejects_duplicates_and_empty(document: dict[str, object]) -> None:
    with pytest.raises(ContractError, match="unique"):
        validate_contract(_with_beta_axis(document, [0.1, 0.1]))
    with pytest.raises(ContractError, match="non-empty"):
        validate_contract(_with_beta_axis(document, []))


def test_structural_knobs_are_not_sweepable(document: dict[str, object]) -> None:
    mutated = copy.deepcopy(document)
    experiments = mutated["experiments"]
    assert isinstance(experiments, dict)
    experiments["RDPO"] = {
        "objective": "rdpo",
        "beta": 0.1,
        "epsilon": 0.1,
        "epsilon_mode": ["estimated", "known_synthetic"],
    }
    with pytest.raises(ContractError):
        validate_contract(mutated)


def test_single_variant_contract_expands_to_base(contract: StudyContract) -> None:
    for experiment_id in EXPERIMENT_MATRIX:
        variants = expand_experiment(contract, experiment_id)
        assert [variant.variant_id for variant in variants] == ["base"], experiment_id
