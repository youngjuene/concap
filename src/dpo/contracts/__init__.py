"""Caption contracts and the executable study contract.

`schema` owns the shared normative vocabularies (tracks, splits, annotation
choices, pair categories, claim statuses, the nine-condition experiment
matrix) and the study-contract validator. `visual` and `audio` own the
per-track caption contracts: deterministic compliance checks and cross-modal
lexical screens.
"""

from dpo.contracts.study_contract import (
    EXPERIMENT_IDS,
    EXPERIMENT_MATRIX,
    OBJECTIVE_NAMES,
    PAIR_CATEGORIES,
    SPLITS,
    TRACKS,
    CaptionContract,
    ContractError,
    ExperimentSpec,
    StudyContract,
    load_contract,
    validate_contract,
)

__all__ = [
    "EXPERIMENT_IDS",
    "EXPERIMENT_MATRIX",
    "OBJECTIVE_NAMES",
    "PAIR_CATEGORIES",
    "SPLITS",
    "TRACKS",
    "CaptionContract",
    "ContractError",
    "ExperimentSpec",
    "StudyContract",
    "load_contract",
    "validate_contract",
]
