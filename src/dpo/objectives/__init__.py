"""The preference-objective registry (PRD section 34, phase 6).

`OBJECTIVES` maps objective names to implementations; `build_objective`
constructs one from a validated experiment table. The matrix runner never
instantiates an objective any other way, so a typo fails at contract
validation instead of surfacing mid-training.
"""

from __future__ import annotations

from collections.abc import Mapping

from dpo.objectives.base import (
    LossOutput,
    ObjectiveError,
    PreferenceBatch,
    PreferenceObjective,
    preference_margin,
)
from dpo.objectives.cdpo import CDPOObjective
from dpo.objectives.dpo import DPOObjective
from dpo.objectives.drdpo import DrDPOObjective
from dpo.objectives.ipo import IPOObjective
from dpo.objectives.rdpo import RDPOObjective
from dpo.objectives.sft import SFTObjective
from dpo.objectives.wdpo import WDPOObjective

OBJECTIVES: dict[str, type] = {
    "dpo": DPOObjective,
    "ipo": IPOObjective,
    "cdpo": CDPOObjective,
    "rdpo": RDPOObjective,
    "drdpo": DrDPOObjective,
    "wdpo": WDPOObjective,
}

__all__ = [
    "OBJECTIVES",
    "CDPOObjective",
    "DPOObjective",
    "DrDPOObjective",
    "IPOObjective",
    "LossOutput",
    "ObjectiveError",
    "PreferenceBatch",
    "PreferenceObjective",
    "RDPOObjective",
    "SFTObjective",
    "WDPOObjective",
    "build_objective",
    "preference_margin",
]


def build_objective(table: Mapping[str, object]) -> PreferenceObjective:
    """Instantiate a preference objective from a validated experiment table."""
    name = str(table.get("objective", ""))
    if name not in OBJECTIVES:
        raise ObjectiveError(f"unknown preference objective {name!r}")
    beta = float(str(table["beta"]))
    if name == "dpo":
        return DPOObjective(beta=beta)
    if name == "ipo":
        return IPOObjective(beta=beta)
    if name == "cdpo":
        return CDPOObjective(beta=beta, epsilon=float(str(table["epsilon"])))
    if name == "rdpo":
        mode = str(table["epsilon_mode"])
        suffix = "_epsilon"
        mode_name = mode.removesuffix(suffix) if mode.endswith(suffix) else mode
        return RDPOObjective(beta=beta, epsilon=float(str(table["epsilon"])), epsilon_mode=mode_name)
    if name == "drdpo":
        return DrDPOObjective(beta=beta, beta_prime=float(str(table["beta_prime"])))
    return WDPOObjective(
        beta=beta,
        flip_warmup_ratio=float(str(table["flip_warmup_ratio"])),
        flip_budget=float(str(table["flip_budget"])),
        tail_quantile=float(str(table["tail_quantile"])),
        max_tail_cap=float(str(table["max_tail_cap"])),
        cap_floor=float(str(table["cap_floor"])),
        enable_correction=bool(table.get("enable_correction", True)),
        enable_winsorization=bool(table.get("enable_winsorization", True)),
        paper_revision=str(table["paper_revision"]),
        code_commit=str(table.get("code_commit", "")),
    )
