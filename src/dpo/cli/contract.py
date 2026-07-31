"""`dpo contract`: validate and lock the study contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from dpo.cli._shared import _emit
from dpo.contracts.study_contract import (
    load_contract,
)
from dpo.core.identity import canonical_bytes, repo_lock_hash


def _contract_validate(arguments: argparse.Namespace) -> int:
    contract = load_contract(arguments.contract)
    _emit(
        {
            "status": "ok",
            "contract_hash": contract.contract_hash,
            "execution_class": contract.execution_class,
            "tracks": sorted(contract.tracks),
            "seeds": list(contract.seeds),
        }
    )
    return 0


def _contract_lock(arguments: argparse.Namespace) -> int:
    from dpo.core.atomic import atomic_write_bytes

    contract = load_contract(arguments.contract)
    document = {
        "schema": "dpo.contract-lock/v1",
        "contract_hash": contract.contract_hash,
        "contract": contract.raw,
        "environment_lock": repo_lock_hash(),
    }
    payload = canonical_bytes(document) + b"\n"
    written = atomic_write_bytes(arguments.out, payload)
    _emit(
        {
            "status": "published" if written else "cached",
            "contract_hash": contract.contract_hash,
            "out": str(Path(arguments.out).resolve()),
        }
    )
    return 0
