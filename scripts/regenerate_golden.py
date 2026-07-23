"""Regenerate the golden training-step snapshots. Review every diff."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dpo.contracts.study_contract import load_contract  # noqa: E402
from tests.conftest import build_world  # noqa: E402
from tests.golden.harness import golden_training_steps  # noqa: E402


def main() -> int:
    contract = load_contract(REPO_ROOT / "configs" / "study" / "canary.toml")
    world = build_world(contract)
    snapshot = golden_training_steps(world)
    destination = REPO_ROOT / "tests" / "golden" / "golden_values.json"
    destination.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
