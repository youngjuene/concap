"""Helpers shared by diff's tests: the synthetic source fixture, relabelled per test.

The fixture describes no real clip. Tests that need development material
relabel a copy; nothing here is study material. (Imported by name, so it has a
unique basename instead of living in conftest.py.)
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).parent / "fixtures"


def smoke_document() -> dict[str, Any]:
    return dict(json.loads((FIXTURES / "diff-smoke-sources.json").read_text(encoding="utf-8")))


def relabelled(material: str) -> dict[str, Any]:
    document = copy.deepcopy(smoke_document())
    for source in document["sources"]:
        source["material"] = material
    return document
