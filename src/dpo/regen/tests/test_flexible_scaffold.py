from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dpo.cli import build_parser
from dpo.cli.regen import _regen_scaffold


@pytest.mark.parametrize("count", [1, 4])
def test_scaffold_explicit_segment_order(tmp_path: Path, monkeypatch: Any, count: int) -> None:
    names = [f"segment-{index}" for index in range(count)]
    clips = [f"source-{index}" for index in range(count)]
    args = build_parser().parse_args(
        [
            "regen",
            "scaffold",
            "--media-dir",
            str(tmp_path),
            "--out",
            str(tmp_path / "doc.json"),
            "--session-id",
            "flex",
            "--study-id",
            "flex",
            "--corpus-id",
            "flex",
            "--clips",
            *clips,
            "--segment-ids",
            *names,
        ]
    )
    calls = []

    def segment(root: Path, name: str, clip: str, calibration: Any, duration: int) -> dict[str, Any]:
        calls.append((name, clip))
        return {"segment": name, "clip_id": clip}

    monkeypatch.setattr("dpo.cli.regen._segment", segment)
    assert _regen_scaffold(args) == 0
    document = json.loads((tmp_path / "doc.json").read_text())
    assert document["schema"] == "dpo.caption-regen/v4"
    assert document["clip_order"] == names
    assert calls == list(zip(names, clips, strict=True))


def test_scaffold_rejects_unsafe_segment_path(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "regen",
            "scaffold",
            "--media-dir",
            str(tmp_path),
            "--out",
            str(tmp_path / "doc.json"),
            "--session-id",
            "flex",
            "--study-id",
            "flex",
            "--corpus-id",
            "flex",
            "--clips",
            "source",
            "--segment-ids",
            "../outside",
        ]
    )
    assert _regen_scaffold(args) == 2
    assert not (tmp_path / "doc.json").exists()
