"""One valid session document, staged media to match it, and the pieces to bend it."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dpo.regen.config import Calibration, Configuration
from dpo.regen.document import REGEN_SCHEMA

CUE_SLOTS = 4
DURATION_MS = 10000


def track(texts: list[str]) -> list[dict[str, Any]]:
    span = DURATION_MS // len(texts)
    return [
        {"index": index, "start_ms": index * span, "end_ms": (index + 1) * span, "text": text}
        for index, text in enumerate(texts)
    ]


def segment(name: str, clip_id: str, *, objects: list[tuple[str, str]], stems: list[str]) -> dict[str, Any]:
    return {
        "segment": name,
        "clip_id": clip_id,
        "video": f"{name}/clip.mp4",
        "still": f"{name}/still.png",
        "duration_ms": DURATION_MS,
        "objects": [
            {"id": object_id, "label": label, "mask": f"{name}/masks/{object_id}.png"}
            for object_id, label in objects
        ],
        "stems": [
            {
                "id": stem,
                "label": stem.replace("_", " ").title(),
                "audio": f"{name}/stems/{stem}.wav",
                "colour": "#3F83D1",
                "gain": 1.0,
                "waveform": [0.5] * 64,
            }
            for stem in stems
        ],
        "prepared_track": track([f"Prepared {name} {index}." for index in range(CUE_SLOTS)]),
        "fallback_track": track([f"Fallback {name} {index}." for index in range(CUE_SLOTS)]),
    }


@pytest.fixture
def configuration() -> Configuration:
    return Configuration(
        study_id="street2026",
        corpus_id="amsterdam",
        calibration=Calibration(cue_slots=CUE_SLOTS, minimum_points=2),
    )


@pytest.fixture
def document(configuration: Configuration) -> dict[str, Any]:
    """Two segments of one scene, both fully prepared (§10)."""
    return {
        "schema": REGEN_SCHEMA,
        "session_id": "street-regen",
        "config": configuration.artifact(),
        "segments": {
            "A": segment(
                "A",
                "amsterdam_006",
                objects=[("building", "Building"), ("person", "Person")],
                stems=["traffic", "bird"],
            ),
            "B": segment(
                "B",
                "amsterdam_012",
                objects=[("road", "Road"), ("vegetation", "Vegetation")],
                stems=["siren", "footsteps"],
            ),
        },
    }


@pytest.fixture
def media_dir(document: dict[str, Any], tmp_path: Path) -> Path:
    """Media staged to match the document: real mask PNGs, stub video and audio.

    The masks are real images because §4's matching reads them; a stub would
    make every point unclassified and the tests would pass for the wrong
    reason. ``person`` is a small square inside the larger ``building``, so the
    overlap rule — smallest container wins — has something to decide.
    """
    from PIL import Image

    root = tmp_path / "media"
    for name in ("A", "B"):
        (root / name / "masks").mkdir(parents=True)
        (root / name / "stems").mkdir(parents=True)
        (root / name / "clip.mp4").write_bytes(b"\x00")
        Image.new("L", (100, 100), 0).save(root / name / "still.png")
        for stem in document["segments"][name]["stems"]:
            (root / name / "stems" / f"{stem['id']}.wav").write_bytes(b"\x00")
        entries = document["segments"][name]["objects"]
        # The first object covers the left half; the second is a 20x20 square
        # inside it, at (10, 10).
        big = Image.new("L", (100, 100), 0)
        for x in range(50):
            for y in range(100):
                big.putpixel((x, y), 255)
        big.save(root / name / "masks" / f"{entries[0]['id']}.png")
        small = Image.new("L", (100, 100), 0)
        for x in range(10, 30):
            for y in range(10, 30):
                small.putpixel((x, y), 255)
        small.save(root / name / "masks" / f"{entries[1]['id']}.png")
    return root
