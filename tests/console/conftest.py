"""One valid console document, and the pieces to bend it out of shape."""

from __future__ import annotations

from typing import Any

import pytest

from dpo.console.config import Calibration, Configuration
from dpo.console.document import CONSOLE_SCHEMA


def source(
    source_id: str,
    *,
    labels: list[str] | None = None,
    share: float = 0.05,
    presence: float = 0.6,
    confidence: float = 0.8,
    energy: float = 0.5,
) -> dict[str, Any]:
    return {
        "id": source_id,
        "labels": labels or [source_id.title()],
        "prose": source_id.replace("_", " "),
        "share": share,
        "presence": presence,
        "confidence": confidence,
        "energy": energy,
    }


def shot(shot_id: str, start_ms: int, end_ms: int, sources: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "shot_id": shot_id,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "raw_caption": "Traffic passes while a siren sounds somewhere behind.",
        "default_caption": "A siren over passing traffic.",
        "sources": sources,
    }


@pytest.fixture
def configuration() -> Configuration:
    return Configuration(
        study_id="street2026", corpus_id="amsterdam", provisional_salience=False, calibration=Calibration()
    )


@pytest.fixture
def document(configuration: Configuration) -> dict[str, Any]:
    """One clip, two shots. The first shot's sources cross; the second's do not."""
    return {
        "schema": CONSOLE_SCHEMA,
        "session_id": "street-console",
        "config": configuration.artifact(),
        "clips": [
            {
                "clip_id": "amsterdam_006",
                "opening": {"grain": "itemized", "alpha": 0.5, "admitted": None},
                "shots": [
                    shot(
                        "s1",
                        0,
                        4200,
                        [
                            # Seen but barely heard, against heard but barely seen:
                            # a pair that crosses somewhere inside the axis.
                            source("idling", share=0.30, presence=0.95, confidence=0.9, energy=0.10),
                            source("siren", share=0.002, presence=0.30, confidence=0.9, energy=0.95),
                            source("footsteps", labels=["Footsteps", "Speech"], share=0.05, presence=0.7),
                        ],
                    ),
                    shot(
                        "s2",
                        4200,
                        9000,
                        [source("traffic", share=0.20, presence=0.9, confidence=0.8, energy=0.7)],
                    ),
                ],
            }
        ],
    }
