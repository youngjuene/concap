from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from dpo.regen.app import build_app
from dpo.regen.config import Calibration, Configuration
from dpo.regen.document import REGEN_SCHEMA
from dpo.regen.regeneration import RegenTemplateWriter


def _track(prefix: str) -> list[dict[str, Any]]:
    return [
        {"index": index, "start_ms": index * 1000, "end_ms": (index + 1) * 1000, "text": f"{prefix} {index}"}
        for index in range(4)
    ]


def _segment(name: str) -> dict[str, Any]:
    return {
        "segment": name,
        "clip_id": f"clip_{name}",
        "video": f"{name}/clip.mp4",
        "audio": f"{name}/audio.wav",
        "duration_ms": 4000,
        "frames": [
            {
                "at_ms": 500 + index * 500,
                "still": f"{name}/frames/{index}.png",
                "objects": [{"id": "object", "label": "Object", "mask": f"{name}/masks/{index}/object.png"}],
            }
            for index in range(5)
        ],
        "stems": [
            {
                "id": "traffic",
                "label": "Traffic",
                "parent": "Sounds of things",
                "audio": f"{name}/stems/traffic.wav",
                "colour": "#3F83D1",
                "gain": 1.0,
                "waveform": [0.5] * 64,
            }
        ],
        "prepared_track": _track(f"Prepared {name}"),
        "fallback_track": _track(f"Fallback {name}"),
    }


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    configuration = Configuration(
        study_id="qa",
        corpus_id="validation",
        calibration=Calibration(cue_slots=4, minimum_points=1),
    )
    document = {
        "schema": REGEN_SCHEMA,
        "session_id": "qa-validation",
        "config": configuration.artifact(),
        "segments": {"A": _segment("A"), "B": _segment("B")},
    }
    return TestClient(build_app(document, tmp_path / "media", tmp_path / "out", RegenTemplateWriter()))


def _participant(client: TestClient) -> str:
    return str(client.post("/api/session", json={}).json()["participant"])


@pytest.mark.parametrize("step", [[], {}])
def test_viewing_rejects_non_string_steps(client: TestClient, step: object) -> None:
    response = client.post(
        "/api/viewing",
        json={"participant": _participant(client), "step": step, "started_at": "t0", "ended_at": "t1"},
    )

    assert response.status_code == 400
    assert response.json()["error"] == "step must be one of ['view_prepared', 'view_regenerated']"


@pytest.mark.parametrize("page", [[], {}])
def test_survey_rejects_non_string_pages(client: TestClient, page: object) -> None:
    response = client.post(
        "/api/survey",
        json={
            "participant": _participant(client),
            "page": page,
            "responses": {},
            "entered_at": "t0",
            "submitted_at": "t1",
        },
    )

    assert response.status_code == 400
    assert response.json()["error"] == "page must be one of ['art', 'survey']"
