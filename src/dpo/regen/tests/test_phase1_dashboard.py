"""Phase1 observational metrics share an authenticated local API and CSV cohort."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from dpo.regen.dashboard import build_dashboard_app
from dpo.regen.dashboard_data import DashboardData
from dpo.regen.study_store import StudyStore

PRIVATE = "private-source-field-never-export"


def fixture(
    root: Path,
    session_id: str = "participant-analysis",
    *,
    historical: bool = False,
    qa_only: Any = False,
) -> str:
    store = StudyStore(root / "viewing/study.sqlite3")
    token = store.create("calibration", "ko")
    records = []
    for round_number, sound, area in [(1, "engine", 100), (2, "car", 250)]:
        visual = "V1" if round_number == 1 else "V2"
        auditory = "A1s" if round_number == 1 else "A2s"
        common = {
            "clip_index": 0,
            "clip_id": "clip-a",
            "instrument_hash": "instrument-a",
            "view_id": f"{session_id}-0-{'P1' if round_number == 1 else 'P6'}",
            "condition": "first" if round_number == 1 else "second",
            "stimulus_assignment": {"audiovisual_congruence": "incongruent", "description_depth": "shallow"},
            "caption_strategy": "opposite",
            "analysis_excluded": False,
            "submitted_at": 1791144000,
        }
        point = {"frame_id": "0", "x": 0.5, "y": 0.5}
        metric = {
            "selected_mask_id": "road_transport",
            "label": "Vehicles",
            "frame_id": "0",
            "selected_mask_area_px": area,
            "frame_area_px": 1000,
            "selected_mask_area_ratio": area / 1000,
            "frame_sha256": "f" * 64,
            "mask_sha256": str(round_number) * 64,
            "matching_rule": "smallest-containing-mask/document-order",
            "threshold": 127,
            "candidate_masks": [],
            "status": "matched",
            "private": PRIVATE,
        }
        records.append(
            {
                **common,
                "page": "P2" if round_number == 1 else "P7",
                "answers": {visual: [point]},
                "items": [{"id": visual, "type": "visual", "text": "Visual observation"}],
                "matches": [
                    {"frame": 0, "x": 0.5, "y": 0.5, "object_id": "road_transport", "label": "Vehicles"}
                ],
                **({} if historical else {"visual_metrics": [metric]}),
            }
        )
        options = [{"value": sound, "label": sound}, {"value": "speech", "label": "Speech"}]
        records.append(
            {
                **common,
                "page": "P3" if round_number == 1 else "P8",
                "answers": {auditory: sound},
                "items": [
                    {"id": auditory, "type": "choice", "text": "Sound observation", "options": options}
                ],
            }
        )

    def initialize(state: dict[str, Any]) -> None:
        state.update(
            session_id=session_id,
            stage="sheet-questionnaire",
            sheet_protocol="dpo.sheet-questionnaire/v1",
            sheet_page="P9",
            sheet_responses=records,
            sheet_viewings=[],
            sheet_clip_order=["A"],
            sheet_config={
                "qa_only": qa_only,
                "not_for_recruitment": True,
                "consent_text": PRIVATE,
                "clips": {
                    "A": {
                        "source_families": {"engine": "things", "car": "things", "speech": "human"},
                        "analysis_codebook": {
                            "version": "test-v1",
                            "av_relations": {"road_transport": {"engine": 1, "car": 1}},
                        },
                    }
                },
            },
        )

    store.mutate(token, "init", 0, "{}", initialize)
    return token


def test_phase1_analysis_api_csv_and_read_only_provenance(tmp_path: Path) -> None:
    token = fixture(tmp_path)
    app = build_dashboard_app(tmp_path / "legacy", tmp_path / "viewing")
    with TestClient(app, base_url="http://localhost", client=("127.0.0.1", 40000)) as client:
        response = client.get("/api/phase1-analysis")
        assert response.status_code == 200, response.text
        data = response.json()
        assert len(data["observations"]) == 2 and len(data["pairs"]) == 1
        pair = data["pairs"][0]
        assert pair["area_delta_pp"] == 15
        assert pair["type_changed"] == 1 and pair["family_changed"] == 0
        assert pair["menu_changed"] is True
        assert data["observations"][0]["av_assigned_condition"] == "incongruent"
        assert data["observations"][0]["av_selection_relation"] == 1
        assert token not in response.text and PRIVATE not in response.text
        assert response.headers["cache-control"] == "no-store"
        for kind, expected in [("observations", 2), ("pairs", 1)]:
            downloaded = client.get(f"/api/export/phase1-{kind}.csv")
            assert downloaded.status_code == 200
            rows = list(csv.DictReader(io.StringIO(downloaded.text.lstrip("\ufeff"))))
            assert len(rows) == expected
            assert token not in downloaded.text and PRIVATE not in downloaded.text
            if kind == "pairs":
                assert float(rows[0]["area_delta_pp"]) == 15
        assert client.post("/api/phase1-analysis", json={}).status_code == 405


def test_phase1_analysis_filters_use_same_denominators_as_csv(tmp_path: Path) -> None:
    fixture(tmp_path)
    fixture(tmp_path, "qa-extra")
    data = DashboardData(tmp_path / "legacy", tmp_path / "viewing")
    assert len(data.phase1_analysis()["pairs"]) == 1
    assert len(data.phase1_analysis(include_qa=True)["pairs"]) == 2
    assert len(data.phase1_analysis(include_qa=True, participant="qa-extra")["pairs"]) == 1
    assert not data.phase1_analysis(language="en")["pairs"]
    assert not data.phase1_analysis(status="complete")["pairs"]
    filtered = data.phase1_analysis(clip_id="not-selected")
    assert not filtered["pairs"] and not filtered["observations"]
    assert filtered["available_clips"] == ["clip-a"]


def test_historical_missing_geometry_is_not_backfilled_or_zero(tmp_path: Path) -> None:
    fixture(tmp_path, historical=True)
    data = DashboardData(tmp_path / "legacy", tmp_path / "viewing").phase1_analysis()
    assert all(row["selected_mask_area_ratio"] is None for row in data["observations"])
    assert data["pairs"][0]["area_delta_pp"] is None
    assert data["pairs"][0]["type_changed"] == 1


@pytest.mark.parametrize("qa_flag", [True, False, "true", 1, None])
def test_explicit_qa_config_excludes_random_session_ids_consistently(tmp_path: Path, qa_flag: Any) -> None:
    fixture(tmp_path, "random-public-session", qa_only=qa_flag)
    app = build_dashboard_app(tmp_path / "legacy", tmp_path / "viewing")
    expected = 0 if qa_flag is True else 1
    with TestClient(app, base_url="http://localhost", client=("127.0.0.1", 40000)) as client:
        assert len(client.get("/api/overview").json()["participants"]) == expected
        assert len(client.get("/api/phase1-analysis").json()["pairs"]) == expected
        included = client.get("/api/overview?include_qa=true").json()["participants"]
        assert len(included) == 1 and included[0]["qa"] is (qa_flag is True)
        assert len(client.get("/api/phase1-analysis?include_qa=true").json()["pairs"]) == 1
        for kind, multiplier in [("observations", 2), ("pairs", 1)]:
            for query, count in [("", expected), ("?include_qa=true", 1)]:
                body = client.get(f"/api/export/phase1-{kind}.csv{query}").text
                assert len(list(csv.DictReader(io.StringIO(body.lstrip("\ufeff"))))) == count * multiplier


def test_new_analysis_and_csv_routes_retain_local_boundary(tmp_path: Path) -> None:
    app = build_dashboard_app(tmp_path / "legacy", tmp_path / "viewing")
    with TestClient(app, base_url="http://localhost", client=("203.0.113.10", 40000)) as client:
        for path in [
            "/api/phase1-analysis",
            "/api/export/phase1-observations.csv",
            "/api/export/phase1-pairs.csv",
        ]:
            assert client.get(path).status_code == 403
