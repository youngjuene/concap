"""A source-backed questionnaire keeps typed answers and page progress atomic."""

from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from dpo.regen.app import build_app
from dpo.regen.regeneration import RegenTemplateWriter


def study_config(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "consent_text": "테스트용 동의서: 실제 참가자 모집에 사용하지 않습니다.",
        "debrief_text": "테스트용 디브리핑입니다.",
        "practice": {
            "reviewed": True,
            "frames": [{"still": "practice.png", "at_ms": 0}],
            "sound_options": [{"value": "practice_a", "label": "연습 선택지 A"}],
        },
        "clips": {
            key: {
                "options_status": "reviewed",
                "stimulus_status": "reviewed",
                "condition": {"audiovisual_congruence": "congruent", "description_depth": "shallow"},
                "second_caption_strategy": "maintain",
                "options": {
                    "A1s": [
                        {"value": "traffic", "label": "차량 소리"},
                        {"value": "bird", "label": "새 소리"},
                    ],
                    "R2": [
                        {"value": "traffic", "label": "차량 소리", "correct": True},
                        {"value": "bird", "label": "새 소리", "correct": False},
                    ],
                },
                "source_families": {"traffic": "things", "bird": "animal"},
            }
            for key in document["segments"]
        },
    }


def setup(
    document: dict[str, Any], media_dir: Path, tmp_path: Path, *, configured: bool = True
) -> tuple[Any, TestClient, dict[str, Any]]:
    Image.new("RGB", (64, 64), "#d2b48c").save(media_dir / "practice.png")
    app = build_app(
        document,
        media_dir,
        tmp_path / "out",
        RegenTemplateWriter(),
        derive=False,
        questionnaire_config=study_config(document) if configured else None,
    )
    client = TestClient(app, client=("127.0.0.1", 40000))
    client.headers["x-study-request"] = "1"
    response = client.post("/api/questionnaire/session", json={})
    assert response.status_code == 200, response.text
    return app, client, response.json()


def submit(client: TestClient, state: dict[str, Any], action: str, **data: Any) -> dict[str, Any]:
    response = client.post(
        f"/api/questionnaire/{action}",
        json={
            "key": str(uuid.uuid4()),
            "revision": state["revision"],
            "data": {"page": state["page"], **data},
        },
    )
    assert response.status_code == 200, response.text
    return dict(response.json())


def values(state: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in state["items"]:
        if item["type"] == "rating":
            result[item["id"]] = 0
        elif item["type"] == "visual":
            result[item["id"]] = [{"frame_id": "0", "x": 0.5, "y": 0.5}]
        elif item["type"] == "multi":
            result[item["id"]] = ["memory_unknown"]
        else:
            result[item["id"]] = item["options"][0]["value"]
    return result


def watch(client: TestClient, state: dict[str, Any]) -> dict[str, Any]:
    duration = state["media"]["duration_ms"]
    for sequence, position in enumerate(range(0, duration + 1, 1000)):
        # Patch the playback clock only; CookieJar expiration uses real wall time.
        with patch("dpo.regen.playback.time", SimpleNamespace(time=lambda tick=100 + position / 1000: tick)):
            state = submit(
                client,
                state,
                "playback",
                position_ms=position,
                sequence=sequence,
                playing=position < duration,
                hidden=False,
            )
    return submit(client, state, "view-ended")


def test_complete_workbook_pages_persist_zero_both_observations_and_shuffled_order(
    document: Any, media_dir: Path, tmp_path: Path
) -> None:
    app, client, state = setup(document, media_dir, tmp_path)
    assert state["page"] == "P0" and not state["setup_missing"]
    state = submit(client, state, "intro-complete", consented=True)
    state = submit(client, state, "practice-complete", answers=values(state))
    pages = []
    while state["page"] != "handoff":
        pages.append(state["page"])
        if state["page"] in {"P1", "P6"}:
            state = watch(client, state)
        elif state["page"] == "generate":
            state = submit(client, state, "generate")
        else:
            before = state
            restored = client.get("/api/questionnaire/state").json()
            assert [item["id"] for item in before["items"]] == [item["id"] for item in restored["items"]]
            assert all("source_cells" not in item for item in state["items"])
            assert all("correct" not in o for item in state["items"] for o in item.get("options", []))
            state = submit(client, state, "submit", answers=values(state))
    export = client.get("/api/questionnaire/export").json()["state"]
    assert len(export["sheet_responses"]) == 18
    assert sum(len(row["answers"]) for row in export["sheet_responses"]) == 68
    assert len(export["sheet_viewings"]) == 4
    assert len(export["sheet_practice"]) == 1 and export["sheet_practice"][0]["analysis_excluded"]
    for page in ("P2", "P3", "P4", "P5", "P7", "P8", "P9", "P10", "P11"):
        assert pages.count(page) == 2
    assert all(row["answers"]["F1"] == 0 for row in export["sheet_responses"] if row["page"] == "P4")
    assert "instrument" not in export


def test_missing_materials_and_skipping_cannot_advance(
    document: Any, media_dir: Path, tmp_path: Path
) -> None:
    _, client, state = setup(document, media_dir, tmp_path, configured=False)
    assert state["setup_missing"]
    result = client.post(
        "/api/questionnaire/intro-complete",
        json={
            "key": "intro",
            "revision": state["revision"],
            "data": {"page": "P0", "consented": True},
        },
    )
    assert result.status_code == 409
    assert client.get("/api/questionnaire/state").json()["page"] == "P0"


def test_required_zero_rating_and_idempotent_submission(
    document: Any, media_dir: Path, tmp_path: Path
) -> None:
    _, client, state = setup(document, media_dir, tmp_path)
    state = submit(client, state, "intro-complete", consented=True)
    state = submit(client, state, "practice-complete", answers=values(state))
    result = client.post(
        "/api/questionnaire/view-ended",
        json={
            "key": "skip",
            "revision": state["revision"],
            "data": {"page": "P1"},
        },
    )
    assert result.status_code == 409
    state = watch(client, state)
    body = {"key": "once", "revision": state["revision"], "data": {"page": "P2", "answers": values(state)}}
    first = client.post("/api/questionnaire/submit", json=body)
    assert first.status_code == 200
    assert client.post("/api/questionnaire/submit", json=body).status_code == 200
    body["data"]["answers"]["V1"][0]["x"] = 0.2
    assert client.post("/api/questionnaire/submit", json=body).status_code == 409
    export = client.get("/api/questionnaire/export").json()["state"]
    assert len(export["sheet_responses"]) == 1


def test_opposite_track_uses_reviewed_first_visual_and_sound_combination(
    document: Any, media_dir: Path, tmp_path: Path
) -> None:
    from copy import deepcopy

    app, client, state = setup(document, media_dir, tmp_path)
    config = study_config(document)
    authored = deepcopy(document["segments"]["A"]["prepared_track"])
    for cue in authored:
        cue["text"] = {"ko": "확정된 반대 조건의 새 소리."}
    config["clips"]["A"].update(
        second_caption_strategy="opposite",
        second_tracks={
            f"{visual}|{sound}": authored
            for visual in ("building", "person", "unclassified")
            for sound in ("traffic", "bird")
        },
    )
    app.state.sheet_questionnaire.config = config
    state = client.get("/api/questionnaire/state").json()
    state = submit(client, state, "intro-complete", consented=True)
    state = submit(client, state, "practice-complete", answers=values(state))
    state = watch(client, state)
    state = submit(client, state, "submit", answers={"V1": [{"frame_id": "0", "x": 0.2, "y": 0.2}]})
    for _ in range(3):
        state = submit(client, state, "submit", answers=values(state))
    assert state["page"] == "generate"
    state = submit(client, state, "generate")
    assert state["page"] == "P6"
    assert all(cue["text"] == "확정된 반대 조건의 새 소리." for cue in state["media"]["captions"])
    exported = client.get("/api/questionnaire/export").json()["state"]
    result = exported["sheet_regenerations"]["A"]
    assert result["selection_key"] == "person|traffic"
    assert result["strategy"] == "opposite" and result["writer"] == "authored-opposite"


def test_formal_frames_cannot_silently_be_used_as_practice(
    document: Any, media_dir: Path, tmp_path: Path
) -> None:
    app, client, _ = setup(document, media_dir, tmp_path)
    app.state.sheet_questionnaire.config["practice"]["frames"][0]["still"] = document["segments"]["A"][
        "frames"
    ][0]["still"]
    state = client.get("/api/questionnaire/state").json()
    assert any("연습 이미지" in error for error in state["setup_missing"])
