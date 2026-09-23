"""End-to-end count matrix for flexible Phase 1 clips and Phase 2 videos."""

from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from dpo.regen.app import build_app
from dpo.regen.config import SOUND_FAMILIES, Calibration, Configuration
from dpo.regen.continuation import ViewingConfig
from dpo.regen.regeneration import RegenTemplateWriter
from dpo.regen.study_schema import VIDEO_ITEMS, digest
from dpo.regen.tests.test_qa_validation import _segment
from dpo.regen.tests.test_study import form_values, make_media


def _answers(blocks: list[dict[str, Any]]) -> dict[str, int]:
    return {item["id"]: 4 for block in blocks for item in block["items"]}


def _short_document(count: int) -> dict[str, Any]:
    names = [f"clip_{index + 1}" for index in range(count)]
    return {
        "schema": "dpo.caption-regen/v4",
        "session_id": f"count-matrix-{count}",
        "clip_order": names,
        "config": Configuration(
            study_id="count-matrix",
            corpus_id="short-clips",
            calibration=Calibration(cue_slots=4, minimum_points=1),
        ).artifact(),
        "segments": {name: _segment(name) for name in names},
    }


def _stage_short_masks(document: dict[str, Any], media: Path) -> None:
    for segment in document["segments"].values():
        for frame in segment["frames"]:
            mask = media / frame["objects"][0]["mask"]
            mask.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (8, 8), 255).save(mask)


def _add_long_video(root: Path, manifest: dict[str, Any], index: int, colour: str) -> None:
    video = root / f"long-{index}.webm"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c={colour}:s=160x90:r=2",
            "-t",
            "300",
            "-c:v",
            "libvpx",
            "-b:v",
            "30k",
            "-y",
            str(video),
        ],
        check=True,
    )
    manifest["viewing_videos"].append(
        {
            "id": f"long-{index}",
            "title": f"Long video {index + 1}",
            "status": "ready",
            "video": video.name,
            "audio": "audio.wav",
            "duration_ms": 300000,
            "cues": [
                {
                    "start_ms": start,
                    "end_ms": start + 5000,
                    "evidence": f"Authored evidence for long video {index + 1} at {start} ms.",
                    "fallback": {"en": f"Caption for video {index + 1} at {start} ms."},
                }
                for start in range(0, 300000, 5000)
            ],
        }
    )


@pytest.fixture(scope="module")
def max_viewing_media(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    root = tmp_path_factory.mktemp("count-matrix-media")
    manifest = make_media(root)
    _add_long_video(root, manifest, 3, "0x704080")
    _add_long_video(root, manifest, 4, "0x807040")
    return root, manifest


def _manifest_for(source: dict[str, Any], count: int) -> dict[str, Any]:
    manifest = deepcopy(source)
    manifest["viewing_videos"] = manifest["viewing_videos"][:count]
    return manifest


def _post(client: TestClient, path: str, payload: dict[str, Any], expected: int = 200) -> dict[str, Any]:
    response = client.post(path, json=payload)
    assert response.status_code == expected, response.text
    return dict(response.json())


def _play_short(
    client: TestClient,
    participant: str,
    detail: dict[str, Any],
    *,
    clip_index: int,
    step: str,
    clock: list[float],
) -> None:
    for sequence, position in enumerate(range(0, int(detail["duration_ms"]) + 1, 1000)):
        clock[0] += 1
        _post(
            client,
            "/api/playback",
            {
                "participant": participant,
                "clip_index": clip_index,
                "step": step,
                "context_id": detail["event_context"]["context_id"],
                "sequence": sequence,
                "position_ms": position,
                "playing": True,
                "hidden": False,
            },
        )


def _record_viewing(client: TestClient, participant: str, clip_index: int, step: str) -> dict[str, Any]:
    return _post(
        client,
        "/api/viewing",
        {
            "participant": participant,
            "clip_index": clip_index,
            "step": step,
            "started_at": f"2026-09-22T00:{clip_index:02d}:00Z",
            "ended_at": f"2026-09-22T00:{clip_index:02d}:10Z",
        },
    )


def _submit_survey(
    client: TestClient, participant: str, page: str, blocks: list[dict[str, Any]], clip_index: int
) -> dict[str, Any]:
    return _post(
        client,
        "/api/survey",
        {
            "participant": participant,
            "page": page,
            "clip_index": clip_index,
            "responses": _answers(blocks),
            "entered_at": f"2026-09-22T01:{clip_index:02d}:00Z",
            "submitted_at": f"2026-09-22T01:{clip_index:02d}:10Z",
        },
    )


def _complete_phase1(client: TestClient, short_count: int, clock: list[float]) -> dict[str, Any]:
    session = _post(client, "/api/session", {})
    participant = str(session["participant"])
    assert session["clip_count"] == short_count
    assert session["protocol_version"] == 3

    for clip_index, segment in enumerate(session["clip_order"]):
        prepared = client.get(f"/api/step/view_prepared?participant={participant}").json()
        assert prepared["clip_index"] == clip_index
        assert prepared["segment"] == segment
        _play_short(client, participant, prepared, clip_index=clip_index, step="view_prepared", clock=clock)
        _record_viewing(client, participant, clip_index, "view_prepared")

        art = client.get(f"/api/step/art?participant={participant}").json()
        assert [block["id"] for block in art["blocks"]] == ["art"]
        _submit_survey(client, participant, "art", art["blocks"], clip_index)

        visual = _post(
            client,
            "/api/visual",
            {
                "participant": participant,
                "clip_index": clip_index,
                "points": [{"frame": 0, "x": 0.5, "y": 0.5}],
            },
        )
        assert visual["clip_index"] == clip_index
        heard = {family: family == "things" for family in SOUND_FAMILIES}
        auditory = _post(
            client,
            "/api/auditory",
            {"participant": participant, "clip_index": clip_index, "heard": heard},
        )
        assert auditory["clip_index"] == clip_index

        regenerated = _post(client, "/api/regenerate", {"participant": participant, "clip_index": clip_index})
        assert regenerated["clip_index"] == clip_index
        updated = client.get(f"/api/step/view_regenerated?participant={participant}").json()
        assert updated["segment"] == segment
        _play_short(
            client,
            participant,
            updated,
            clip_index=clip_index,
            step="view_regenerated",
            clock=clock,
        )
        _record_viewing(client, participant, clip_index, "view_regenerated")

        survey = client.get(f"/api/step/survey?participant={participant}").json()
        assert [block["id"] for block in survey["blocks"]] == ["art", "caption"]
        result = _submit_survey(client, participant, "survey", survey["blocks"], clip_index)
        assert result["step"] == ("view_prepared" if clip_index + 1 < short_count else "overall")

    overall = client.get(f"/api/step/overall?participant={participant}").json()
    assert [block["id"] for block in overall["blocks"]] == ["prss"]
    done = _submit_survey(client, participant, "overall", overall["blocks"], short_count - 1)
    assert done["step"] == "done"
    return session


def _study_action(
    client: TestClient,
    base: str,
    state: dict[str, Any],
    action: str,
    data: dict[str, Any] | None = None,
    *,
    expected: int = 200,
) -> dict[str, Any]:
    result = _post(
        client,
        f"{base}api/study/{action}",
        {"revision": state["revision"], "key": f"{action}-{state['revision']}", "data": data or {}},
        expected,
    )
    return result if expected == 200 else state


def _complete_phase2(
    client: TestClient, base: str, state: dict[str, Any], long_count: int, clock: list[float]
) -> dict[str, Any]:
    state = _study_action(client, base, state, "start-viewing")
    assert state["stage"] == "watch"
    assert state["viewing_total"] == long_count
    assert state["survey_flow"] == "long-video-surveys/v2"

    for index in range(long_count):
        video_id = f"long-{index}"
        assert state["video"]["id"] == video_id
        _study_action(client, base, state, "video-ended", {"video_id": video_id}, expected=400)
        for sequence, position in enumerate(range(0, 300001, 5000)):
            clock[0] += 5
            state = _study_action(
                client,
                base,
                state,
                "playback",
                {"video_id": video_id, "position_ms": position, "sequence": sequence, "playing": True},
            )
        state = _study_action(client, base, state, "video-ended", {"video_id": video_id})
        assert state["stage"] == "video-survey"
        assert [item["id"] for item in state["items"]] == [item["id"] for item in VIDEO_ITEMS]
        answers = form_values(state["items"])
        _study_action(client, base, state, "draft", {"video_id": "wrong", "answers": answers}, expected=409)
        state = _study_action(client, base, state, "draft", {"video_id": video_id, "answers": answers})
        assert state["draft"] == {"video_id": video_id, "answers": answers}
        state = _study_action(client, base, state, "video-survey", {"video_id": video_id, "answers": answers})
        if index + 1 < long_count:
            assert state["stage"] == "break"
            state = _study_action(client, base, state, "continue")

    assert state["stage"] == "final"
    final_answers = form_values(state["items"])
    final_ids = [item["id"] for item in state["items"]]
    assert final_ids[:4] == ["control_texture", "control_context", "effort", "timing"]
    assert any(item_id.startswith("prss_") for item_id in final_ids)
    state = _study_action(client, base, state, "final-survey", final_answers)
    assert state["stage"] == "done"
    return state


def _client_for_counts(
    tmp_path: Path, max_viewing_media: tuple[Path, dict[str, Any]], short_count: int, long_count: int
) -> tuple[TestClient, Any, Path, dict[str, Any]]:
    document = _short_document(short_count)
    media = tmp_path / "short-media"
    _stage_short_masks(document, media)
    viewing_root, source_manifest = max_viewing_media
    manifest = _manifest_for(source_manifest, long_count)
    manifest_path = tmp_path / "viewing.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    out = tmp_path / "phase1-out"
    app = build_app(
        document,
        media,
        out,
        RegenTemplateWriter(),
        derive=False,
        viewing=ViewingConfig(manifest_path, viewing_root, tmp_path / "phase2-out"),
    )
    client = TestClient(app)
    client.headers["x-study-request"] = "1"
    return client, app, out, document


@pytest.mark.parametrize(("short_count", "long_count"), [(1, 1), (2, 3), (4, 5)])
def test_short_long_count_matrix_end_to_end(
    tmp_path: Path,
    max_viewing_media: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    short_count: int,
    long_count: int,
) -> None:
    clock = [1000.0]
    monkeypatch.setattr("dpo.regen.study_api.time.time", lambda: clock[0])
    client, app, out, document = _client_for_counts(tmp_path, max_viewing_media, short_count, long_count)

    session = _complete_phase1(client, short_count, clock)
    participant = str(session["participant"])
    viewings = [json.loads(row) for row in (out / f"viewings-{participant}.jsonl").read_text().splitlines()]
    responses = [json.loads(row) for row in (out / f"responses-{participant}.jsonl").read_text().splitlines()]
    snapshot = json.loads((out / f"snapshot-{participant}.json").read_text())

    assert len(viewings) == short_count * 2
    assert len(responses) == short_count * 2 + 1
    assert [row["segment"] for row in viewings[::2]] == session["clip_order"]
    assert [row["scope"] for row in viewings] == ["clip_original", "clip_updated"] * short_count
    assert [row["scope"] for row in responses] == ["clip_original", "clip_updated"] * short_count + [
        "overall"
    ]
    assert responses[-1]["clip_index"] is None
    assert set(snapshot["visual_by_segment"]) == set(session["clip_order"])
    assert set(snapshot["auditory_by_segment"]) == set(session["clip_order"])
    assert set(snapshot["regenerated_tracks"]) == set(session["clip_order"])

    handoff = _post(
        client,
        "/api/continuation",
        {"participant": participant, "token": session["viewing_token"]},
    )
    base = str(handoff["url"])
    token = str(session["viewing_token"])
    frozen = app.state.continuation.store.state(token)
    assert frozen["stage"] == "ready"
    assert frozen["profile"]["sample_count"] == short_count
    assert len(frozen["calibration_source"]["viewings"]) == short_count * 2
    assert len(frozen["calibration_source"]["responses"]) == short_count * 2 + 1
    assert [row["clip"] for row in frozen["observations"]] == [
        document["segments"][segment]["clip_id"] for segment in session["clip_order"]
    ]
    assert frozen["profile"]["calibration_source_hash"] == digest(frozen["calibration_source"])

    state = client.get(f"{base}api/study/state").json()
    state = _complete_phase2(client, base, state, long_count, clock)
    stored = app.state.continuation.store.state(token)
    assert stored["stage"] == "done"
    assert len(stored["completions"]) == long_count
    assert len(stored["video_surveys"]) == long_count
    assert [record["video_id"] for record in stored["completions"]] == [
        f"long-{index}" for index in range(long_count)
    ]
    assert [record["video_id"] for record in stored["video_surveys"]] == [
        f"long-{index}" for index in range(long_count)
    ]
    assert stored["final_survey"]["profile_hash"] == frozen["profile"]["hash"]
    assert state["stage"] == "done"


def test_handoff_rejects_wrong_short_clip_identity(
    tmp_path: Path, max_viewing_media: tuple[Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [1000.0]
    monkeypatch.setattr("dpo.regen.study_api.time.time", lambda: clock[0])
    client, _, out, _ = _client_for_counts(tmp_path, max_viewing_media, 1, 1)
    session = _complete_phase1(client, 1, clock)
    participant = str(session["participant"])
    path = out / f"viewings-{participant}.jsonl"
    rows = [json.loads(row) for row in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["clip_id"] = "wrong-clip"
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")

    response = client.post(
        "/api/continuation",
        json={"participant": participant, "token": session["viewing_token"]},
    )

    assert response.status_code == 409
    assert response.json()["error"] == "Calibration viewing does not match its clip and condition."
