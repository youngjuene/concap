from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from fastapi.testclient import TestClient
from PIL import Image

from dpo.regen.app import build_app
from dpo.regen.config import SOUND_FAMILIES, Calibration, Configuration, Scale
from dpo.regen.continuation import ViewingConfig
from dpo.regen.document import REGEN_SCHEMA
from dpo.regen.items import AUTHORED, Block, Item, ItemSet
from dpo.regen.regeneration import RegenTemplateWriter
from dpo.regen.study_schema import digest
from dpo.regen.tests.test_qa_validation import _segment
from dpo.regen.tests.test_study import make_media


def _client(
    tmp_path: Path,
    *,
    configuration: Configuration | None = None,
    items: ItemSet | None = None,
    viewing: ViewingConfig | None = None,
) -> tuple[TestClient, Path, Any]:
    configuration = configuration or Configuration(
        study_id="hierarchy", corpus_id="short-clips", calibration=Calibration(cue_slots=4, minimum_points=1)
    )
    document: dict[str, Any] = {
        "schema": REGEN_SCHEMA,
        "session_id": "hierarchy",
        "config": configuration.artifact(),
        "segments": {"A": _segment("A"), "B": _segment("B")},
    }
    media = tmp_path / "media"
    for segment in document["segments"].values():
        for frame in segment["frames"]:
            mask = media / frame["objects"][0]["mask"]
            mask.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (8, 8), 255).save(mask)
    out = tmp_path / "out"
    app = build_app(document, media, out, RegenTemplateWriter(), items=items, derive=False, viewing=viewing)
    return TestClient(app), out, app


def _answers(blocks: list[dict[str, Any]]) -> dict[str, int]:
    return {item["id"]: 4 for block in blocks for item in block["items"]}


def _submit(
    client: TestClient, participant: str, page: str, blocks: list[dict[str, Any]], clip: int
) -> dict[str, Any]:
    response = client.post(
        "/api/survey",
        json={
            "participant": participant,
            "page": page,
            "clip_index": clip,
            "responses": _answers(blocks),
            "entered_at": f"{page}-{clip}-in",
            "submitted_at": f"{page}-{clip}-out",
        },
    )
    assert response.status_code == 200, response.text
    return cast(dict[str, Any], response.json())


def _finish_clip(client: TestClient, participant: str, step: str, clip: int) -> dict[str, Any]:
    response = client.post(
        "/api/viewing",
        json={
            "participant": participant,
            "step": step,
            "clip_index": clip,
            "started_at": f"{step}-{clip}-start",
            "ended_at": f"{step}-{clip}-end",
        },
    )
    assert response.status_code == 200, response.text
    return cast(dict[str, Any], response.json())


def _observe(client: TestClient, participant: str, clip: int) -> None:
    visual = client.post(
        "/api/visual",
        json={"participant": participant, "clip_index": clip, "points": [{"frame": 0, "x": 0.5, "y": 0.5}]},
    )
    assert visual.status_code == 200, visual.text
    heard = {family: family == "things" for family in SOUND_FAMILIES}
    auditory = client.post(
        "/api/auditory",
        json={"participant": participant, "clip_index": clip, "heard": heard},
    )
    assert auditory.status_code == 200, auditory.text


def _regenerate(client: TestClient, participant: str, clip: int) -> dict[str, Any]:
    response = client.post("/api/regenerate", json={"participant": participant, "clip_index": clip})
    assert response.status_code == 200, response.text
    return cast(dict[str, Any], response.json())


def _custom_items() -> ItemSet:
    return ItemSet(
        provenance=AUTHORED,
        blocks={
            "art": Block("art", {"en": "Scene"}, (Item("art_clear", {"en": "The scene was clear."}),)),
            "caption": Block(
                "caption",
                {"en": "Captions"},
                (Item("caption_useful", {"en": "The captions were useful."}),),
            ),
            "prss": Block(
                "prss",
                {"en": "Custom restorative scale"},
                (Item("custom_prss_away", {"en": "This place gave me distance from routine."}),),
            ),
        },
    )


def _complete_short_hierarchy(client: TestClient) -> dict[str, Any]:
    session = cast(dict[str, Any], client.post("/api/session", json={}).json())
    participant = str(session["participant"])
    assert session["flow_version"] == "clip-caption-prss-v2"
    assert session["clip_index"] == 0
    assert session["clip_count"] == 2

    first_segment = session["assignment"]["prepared_segment"]
    second_segment = session["assignment"]["regenerated_segment"]
    for clip, segment in enumerate((first_segment, second_segment)):
        detail = client.get(f"/api/step/view_prepared?participant={participant}").json()
        assert detail["clip_index"] == clip
        assert detail["segment"] == segment
        _finish_clip(client, participant, "view_prepared", clip)
        art = client.get(f"/api/step/art?participant={participant}").json()
        assert [block["id"] for block in art["blocks"]] == ["art"]
        _submit(client, participant, "art", art["blocks"], clip)
        _observe(client, participant, clip)
        _regenerate(client, participant, clip)
        updated = client.get(f"/api/step/view_regenerated?participant={participant}").json()
        assert updated["segment"] == segment
        _finish_clip(client, participant, "view_regenerated", clip)
        survey = client.get(f"/api/step/survey?participant={participant}").json()
        assert [block["id"] for block in survey["blocks"]] == ["art", "caption"]
        result = _submit(client, participant, "survey", survey["blocks"], clip)
        assert result["step"] == ("view_prepared" if clip == 0 else "overall")

    overall = client.get(f"/api/step/overall?participant={participant}").json()
    assert [block["id"] for block in overall["blocks"]] == ["prss"]
    done = _submit(client, participant, "overall", overall["blocks"], 1)
    assert done["step"] == "done"
    return session


def test_short_clip_flow_repeats_original_updated_surveys_then_overall_prss(tmp_path: Path) -> None:
    client, out, _ = _client(tmp_path)
    session = _complete_short_hierarchy(client)
    participant = str(session["participant"])
    first_segment = str(session["assignment"]["prepared_segment"])
    second_segment = str(session["assignment"]["regenerated_segment"])

    viewings = [
        json.loads(line)
        for line in (out / f"viewings-{participant}.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    responses = [
        json.loads(line)
        for line in (out / f"responses-{participant}.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["segment"] for row in viewings] == [
        first_segment,
        first_segment,
        second_segment,
        second_segment,
    ]
    assert [row["scope"] for row in viewings] == [
        "clip_original",
        "clip_updated",
        "clip_original",
        "clip_updated",
    ]
    assert [row["scope"] for row in responses] == [
        "clip_original",
        "clip_updated",
        "clip_original",
        "clip_updated",
        "overall",
    ]
    snapshot = json.loads((out / f"snapshot-{participant}.json").read_text(encoding="utf-8"))
    assert set(snapshot["regenerated_tracks"]) == {first_segment, second_segment}
    assert set(snapshot["visual_by_segment"]) == {first_segment, second_segment}


def test_versioned_short_clip_posts_reject_stale_clip_index(tmp_path: Path) -> None:
    client, _, _ = _client(tmp_path)
    session = client.post("/api/session", json={}).json()
    participant = session["participant"]
    _finish_clip(client, participant, "view_prepared", 0)
    art = client.get(f"/api/step/art?participant={participant}").json()
    _submit(client, participant, "art", art["blocks"], 0)
    _observe(client, participant, 0)
    _regenerate(client, participant, 0)
    _finish_clip(client, participant, "view_regenerated", 0)
    survey = client.get(f"/api/step/survey?participant={participant}").json()
    _submit(client, participant, "survey", survey["blocks"], 0)

    stale = client.post(
        "/api/viewing",
        json={
            "participant": participant,
            "step": "view_prepared",
            "clip_index": 0,
            "started_at": "stale-start",
            "ended_at": "stale-end",
        },
    )
    assert stale.status_code == 409
    assert stale.json()["clip_index"] == 1
    stale_stage = client.get(f"/api/step/survey?participant={participant}")
    assert stale_stage.status_code == 409
    assert stale_stage.json()["step"] == "view_prepared"
    assert stale_stage.json()["clip_index"] == 1
    assert stale_stage.json()["flow_version"] == "clip-caption-prss-v2"


def test_versionless_legacy_snapshot_keeps_original_survey_blocks(tmp_path: Path) -> None:
    client, out, _ = _client(tmp_path)
    session = client.post("/api/session", json={}).json()
    participant = session["participant"]
    snapshot_path = out / f"snapshot-{participant}.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot.pop("flow_version")
    snapshot.pop("clip_index")
    snapshot.pop("clip_count")
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    resumed = client.post("/api/session", json={"participant": participant}).json()
    assert resumed["flow_version"] == "legacy-ab-v1"
    assert resumed["clip_count"] == 1
    _finish_clip(client, participant, "view_prepared", 0)
    art = client.get(f"/api/step/art?participant={participant}").json()
    assert [block["id"] for block in art["blocks"]] == ["art"]
    _submit(client, participant, "art", art["blocks"], 0)
    _observe(client, participant, 0)
    _regenerate(client, participant, 0)
    _finish_clip(client, participant, "view_regenerated", 0)
    survey = client.get(f"/api/step/survey?participant={participant}").json()
    assert [block["id"] for block in survey["blocks"]] == ["art", "caption", "prss"]


def test_continuation_freezes_custom_prss_items_and_scale(tmp_path: Path) -> None:
    items = _custom_items()
    scale = Scale(
        points=5,
        anchors=("Low", "High"),
        anchors_by_language={"en": ("Low", "High"), "ko": ("낮음", "높음")},
    )
    configuration = Configuration(
        study_id="custom-handoff",
        corpus_id="short-clips",
        calibration=Calibration(cue_slots=4, minimum_points=1, scale=scale),
    )
    viewing_media = tmp_path / "viewing-media"
    manifest = make_media(viewing_media)
    manifest_path = tmp_path / "viewing.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    viewing = ViewingConfig(manifest_path, viewing_media, tmp_path / "viewing-out")
    client, _, app = _client(tmp_path, configuration=configuration, items=items, viewing=viewing)
    session = _complete_short_hierarchy(client)
    participant = str(session["participant"])

    linked = app.state.continuation.handoff(participant, session["viewing_token"], "en")
    state = app.state.continuation.store.state(linked["token"])
    frozen = state["instrument"]
    assert frozen["prss"]["items_digest"] == items.digest
    assert frozen["prss"]["items_provenance"] == AUTHORED
    assert frozen["prss"]["scale"] == scale.record()
    assert frozen["items"]["overall"][-1]["id"] == "custom_prss_away"
    assert frozen["items"]["overall"][-1]["points"] == 5
    assert frozen["items"]["overall"][-1]["anchors"] == ["Low", "High"]
    assert state["profile"]["sample_count"] == 2
    assert state["profile"]["calibration_source_hash"] == digest(state["calibration_source"])


def test_continuation_freeze_repairs_preupgrade_waiting_link_flow(tmp_path: Path) -> None:
    viewing_media = tmp_path / "viewing-media"
    manifest = make_media(viewing_media)
    manifest_path = tmp_path / "viewing.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    viewing = ViewingConfig(manifest_path, viewing_media, tmp_path / "viewing-out")
    client, _, app = _client(tmp_path, viewing=viewing)
    session = _complete_short_hierarchy(client)
    participant = str(session["participant"])
    token = str(session["viewing_token"])
    store = app.state.continuation.store
    current = store.state(token)

    def preupgrade_waiting(state: dict[str, Any]) -> None:
        assert state["stage"] == "awaiting-calibration"
        state.pop("flow_version", None)
        state.pop("video_surveys", None)
        instrument = state.get("instrument")
        if isinstance(instrument, dict):
            instrument["flow"] = "long-video-surveys/v1"

    store.mutate(token, "fixture-preupgrade-waiting", current["revision"], "fixture", preupgrade_waiting)

    linked = app.state.continuation.handoff(participant, token, "en")
    state = store.state(linked["token"])
    assert state["stage"] == "ready"
    assert state["flow_version"] == 2
    assert state["video_surveys"] == []
    assert state["instrument"]["flow"] == "long-video-surveys/v2"
