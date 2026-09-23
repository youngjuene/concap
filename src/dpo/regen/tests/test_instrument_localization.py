from __future__ import annotations

import json
from pathlib import Path

import pytest

from dpo.regen.config import (
    DEFAULT_SCALE_ANCHORS,
    SCALE_SCHEMA,
    Calibration,
    ConfigError,
    Configuration,
    Scale,
    load_configuration,
)
from dpo.regen.items import load_items
from dpo.regen.study_instrument import (
    INSTRUMENT_PROVENANCE,
    INSTRUMENT_SCHEMA,
    instrument_snapshot,
    localized_pilot_artifact,
    prepare_pilot,
    rendered_items,
)
from dpo.regen.study_schema import FINAL_ITEMS


def test_prepared_pilot_preserves_source_and_refuses_overwrite(tmp_path: Path) -> None:
    from dpo.regen.document import REGEN_SCHEMA, load_regen_document
    from dpo.regen.tests.test_qa_validation import _segment

    source = tmp_path / "original.json"
    original = {
        "schema": REGEN_SCHEMA,
        "session_id": "pilot",
        "config": Configuration(study_id="pilot", corpus_id="test").artifact(),
        "segments": {"A": _segment("A"), "B": _segment("B")},
    }
    source.write_text(json.dumps(original))
    source_bytes = source.read_bytes()
    output = tmp_path / "prepared" / "regen.json"
    result = prepare_pilot(source, output)
    prepared = load_regen_document(output)
    assert source.read_bytes() == source_bytes
    assert prepared["segments"] == original["segments"]
    assert prepared["session_id"] != original["session_id"]
    assert result["fresh_output_required"] is True
    assert result["questionnaire_wording_changed"] is False
    assert prepared["config"]["calibration"]["scale"]["anchors_by_language"]["ko"]
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        prepare_pilot(source, output)
    assert output.read_bytes() == before


def test_korean_scale_is_served_by_actual_survey_api(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from dpo.regen.app import build_app
    from dpo.regen.document import REGEN_SCHEMA
    from dpo.regen.regeneration import RegenTemplateWriter
    from dpo.regen.tests.test_qa_validation import _segment

    config = Configuration(
        study_id="pilot", corpus_id="test", calibration=Calibration(languages=("en", "ko"))
    )
    segments = {"A": _segment("A"), "B": _segment("B")}
    for segment in segments.values():
        for track in ("prepared_track", "fallback_track"):
            for cue in segment[track]:
                cue["text"] = {"en": cue["text"], "ko": "차량 소리가 들립니다."}
    document = {
        "schema": REGEN_SCHEMA,
        "session_id": "api-pilot",
        "config": localized_pilot_artifact(config.artifact()),
        "segments": segments,
    }
    client = TestClient(build_app(document, tmp_path / "media", tmp_path / "out", RegenTemplateWriter()))
    participant = client.post("/api/session", json={}).json()["participant"]
    assert (
        client.post("/api/language", json={"participant": participant, "language": "ko"}).status_code == 200
    )
    response = client.post(
        "/api/viewing",
        json={
            "participant": participant,
            "step": "view_prepared",
            "clip_index": 0,
            "started_at": "t0",
            "ended_at": "t1",
        },
    )
    assert response.status_code == 200
    detail = client.get(f"/api/step/art?participant={participant}").json()
    assert detail["scale"]["anchors"] == ["전혀 그렇지 않다", "매우 그렇다"]
    assert detail["scale"]["schema"] == SCALE_SCHEMA


@pytest.mark.parametrize("pair", [["low"], ["low", "high", "extra"], [1, 5], "ab"])
def test_localized_scale_rejects_malformed_anchor_pairs(pair: object) -> None:
    raw = Configuration(study_id="test", corpus_id="test").artifact()
    raw["calibration"]["scale"]["anchors_by_language"] = {"ko": pair}
    with pytest.raises(ConfigError):
        load_configuration(raw)


def test_localized_scale_rejects_unknown_schema() -> None:
    raw = Configuration(study_id="test", corpus_id="test").artifact()
    raw["calibration"]["scale"]["schema"] = "dpo.caption-regen-scale/v999"
    with pytest.raises(ConfigError, match="schema"):
        load_configuration(raw)


def test_pilot_localization_preserves_range_and_refuses_unreviewed_custom_wording() -> None:
    raw = Configuration(
        study_id="test", corpus_id="test", calibration=Calibration(scale=Scale(points=5))
    ).artifact()
    assert localized_pilot_artifact(raw)["calibration"]["scale"]["points"] == 5
    raw["calibration"]["scale"]["anchors"] = ["Never", "Always"]
    with pytest.raises(ConfigError, match="Custom scale"):
        localized_pilot_artifact(raw)


def test_legacy_scale_artifact_round_trips_without_changing_identity() -> None:
    configuration = Configuration(
        study_id="legacy",
        corpus_id="same",
        calibration=Calibration(languages=("en", "ko"), scale=Scale(anchors=DEFAULT_SCALE_ANCHORS)),
    )
    artifact = configuration.artifact()

    assert artifact["calibration"]["scale"] == {"points": 7, "anchors": list(DEFAULT_SCALE_ANCHORS)}
    assert load_configuration(artifact).artifact() == artifact
    assert (
        Configuration(
            study_id="legacy",
            corpus_id="same",
            calibration=Calibration(languages=("en", "ko"), scale=Scale(anchors=DEFAULT_SCALE_ANCHORS)),
        ).hash
        == configuration.hash
    )


def test_localized_scale_anchors_are_versioned_and_config_hashed() -> None:
    english = DEFAULT_SCALE_ANCHORS
    korean = ("전혀 그렇지 않다", "매우 그렇다")
    first = Configuration(
        study_id="localized",
        corpus_id="same",
        calibration=Calibration(
            languages=("en", "ko"),
            scale=Scale(anchors=english, anchors_by_language={"en": english, "ko": korean}),
        ),
    )
    second = Configuration(
        study_id="localized",
        corpus_id="same",
        calibration=Calibration(
            languages=("en", "ko"),
            scale=Scale(anchors=english, anchors_by_language={"en": english, "ko": ("낮음", "높음")}),
        ),
    )

    scale = first.artifact()["calibration"]["scale"]
    assert scale["schema"] == SCALE_SCHEMA
    assert scale["anchors_by_language"]["ko"] == list(korean)
    assert first.hash != second.hash


def test_korean_survey_scale_selection_keeps_integer_response_contract() -> None:
    english = DEFAULT_SCALE_ANCHORS
    korean = ("전혀 그렇지 않다", "매우 그렇다")
    configuration = Configuration(
        study_id="localized",
        corpus_id="contract",
        calibration=Calibration(
            languages=("en", "ko"),
            scale=Scale(anchors=english, anchors_by_language={"en": english, "ko": korean}),
        ),
    )

    assert configuration.scale.anchors_for("ko") == korean
    assert configuration.scale.anchors_for("en") == english
    assert configuration.scale.accepts(4)
    assert not configuration.scale.accepts(True)
    responses = {item: 4 for item in load_items().item_ids("art")}
    assert all(configuration.scale.accepts(value) for value in responses.values())
    assert load_items().page_blocks("art")[0].record("ko")["items"][0]["text"].startswith("이 장면")


def test_viewing_korean_strings_cover_current_js_copy() -> None:
    strings: dict[str, str] = json.loads(
        (Path(__file__).parents[1] / "study-ko.json").read_text(encoding="utf-8")
    )
    required = {
        "Request failed",
        "Receipt: {id}",
        "The captions accurately described sounds I could hear.",
        "I could obtain the sound-source and scene detail I wanted using the controls.",
        "Please watch the full video before continuing. Replay any skipped sections.",
        "Your calibration is saved. The three viewing videos are not ready yet.",
        "Viewing captions are not prepared in your language yet.",
    }

    assert required <= set(strings)
    assert all(strings[key] != key for key in required)


def test_v2_instrument_snapshot_localizes_labels_without_changing_values() -> None:
    english = instrument_snapshot("en")
    korean = instrument_snapshot("ko")

    assert korean["schema"] == INSTRUMENT_SCHEMA
    assert korean["provenance"] == INSTRUMENT_PROVENANCE
    assert korean["hash"] != english["hash"]
    timing = next(item for item in korean["items"]["final"] if item["id"] == "timing")
    assert timing["options"][0] == "Fast enough"
    assert timing["option_labels"][0] == "충분히 빨랐다"
    control = next(item for item in korean["items"]["final"] if item["id"] == "control_context")
    assert control["na"] is True
    assert control["na_label"] == "해당 없음"


def test_v2_rendered_items_hash_changes_with_displayed_translation() -> None:
    translations = {
        "The captions accurately described sounds I could hear.": "자막 A",
        "Strongly disagree": "전혀 그렇지 않다",
        "Strongly agree": "매우 그렇다",
    }
    changed = {
        **translations,
        "The captions accurately described sounds I could hear.": "자막 B",
    }

    first = rendered_items(FINAL_ITEMS[:1], "ko", translations)
    second = rendered_items(FINAL_ITEMS[:1], "ko", changed)
    assert first[0]["text"] == "자막 A"
    assert second[0]["text"] == "자막 B"
    assert instrument_snapshot("ko", translations)["hash"] != instrument_snapshot("ko", changed)["hash"]


def test_localized_pilot_artifact_opts_in_with_new_identity() -> None:
    source = Configuration(
        study_id="regen-pilot",
        corpus_id="corpus",
        calibration=Calibration(languages=("en", "ko")),
    )
    localized = localized_pilot_artifact(source.artifact())

    assert localized["study_id"] == "regen-pilot-localized-scale-v1"
    assert localized["calibration"]["scale"]["schema"] == SCALE_SCHEMA
    assert localized["calibration"]["scale"]["anchors_by_language"]["ko"] == [
        "전혀 그렇지 않다",
        "매우 그렇다",
    ]
    assert load_configuration(localized).hash != source.hash
