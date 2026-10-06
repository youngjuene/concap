"""Questionnaire values affect the actual Phase 2 prompt without becoming detail ratings."""

from __future__ import annotations

import copy
import json
import signal
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from dpo.regen.config import Scale
from dpo.regen.continuation import ViewingConfig
from dpo.regen.items import ItemSet, load_items
from dpo.regen.study_personalization import questionnaire_profile
from dpo.regen.study_schema import caption_instruction, compile_profile, digest
from dpo.regen.study_worker import worker
from dpo.regen.tests.test_study import make_media
from dpo.regen.tests.test_survey_hierarchy import _client, _complete_short_hierarchy


def records(
    items: ItemSet | None = None, points: int = 7
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    items = items or load_items()
    responses, viewings = [], []
    for index in range(2):
        for page, condition, scope, blocks in [
            ("art", "prepared", "clip_original", ["art"]),
            ("survey", "regenerated", "clip_updated", ["art", "caption"]),
        ]:
            view = {
                "view_id": f"{index}-{page}",
                "clip_id": f"clip-{index}",
                "clip_index": index,
                "condition": condition,
            }
            viewings.append(view)
            responses.append(
                {
                    **view,
                    "page": page,
                    "scope": scope,
                    "items_digest": items.digest,
                    "items_provenance": items.provenance,
                    "responses": {
                        item.id: (points + 1) // 2 for name in blocks for item in items.blocks[name].items
                    },
                }
            )
    responses.append(
        {
            "view_id": "overall",
            "page": "overall",
            "scope": "overall",
            "items_digest": items.digest,
            "items_provenance": items.provenance,
            "responses": {item.id: (points + 1) // 2 for item in items.blocks["prss"].items},
        }
    )
    return responses, viewings


def personalized(responses: list[dict[str, Any]], viewings: list[dict[str, Any]]) -> dict[str, Any]:
    questionnaire = questionnaire_profile(responses, viewings, load_items(), Scale(), "en")
    return compile_profile({}, [], "en", questionnaire=questionnaire)


@pytest.mark.parametrize(
    "page,item_id",
    [
        ("art", "art_away_1"),
        ("survey", "art_away_1"),
        ("survey", "caption_credible_1"),
        ("survey", "caption_restorative_3"),
        ("overall", "prss_away_1"),
    ],
)
def test_changing_each_rating_family_changes_prompt_and_cache_material(page: str, item_id: str) -> None:
    responses, viewings = records()
    before = personalized(responses, viewings)
    next(row for row in responses if row["page"] == page)["responses"][item_id] = 7
    after = personalized(responses, viewings)
    cue = {"start_ms": 0, "end_ms": 5000, "evidence": "A vehicle passes."}
    axes = {"texture": 0.5, "context": 0.5}
    assert before["hash"] != after["hash"]
    assert before["template"] != after["template"]
    assert digest(caption_instruction(before, axes, cue)) != digest(caption_instruction(after, axes, cue))
    assert before["defaults"] == after["defaults"] == axes


@pytest.mark.parametrize("points", [5, 7])
def test_reverse_coding_and_scales_drive_conciseness_guidance(points: int) -> None:
    responses, viewings = records(points=points)
    for row in responses:
        if row["page"] == "survey":
            row["responses"]["caption_restorative_3"] = points
    result = questionnaire_profile(responses, viewings, load_items(), Scale(points=points), "en")
    assert result["normalized_scores"]["caption_non_distraction"] == 0
    assert any("compact syntax" in line for line in result["guidance"])
    assert result["items"]["caption_restorative_3"]["reverse"] is True
    assert result["records"][1]["answers"]["caption_restorative_3"] == points


def test_paired_art_change_uses_matching_clip_only() -> None:
    responses, viewings = records()
    for row in responses:
        if row["page"] == "art":
            row["responses"] = dict.fromkeys(row["responses"], 7)
        elif row["page"] == "survey":
            row["responses"].update({key: 1 for key in row["responses"] if key.startswith("art_")})
    result = questionnaire_profile(responses, viewings, load_items(), Scale(), "en")
    assert result["normalized_scores"]["paired_art_change"] == -1
    assert any("does not establish causation" in line for line in result["guidance"])
    # Same numeric ART responses across different footage must not be paired.
    for row, view in zip(responses[:4], viewings, strict=True):
        if row["page"] == "survey":
            row["clip_id"] = view["clip_id"] = "different-" + row["clip_id"]
    result = questionnaire_profile(responses, viewings, load_items(), Scale(), "en")
    assert "paired_art_change" not in result["normalized_scores"]


def test_custom_wording_is_preserved_without_reusing_builtin_semantics() -> None:
    original = load_items()
    changed = replace(
        original.blocks["caption"].items[-1], text={"en": "Custom question", "ko": "사용자 질문"}
    )
    block = replace(original.blocks["caption"], items=(changed,))
    items = replace(original, blocks={**original.blocks, "caption": block})
    responses, viewings = records(items)
    result = questionnaire_profile(responses, viewings, items, Scale(), "ko")
    assert result["items"][changed.id]["text"] == "사용자 질문"
    assert changed.id in result["unmapped_item_ids"]
    assert "caption_non_distraction" not in result["normalized_scores"]
    responses[1]["responses"][changed.id] = 7
    assert result["prompt"] != questionnaire_profile(responses, viewings, items, Scale(), "ko")["prompt"]


@pytest.mark.parametrize("damage", ["missing", "unknown", "range", "boolean", "wording", "clip", "duplicate"])
def test_invalid_or_mismatched_responses_are_not_silently_personalized(damage: str) -> None:
    responses, viewings = records()
    if damage == "missing":
        responses[0]["responses"].pop("art_away_1")
    elif damage == "unknown":
        responses[0]["responses"]["unknown"] = 4
    elif damage in ("range", "boolean"):
        responses[0]["responses"]["art_away_1"] = 8 if damage == "range" else True
    elif damage == "wording":
        responses[0]["items_digest"] = "other-instrument"
    elif damage == "clip":
        responses[0]["clip_id"] = "different-clip"
    else:
        responses.append(copy.deepcopy(responses[0]))
    with pytest.raises(ValueError):
        personalized(responses, viewings)


def test_live_controls_override_guidance_and_frozen_profile_is_not_mutated() -> None:
    responses, viewings = records()
    profile = personalized(responses, viewings)
    held = copy.deepcopy(profile)
    cue = {"start_ms": 0, "end_ms": 5000, "evidence": "A vehicle passes."}
    low = caption_instruction(profile, {"texture": 0, "context": 0}, cue)
    high = caption_instruction(profile, {"texture": 1, "context": 1}, cue)
    assert profile["questionnaire"]["prompt"] in low and profile["questionnaire"]["prompt"] in high
    assert "Use up to 0 supported acoustic descriptors" in low
    assert "Use up to 4 supported acoustic descriptors" in high
    assert "Current interactive controls override" in high
    assert profile == held


def test_low_and_high_ratings_produce_different_writing_guidance() -> None:
    responses, viewings = records()
    reverse = {item.id: item.reverse for block in load_items().blocks.values() for item in block.items}
    profiles = []
    for positive in (False, True):
        changed = copy.deepcopy(responses)
        for row in changed:
            if row["page"] != "art":
                row["responses"] = {key: 7 if positive != reverse[key] else 1 for key in row["responses"]}
        profiles.append(personalized(changed, viewings)["questionnaire"])
    low, high = profiles
    assert low["normalized_scores"]["caption_credibility"] == 0
    assert high["normalized_scores"]["caption_credibility"] == 1
    assert any("compact syntax" in line for line in low["guidance"])
    assert not any("compact syntax" in line for line in high["guidance"])
    assert low["guidance"] != high["guidance"]
    assert "clip-0" not in low["prompt"]
    assert low["records"][0]["clip_id"] == "clip-0"


def test_unversioned_different_clip_design_does_not_create_paired_art_effect() -> None:
    responses, viewings = records()
    legacy = [copy.deepcopy(responses[0]), copy.deepcopy(responses[3])]
    for row in legacy:
        row["scope"] = "view"
    legacy[1]["responses"].update(responses[4]["responses"])
    result = questionnaire_profile(legacy, viewings, load_items(), Scale(), "en")
    assert "paired_art_change" not in result["normalized_scores"]
    assert len(result["records"]) == 2
    assert "soundscape_restoration" in result["normalized_scores"]


def test_handoff_freezes_ratings_and_schedules_them_into_model_messages(
    tmp_path: Path, monkeypatch: Any
) -> None:
    media = tmp_path / "media"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(make_media(media)))
    client, _, app = _client(
        tmp_path / "short", viewing=ViewingConfig(manifest_path, media, tmp_path / "long")
    )
    session = _complete_short_hierarchy(client)
    continuation = app.state.continuation
    linked = continuation.handoff(session["participant"], session["viewing_token"], "en")
    store, token = continuation.store, linked["token"]
    frozen = store.state(token)
    profile = frozen["profile"]
    assert profile["personalization_source"] == "phase1_questionnaire"
    assert [len(row["answers"]) for row in profile["questionnaire"]["records"]] == [8, 14, 8, 14, 8]
    assert profile["hash"] == digest({key: value for key, value in profile.items() if key != "hash"})
    # An already-frozen session must resume without reconstructing its profile.
    monkeypatch.setattr("dpo.regen.continuation.questionnaire_profile", lambda *args: pytest.fail("refrozen"))
    assert continuation.handoff(session["participant"], token, "en")["url"] == linked["url"]
    assert store.state(token)["profile"] == profile

    # Use the actual HTTP scheduler and inspect the job it will deliver to Gemma.
    viewing = client
    viewing.headers["x-study-request"] = "1"
    viewing.cookies.set(continuation.app.state.cookie_name, token)
    response = viewing.post(
        linked["url"] + "api/study/start-viewing",
        json={"key": "start", "revision": frozen["revision"], "data": {}},
    )
    assert response.status_code == 200, response.text
    state = response.json()
    response = viewing.post(
        linked["url"] + "api/study/settings",
        json={
            "key": "detail",
            "revision": state["revision"],
            "data": {
                "video_id": state["video"]["id"],
                "texture": 1,
                "context": 0,
            },
        },
    )
    assert response.status_code == 200, response.text
    job = store.claim()
    assert job is not None
    spec = json.loads(job["spec"])
    assert profile["questionnaire"]["prompt"] in spec["instruction"]
    assert "Acoustic detail: 1.00/1. Source and scene detail: 0.00/1." in spec["instruction"]

    captured = []

    class Adapter:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def generate_stimulus(self, messages: Any, **kwargs: Any) -> str:
            captured.append(messages)
            return json.dumps({"categories": ["things"], "qualities": ["steady"]})

    class Pipe:
        def __init__(self) -> None:
            self.incoming = iter([spec, None])
            self.results: list[dict[str, Any]] = []

        def recv(self) -> Any:
            return next(self.incoming)

        def send(self, result: dict[str, Any]) -> None:
            self.results.append(result)

    monkeypatch.setattr("dpo.models.gemma4.adapter.GemmaCaptionAdapter", Adapter)
    repo = Path(__file__).resolve().parents[4]
    pipe = Pipe()
    previous = signal.getsignal(signal.SIGINT)
    try:
        worker(
            pipe,  # type: ignore[arg-type]
            {
                "backend_config": str(repo / "configs/gemma4/e4b-audio.toml"),
                "contract": str(repo / "configs/study/street-audio.toml"),
            },
        )
    finally:
        signal.signal(signal.SIGINT, previous)
    assert pipe.results[0]["fallback"] is False, pipe.results
    assert pipe.results[0]["text"] == "Steady traffic and machinery sounds are audible."
    assert pipe.results[0]["rendering"]["categories"] == ["things"]
    assert captured[0][0] == {"role": "system", "content": spec["instruction"]}
    assert pipe.results[0]["messages"] == captured[0]
