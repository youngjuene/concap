"""Regression contracts from the real GPU QA, without claiming audio ground truth."""

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

from dpo.regen.study_schema import caption_instruction


def render_result(raw: str, settings: dict[str, Any]) -> dict[str, Any]:
    from dpo.regen.caption_controls import render_result as render

    return render(raw, settings)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("물가에서 사람들의 발소리가 들린다.", {"fallback": True}),
        (
            "No sounds are audible in the current excerpt.",
            {"fallback": False, "text": "뚜렷하게 들리는 소리가 없다."},
        ),
        (
            '{"categories":["human"],"qualities":["soft"]}',
            {"fallback": False, "text": "작은 사람 소리가 들린다."},
        ),
        (
            ["물가에서 사람들의 발소리가 들린다.", '{"categories":["human"],"qualities":["soft"]}'],
            {"fallback": False, "text": "작은 사람 소리가 들린다."},
        ),
    ],
)
def test_worker_rejects_control_leak_and_localizes_silence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raw: str | list[str], expected: dict[str, Any]
) -> None:
    from dpo.regen.study_worker import worker

    responses = iter(raw if isinstance(raw, list) else [raw, raw])

    class Adapter:
        def __init__(self, **kwargs: object) -> None:
            pass

        def generate_stimulus(self, *args: object, **kwargs: object) -> str:
            return next(responses)

        def _require_loaded(self) -> tuple[None, None]:
            return None, None

    module = ModuleType("dpo.models.gemma4.adapter")
    module.GemmaCaptionAdapter = Adapter  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr("dpo.regen.study_worker.signal.signal", lambda *args: None)
    audio = tmp_path / "excerpt.wav"
    audio.touch()
    requests = iter(
        [{**spec(), "excerpt": str(audio), "instruction": "QA contract", "fallback": "대체 자막"}, None]
    )
    replies: list[dict[str, Any]] = []

    class Pipe:
        def recv(self) -> object:
            return next(requests)

        def send(self, value: dict[str, Any]) -> None:
            replies.append(value)

    repo = Path(__file__).resolve().parents[4]
    worker(
        cast(Any, Pipe()),
        {
            "backend_config": str(repo / "configs/gemma4/e4b-audio.toml"),
            "contract": str(repo / "configs/study/street-audio.toml"),
        },
    )
    assert {key: replies[0][key] for key in expected} == expected
    assert replies[0]["raw"] == (raw[-1] if isinstance(raw, list) else raw)
    if isinstance(raw, list):
        assert len(replies[0]["attempts"]) == 2
        assert "structured" in replies[0]["attempts"][0]["validation_error"]
        assert "validation_error" not in replies[0]["attempts"][1]


def spec(texture: float = 0.8, context: float = 0.2, language: str = "ko") -> dict[str, Any]:
    return {"axes": {"texture": texture, "context": context}, "language": language}


def test_low_context_does_not_accept_the_observed_location_leak() -> None:
    with pytest.raises(ValueError, match="structured"):
        render_result("물가에서 사람들의 발소리가 들린다.", spec())


def test_low_context_renders_only_model_selected_categories_and_acoustics() -> None:
    raw = json.dumps({"categories": ["human", "things"], "qualities": ["soft", "rhythmic"]})
    result = render_result(raw, spec())
    assert result["text"] == "작고 규칙적인 사람 소리와 교통·기계 소리가 들린다."
    assert result["rendering"]["categories"] == ["human", "things"]
    assert result["rendering"]["qualities"] == ["soft", "rhythmic"]


def test_zero_texture_omits_acoustics_and_records_omissions() -> None:
    raw = json.dumps({"categories": ["human"], "qualities": ["loud", "rough"]})
    result = render_result(raw, spec(texture=0))
    assert result["text"] == "사람 소리가 들린다."
    assert result["rendering"]["omitted_qualities"] == ["loud", "rough"]


@pytest.mark.parametrize(
    "bad",
    [
        {"categories": ["물가에서"], "qualities": []},
        {"categories": ["human"], "qualities": ["물가에서"]},
        {"categories": ["human"], "qualities": [], "location": "물가에서"},
        {"categories": ["none", "human"], "qualities": []},
        {"categories": ["human", "human"], "qualities": []},
        {"categories": [], "qualities": []},
        {"categories": "human", "qualities": []},
    ],
)
def test_untrusted_free_text_cannot_escape_low_context_rendering(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        render_result(json.dumps(bad, ensure_ascii=False), spec())


@pytest.mark.parametrize(
    "raw",
    [
        "No sounds are audible in the excerpt.",
        "No sounds are audible in the current excerpt.",
    ],
)
@pytest.mark.parametrize("context", [0.2, 0.5, 1.0])
def test_observed_english_silence_preserves_meaning_in_korean(raw: str, context: float) -> None:
    result = render_result(raw, spec(context=context))
    assert result["text"] == "뚜렷하게 들리는 소리가 없다."
    assert result["normalization"] == "localized_no_audible_event"


def test_silence_normalization_does_not_swallow_other_events() -> None:
    with pytest.raises(ValueError, match="language"):
        render_result("No sounds are audible except footsteps.", spec(context=0.5))


def test_high_context_preserves_a_valid_model_sentence() -> None:
    raw = "물가에서 잔잔한 물소리가 들린다."
    assert render_result(raw, spec(context=1)) == {"text": raw}


def test_actual_high_context_visual_only_result_is_not_a_sound_caption() -> None:
    with pytest.raises(ValueError, match="auditory"):
        render_result("물가 길 옆 건물에서 물을 바라보는 장면입니다.", spec(texture=0, context=1))


def test_low_context_english_uses_same_control_contract() -> None:
    raw = '{"categories":["natural"],"qualities":["soft"]}'
    assert render_result(raw, spec(language="en"))["text"] == "Soft nature sounds are audible."


def test_existing_family_labels_are_normalized_without_losing_raw_selection() -> None:
    raw = '{"categories":["people"],"qualities":["intermittent"]}'
    result = render_result(raw, spec())
    assert result["text"] == "간헐적인 사람 소리가 들린다."
    assert result["rendering"]["categories"] == ["human"]
    assert result["rendering"]["raw_categories"] == ["people"]


def test_low_context_prompt_has_one_unambiguous_output_format() -> None:
    profile = {"template": "Write sound captions in Korean.", "language": "ko"}
    cue = {"evidence": "People beside water", "start_ms": 0, "end_ms": 5000}
    low = caption_instruction(profile, spec()["axes"], cue)
    high = caption_instruction(profile, spec(context=1)["axes"], cue)
    assert "categories" in low and "qualities" in low
    assert "One readable sentence" not in low
    assert "One readable sentence" in high
    assert "한국어" in high
