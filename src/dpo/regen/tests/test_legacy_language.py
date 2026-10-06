"""The legacy language fix is independently testable without the v2 study."""

from dpo.caption.writer import CaptionRequest
from dpo.regen.captions import Cue
from dpo.regen.config import Calibration, Configuration
from dpo.regen.copy import strings_for
from dpo.regen.regeneration import SAME_CLIP_FLOW, RegenRequestBuilder, Report, regenerate


def test_actual_legacy_instruction_uses_participant_language() -> None:
    config = Configuration(study_id="test", corpus_id="test", calibration=Calibration(languages=("en", "ko")))
    default, korean = RegenRequestBuilder(config), RegenRequestBuilder(config, "ko")
    request = korean.request("clip", Cue(0, 0, 2500, {"en": "A sound."}), 4, Report((), ()), None)
    assert default.instruction(request) == korean.instruction(request)
    assert "Write it in Korean" in default.instruction(request)


def test_actual_regen_instruction_names_legacy_different_segment() -> None:
    config = Configuration(study_id="test", corpus_id="test")
    builder = RegenRequestBuilder(config)
    request = builder.request("clip", Cue(0, 0, 2500, {"en": "A sound."}), 1, Report(("Bus",), ()), None)
    instruction = builder.instruction(request)

    assert "watched a different segment" in instruction
    assert "watched this same clip" not in instruction
    assert request.settings_key == "0|en|Bus|"


def test_actual_regen_instruction_names_versioned_same_clip_and_cache_key() -> None:
    config = Configuration(study_id="test", corpus_id="test")
    builder = RegenRequestBuilder(config, flow_version=SAME_CLIP_FLOW)
    request = builder.request("clip", Cue(0, 0, 2500, {"en": "A sound."}), 1, Report(("Bus",), ()), None)
    instruction = builder.instruction(request)

    assert "watched this same clip" in instruction
    assert "watched a different segment" not in instruction
    assert request.settings_key == f"0|en|Bus||{SAME_CLIP_FLOW}"


def test_regenerate_dispatcher_and_logged_prompt_share_versioned_instruction() -> None:
    config = Configuration(study_id="test", corpus_id="test")
    slots = (Cue(0, 0, 2500, {"en": "A sound."}),)

    class DispatcherWriter:
        identity = "dispatcher"

        def __init__(self) -> None:
            self.seen: list[str] = []

        def instruction(self, request: CaptionRequest) -> str:
            text = RegenRequestBuilder(config).instruction(request)
            self.seen.append(text)
            return text

        def write(self, request: CaptionRequest) -> str:
            self.instruction(request)
            return "bus"

    legacy = DispatcherWriter()
    legacy_result = regenerate(
        legacy, config, clip_id="clip", slots=slots, fallback=slots, report=Report(("Bus",), ("Bus",))
    )
    same_clip = DispatcherWriter()
    same_clip_result = regenerate(
        same_clip,
        config,
        clip_id="clip",
        slots=slots,
        fallback=slots,
        report=Report(("Bus",), ("Bus",)),
        settings={"flow_version": SAME_CLIP_FLOW},
    )

    assert legacy_result.prompts == tuple(legacy.seen)
    assert same_clip_result.prompts == tuple(same_clip.seen)
    assert "watched a different segment" in legacy_result.prompts[0]
    assert "watched this same clip" in same_clip_result.prompts[0]
    assert legacy_result.settings == {}
    assert same_clip_result.settings["flow_version"] == SAME_CLIP_FLOW


def test_actual_legacy_chrome_localises_page_landmarks() -> None:
    assert strings_for("en")["rail"]["label"] == "Progress"
    assert strings_for("ko")["rail"]["label"] == "진행 상황"
    assert strings_for("en")["network"]["retry"].startswith("Could not save")
    assert strings_for("ko")["network"]["retry"].startswith("저장하지 못했습니다")
