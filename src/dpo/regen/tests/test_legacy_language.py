"""The legacy language fix is independently testable without the v2 study."""

from dpo.regen.captions import Cue
from dpo.regen.config import Calibration, Configuration
from dpo.regen.regeneration import RegenRequestBuilder, Report


def test_actual_legacy_instruction_uses_participant_language() -> None:
    config = Configuration(study_id="test", corpus_id="test", calibration=Calibration(languages=("en", "ko")))
    default, korean = RegenRequestBuilder(config), RegenRequestBuilder(config, "ko")
    request = korean.request("clip", Cue(0, 0, 2500, {"en": "A sound."}), 4, Report((), ()), None)
    assert default.instruction(request) == korean.instruction(request)
    assert "Write it in Korean" in default.instruction(request)
