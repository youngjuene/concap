"""Audio caption contract: compliance checks and the visual-claim screen.

An audio caption must not assert visual evidence. The lexicon below flags
sight, color, and appearance vocabulary for the audit queue; it is a recall aid
for human auditors and the candidate audit, never a semantic verdict.
"""

from __future__ import annotations

from dpo.contracts.captions import ComplianceReport, check_caption
from dpo.contracts.study_contract import CaptionContract, ContractError

# Visual-evidence vocabulary that an audio-only caption cannot support.
VISUAL_CLAIM_LEXICON = frozenset(
    {
        "appears",
        "blue",
        "bright",
        "brightly",
        "camera",
        "color",
        "colored",
        "colorful",
        "dark",
        "displayed",
        "frame",
        "glances",
        "green",
        "image",
        "looks",
        "onscreen",
        "photo",
        "picture",
        "red",
        "scene",
        "screen",
        "see",
        "seen",
        "shiny",
        "shown",
        "shows",
        "sign",
        "stares",
        "video",
        "visible",
        "visibly",
        "visual",
        "watches",
        "watching",
        "wearing",
        "white",
        "yellow",
    }
)


def check_audio_caption(text: str, contract: CaptionContract) -> ComplianceReport:
    if contract.track != "audio":
        raise ContractError("check_audio_caption requires an audio caption contract")
    return check_caption(text, contract, cross_modal_lexicon=VISUAL_CLAIM_LEXICON)
