"""Visual caption contract: compliance checks and the audio-claim screen.

A visual caption must not assert audible evidence. The lexicon below flags
sound, speech, and music vocabulary for the audit queue; it is a recall aid for
human auditors and the candidate audit, never a semantic verdict.
"""

from __future__ import annotations

from dpo.contracts.captions import ComplianceReport, check_caption
from dpo.contracts.study_contract import CaptionContract, ContractError

# Audible-evidence vocabulary that a visual-only caption cannot support.
AUDIO_CLAIM_LEXICON = frozenset(
    {
        "audio",
        "audible",
        "bang",
        "bark",
        "barking",
        "barks",
        "beep",
        "beeping",
        "buzz",
        "buzzing",
        "chatter",
        "cheering",
        "chirp",
        "chirping",
        "clang",
        "crash",
        "cries",
        "crying",
        "hear",
        "heard",
        "honk",
        "honking",
        "hum",
        "humming",
        "laughter",
        "loud",
        "loudly",
        "melody",
        "music",
        "noise",
        "noisy",
        "quiet",
        "quietly",
        "rattle",
        "ringing",
        "rings",
        "roar",
        "roaring",
        "rumble",
        "said",
        "says",
        "scream",
        "screaming",
        "shout",
        "shouting",
        "silence",
        "silent",
        "sings",
        "singing",
        "siren",
        "sound",
        "sounds",
        "speech",
        "squeak",
        "talking",
        "talks",
        "thud",
        "whistle",
        "whistling",
        "yelling",
        "yells",
    }
)


def check_visual_caption(text: str, contract: CaptionContract) -> ComplianceReport:
    if contract.track != "visual":
        raise ContractError("check_visual_caption requires a visual caption contract")
    return check_caption(text, contract, cross_modal_lexicon=AUDIO_CLAIM_LEXICON)
