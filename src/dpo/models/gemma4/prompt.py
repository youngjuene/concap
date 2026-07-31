"""Chat-message construction for the caption tracks on the Gemma backend.

The system prompt is the caption contract's prompt verbatim — the contract is
the single source of the instruction, its hash is pinned, and both tracks use
identical preprocessing apart from the media payload. Media enters as typed
chat content parts; the visual track attaches frames only and the audio track
attaches audio only.
"""

from __future__ import annotations

from typing import Any

from dpo.contracts.study_contract import CaptionContract, ContractError

CHAT_TEMPLATE_KWARGS: dict[str, bool] = {"enable_thinking": False}


def template_kwargs(contract: CaptionContract) -> dict[str, Any]:
    """Chat-template kwargs for one track: thinking off, frames per the contract.

    Scoring and generation must render a prompt the same way, so both go
    through here; the frame budget only exists on the visual track.
    """
    kwargs: dict[str, Any] = dict(CHAT_TEMPLATE_KWARGS)
    if contract.video_frames:
        kwargs["num_frames"] = contract.video_frames
    return kwargs


def system_message(contract: CaptionContract) -> dict[str, Any]:
    return {"role": "system", "content": contract.prompt}


def user_media_message(contract: CaptionContract, *, media_reference: str) -> dict[str, Any]:
    """One user turn carrying exactly the track's media modality."""
    if contract.track == "visual":
        part: dict[str, Any] = {"type": "video", "video": media_reference}
    elif contract.track == "audio":
        part = {"type": "audio", "audio": media_reference}
    else:  # pragma: no cover - CaptionContract validates track
        raise ContractError(f"unknown track {contract.track!r}")
    return {"role": "user", "content": [part]}


def prompt_messages(contract: CaptionContract, *, media_reference: str) -> list[dict[str, Any]]:
    return [system_message(contract), user_media_message(contract, media_reference=media_reference)]


def assistant_message(completion: str) -> list[dict[str, str]]:
    return [{"role": "assistant", "content": completion.strip()}]


def stimulus_messages(
    instruction: str, *, audio_reference: str, video_reference: str | None = None
) -> list[dict[str, Any]]:
    """Messages for human-study stimulus generation, optionally seeing the frame.

    Deliberately NOT track-bound. A congruency ladder needs the model to see
    what the caption should attribute a sound to, which the audio track's own
    messages can never carry — so this is the one builder that may put video and
    audio in the same turn, and it exists only to construct what a participant
    reads. Nothing scored, trained on, or compared against preference data may
    come through here; those go through prompt_messages and stay isolated.
    """
    parts: list[dict[str, Any]] = []
    if video_reference is not None:
        parts.append({"type": "video", "video": video_reference})
    parts.append({"type": "audio", "audio": audio_reference})
    return [{"role": "system", "content": instruction}, {"role": "user", "content": parts}]
