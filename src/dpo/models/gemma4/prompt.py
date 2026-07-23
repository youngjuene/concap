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
