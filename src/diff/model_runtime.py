"""What both runners share: the loaded-model record, refusals, chat rendering, RNG.

Kept apart so the Qwen baseline never imports the Trida runner (or the reverse)
and so these pieces can be tested without a model. torch is imported inside the
functions that need it, so importing this module costs nothing outside the
generator environment.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from diff.model_profile import ModelProfile, ProfileMismatch
from diff.rewrite_prompts import RenderedPrompt


class RefusedCall(ValueError):
    """A model call the protocol forbids; raised before the model is touched."""


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    device: str
    profile: ModelProfile
    loaded_code: dict[str, str] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    token_ids: dict[str, int] = field(default_factory=dict)

    @property
    def special_ids(self) -> frozenset[int]:
        return frozenset(
            int(token_id) for token_id, added in self.tokenizer.added_tokens_decoder.items() if added.special
        )

    def decode(self, token_ids: list[int]) -> str:
        return str(self.tokenizer.decode(token_ids, skip_special_tokens=False))


def chat_token_ids(tokenizer: Any, prompt: RenderedPrompt) -> list[int]:
    """The prompt as the model's own chat template tokenizes it: system turn, user turn, generation cue."""
    ids = tokenizer.apply_chat_template(prompt.messages(), add_generation_prompt=True, tokenize=True)
    return [int(token) for token in ids]


def checked_token_ids(
    tokenizer: Any, expected: Mapping[str, int], names: Mapping[str, str]
) -> dict[str, int]:
    """Confirm the pinned tokenizer maps each named token to the protocol's ID; never patch.

    ``names`` maps a protocol name to the token string, ``expected`` maps the
    same name to the ID the protocol records.
    """
    found = {name: int(tokenizer.convert_tokens_to_ids(token)) for name, token in names.items()}
    problems = [
        f"{names[name]} is {found[name]} in the tokenizer, but the protocol records {expected[name]}"
        for name in names
        if found[name] != expected[name]
    ]
    if problems:
        raise ProfileMismatch(problems)
    return found


def rng_snapshot(device: str) -> tuple[Any, Any]:
    """CPU and (if used) CUDA RNG states, so a call can be shown to consume no randomness (AC-07)."""
    import torch

    cuda = (
        torch.cuda.get_rng_state(device) if device.startswith("cuda") and torch.cuda.is_available() else None
    )
    return torch.get_rng_state(), cuda


def rng_unchanged(before: tuple[Any, Any], after: tuple[Any, Any]) -> bool:
    import torch

    cpu_same = bool(torch.equal(before[0], after[0]))
    if before[1] is None or after[1] is None:
        cuda_same = before[1] is None and after[1] is None
    else:
        cuda_same = bool(torch.equal(before[1], after[1]))
    return cpu_same and cuda_same
