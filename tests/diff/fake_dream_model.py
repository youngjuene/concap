"""Weight-free stand-ins for the Dream and Qwen call shapes, for the runner tests.

``FakeDream.diffusion_generate`` keeps Dream's schedule — linear timesteps
from 1 to eps, ``int(n_mask × (1 − s/t))`` positions per step and every
remaining one at the last step, one forward per step — and fills a scripted
text left to right. It is not the pinned ``generation_utils.py``: the CPU
tests against a vendored copy of that file follow a passing G0.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

MASK, ENDOFTEXT, IM_START, IM_END, BEGINOFTEXT = 151666, 151643, 151644, 151645, 151665
NEWLINE = 198


class FakeTokenizer:
    """Characters map to IDs below the special range; the template mimics Dream's chat format."""

    added_tokens_decoder = {
        token: SimpleNamespace(special=True) for token in (ENDOFTEXT, IM_START, IM_END, BEGINOFTEXT, MASK)
    }
    names = {
        "<|mask|>": MASK,
        "<|endoftext|>": ENDOFTEXT,
        "<|im_end|>": IM_END,
        "<|im_start|>": IM_START,
        "<|beginoftext|>": BEGINOFTEXT,
    }

    def __init__(self, *, prefix: tuple[int, ...] = ()) -> None:
        self.prefix = list(prefix)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [1000 + ord(character) for character in text]

    def apply_chat_template(
        self, messages: list[dict[str, str]], add_generation_prompt: bool, tokenize: bool
    ) -> list[int]:
        ids = list(self.prefix)
        for message in messages:
            ids += [
                IM_START,
                *self.encode(message["role"]),
                NEWLINE,
                *self.encode(message["content"]),
                IM_END,
                NEWLINE,
            ]
        if add_generation_prompt:
            ids += [IM_START, *self.encode("assistant"), NEWLINE]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(chr(token - 1000) if 1000 <= token < ENDOFTEXT else f"<{token}>" for token in ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.names[token]


class FakeDream(torch.nn.Module):
    def __init__(
        self,
        text: str,
        *,
        tokenizer: FakeTokenizer,
        terminator: int = ENDOFTEXT,
        leave_mask: bool = False,
        extra_forward: bool = False,
        use_rng: bool = False,
    ) -> None:
        super().__init__()
        self.text_ids = [*tokenizer.encode(text), terminator]
        self.leave_mask, self.extra_forward, self.use_rng = leave_mask, extra_forward, use_rng
        self.calls: list[dict[str, Any]] = []

    def forward(self, x: torch.Tensor, *args: Any) -> SimpleNamespace:
        return SimpleNamespace(logits=None)

    def _prepare_generation_config(self, generation_config: Any, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            **kwargs, bos_token_id=ENDOFTEXT, eos_token_id=ENDOFTEXT, pad_token_id=ENDOFTEXT
        )

    def diffusion_generate(self, inputs: torch.Tensor, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(dict(kwargs))
        steps, canvas, eps = int(kwargs["steps"]), int(kwargs["max_new_tokens"]), float(kwargs["eps"])
        prompt = inputs.shape[1]
        x = torch.cat([inputs, torch.full((1, canvas), MASK, dtype=torch.long)], dim=1)
        target = (self.text_ids + [ENDOFTEXT] * canvas)[:canvas]
        timesteps = torch.linspace(1, eps, steps + 1)
        history = []
        for index in range(steps):
            self(x)
            masked = [p for p in range(prompt, x.shape[1]) if int(x[0, p]) == MASK]
            t, s = timesteps[index], timesteps[index + 1]
            number = int(len(masked) * (1 - s / t)) if index < steps - 1 else len(masked)
            for position in masked[:number]:
                x[0, position] = target[position - prompt]
            if self.leave_mask and index == steps - 1:
                x[0, x.shape[1] - 1] = MASK
            history.append(x.clone())
        if self.extra_forward:
            self(x)
        if self.use_rng:
            torch.rand(1)
        return SimpleNamespace(sequences=x, history=history)


class FakeQwen(torch.nn.Module):
    def __init__(self, text: str, *, tokenizer: FakeTokenizer, terminator: int = IM_END) -> None:
        super().__init__()
        self.text_ids = [*tokenizer.encode(text), terminator]
        self.calls: list[dict[str, Any]] = []

    def _prepare_generation_config(
        self, generation_config: Any, **kwargs: Any
    ) -> tuple[SimpleNamespace, dict[str, Any]]:
        resolved = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "pad_token_id": ENDOFTEXT, **kwargs}
        return SimpleNamespace(**resolved), {}

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.calls.append(dict(kwargs))
        return torch.cat([input_ids, torch.tensor([self.text_ids], dtype=torch.long)], dim=1)
