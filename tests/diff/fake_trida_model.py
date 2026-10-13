"""Weight-free stand-ins for the Trida and Qwen call shapes, for the runner tests.

``FakeTrida.generate`` keeps what the contract relies on in Trida's pinned
``generate``: ``max_new_tokens // 32`` blocks, one forward per block, a scripted
text written left to right, a stop at ``<|im_end|>`` with ``<|pad|>`` after it,
and no randomness unless a test asks for it. It is not the pinned
``modeling_trida.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

# Trida-7B's IDs (the tests' default tokenizer uses its chat format)
MASK, IM_START, IM_END, PAD, BOS = 128012, 128000, 128001, 128004, 0
# Qwen2.5's IDs
QWEN_ENDOFTEXT, QWEN_IM_START, QWEN_IM_END = 151643, 151644, 151645
NEWLINE = 198
SPECIALS = (BOS, IM_START, IM_END, PAD, MASK, QWEN_ENDOFTEXT, QWEN_IM_START, QWEN_IM_END)


class FakeTokenizer:
    """Characters map to IDs between 1000 and Trida's special range; the template mimics the chat format."""

    added_tokens_decoder = {token: SimpleNamespace(special=True) for token in SPECIALS}
    names = {
        "<|mask|>": MASK,
        "<|im_start|>": IM_START,
        "<|im_end|>": IM_END,
        "<|pad|>": PAD,
        "<|endoftext|>": BOS,
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
        return "".join(chr(token - 1000) if 1000 <= token < IM_START else f"<{token}>" for token in ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.names[token]


class FakeTrida(torch.nn.Module):
    def __init__(
        self,
        text: str,
        *,
        tokenizer: FakeTokenizer,
        stop: bool = True,
        leave_mask: bool = False,
        use_rng: bool = False,
    ) -> None:
        super().__init__()
        self.text_ids = tokenizer.encode(text)
        self.stop, self.leave_mask, self.use_rng = stop, leave_mask, use_rng
        self.calls: list[dict[str, Any]] = []

    def forward(self, x: torch.Tensor, *args: Any) -> SimpleNamespace:
        return SimpleNamespace(logits=None)

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.calls.append(dict(kwargs))
        canvas, block = int(kwargs["max_new_tokens"]), int(kwargs["block_size"])
        new = list(self.text_ids)
        if self.leave_mask:
            new.append(MASK)
        new = new[:canvas]
        if self.stop and len(new) < canvas:
            new += [IM_END]
        for _ in range(max(1, canvas // block)):
            self(input_ids)
        if self.use_rng:
            torch.rand(1)
        return torch.cat([input_ids, torch.tensor([new], dtype=torch.long)], dim=1)


class FakeQwen(torch.nn.Module):
    def __init__(self, text: str, *, tokenizer: FakeTokenizer, terminator: int = QWEN_IM_END) -> None:
        super().__init__()
        self.text_ids = [*tokenizer.encode(text), terminator]
        self.calls: list[dict[str, Any]] = []

    def _prepare_generation_config(
        self, generation_config: Any, **kwargs: Any
    ) -> tuple[SimpleNamespace, dict[str, Any]]:
        resolved = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "pad_token_id": QWEN_ENDOFTEXT, **kwargs}
        return SimpleNamespace(**resolved), {}

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.calls.append(dict(kwargs))
        return torch.cat([input_ids, torch.tensor([self.text_ids], dtype=torch.long)], dim=1)
