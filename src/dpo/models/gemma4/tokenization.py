"""Tokenization safety helpers for the Gemma backend.

The one invariant everything downstream depends on: the tokenized prompt must
be a strict prefix of the tokenized prompt+completion, or completion masking
silently supervises the wrong tokens. ``prompt_and_full_encodings`` is the
single tokenization path for teacher-forced scoring — it returns the FULL
processor encoding (input ids plus every media tensor the model's forward
needs) and proves the strict-prefix invariant per call, so a template or
tokenizer boundary that breaks it fails loudly instead of misaligning the
supervised span.
"""

from __future__ import annotations

from typing import Any

from dpo.models.gemma4.prompt import CHAT_TEMPLATE_KWARGS


class TokenizationError(ValueError):
    """Raised when the chat-template/tokenizer boundary breaks the prefix invariant."""


def _encoded_ids(encoded: Any) -> list[int]:
    # apply_chat_template(return_dict=True) yields a BatchFeature (dict-like but
    # not a dict subclass); a raw list/tuple means ids were returned directly.
    ids = encoded if isinstance(encoded, (list, tuple)) else encoded["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def prompt_and_full_encodings(
    processor: Any, prompt: list[dict[str, Any]], completion: list[dict[str, Any]]
) -> tuple[int, Any]:
    """(prompt length, full prompt+completion encoding) under one template.

    The full encoding carries ``input_ids`` plus the media tensors
    (``input_features``/``pixel_values``/masks) that conditioning requires —
    callers must pass the WHOLE encoding to the model's forward, never the
    ids alone, or scoring silently ignores the media. The supervised span is
    the suffix of ``input_ids`` past the returned prompt length; the
    strict-prefix invariant is verified here on every call.
    """
    prompt_encoding = processor.apply_chat_template(
        list(prompt),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        **CHAT_TEMPLATE_KWARGS,
    )
    full_encoding = processor.apply_chat_template(
        list(prompt) + list(completion),
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        **CHAT_TEMPLATE_KWARGS,
    )
    prompt_ids = _encoded_ids(prompt_encoding)
    full_ids = _encoded_ids(full_encoding)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise TokenizationError(
            "tokenized prompt is not a prefix of prompt+completion; the chat"
            " template/tokenizer boundary is unreliable and completion masking"
            " would misalign"
        )
    if len(full_ids) <= len(prompt_ids):
        raise TokenizationError("prompt+completion adds no tokens past the prompt; nothing to supervise")
    return len(prompt_ids), full_encoding
