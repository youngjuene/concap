"""Tokenization safety helpers for the Gemma backend.

The one invariant everything downstream depends on: the tokenized prompt must
be a strict prefix of the tokenized prompt+completion, or completion masking
silently supervises the wrong tokens. ``prompt_and_full_ids`` is the single
tokenization path that makes the supervised span locatable; any future
training wiring must re-prove the strict-prefix invariant per row before GPU
work.
"""

from __future__ import annotations

from typing import Any

from dpo.models.gemma4.prompt import CHAT_TEMPLATE_KWARGS


def _encoded_ids(encoded: Any) -> list[int]:
    # apply_chat_template(return_dict=True) yields a BatchFeature (dict-like but
    # not a dict subclass); a raw list/tuple means ids were returned directly.
    ids = encoded if isinstance(encoded, (list, tuple)) else encoded["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def prompt_and_full_ids(
    processor: Any, prompt: list[dict[str, Any]], completion: list[dict[str, Any]]
) -> tuple[list[int], list[int]]:
    """Tokenize a prompt and prompt+completion conversation under one template.

    Returned as a pair so callers can locate the supervised span as the suffix
    of ``full_ids`` past ``len(prompt_ids)``. Callers that supervise the span
    must first verify ``full_ids[: len(prompt_ids)] == prompt_ids`` (the
    strict-prefix invariant) — a template/tokenizer boundary that breaks it
    would silently misalign completion masking.
    """
    prompt_ids = _encoded_ids(
        processor.apply_chat_template(
            list(prompt),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            **CHAT_TEMPLATE_KWARGS,
        )
    )
    full_ids = _encoded_ids(
        processor.apply_chat_template(
            list(prompt) + list(completion),
            tokenize=True,
            return_dict=True,
            **CHAT_TEMPLATE_KWARGS,
        )
    )
    return prompt_ids, full_ids
