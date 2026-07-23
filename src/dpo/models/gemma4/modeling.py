from __future__ import annotations

import os
from typing import Any

from dpo.models.gemma4.backend_config import BackendConfig


class ModelSetupError(RuntimeError):
    """Raised when the loaded model would violate the intended QLoRA setup."""


# The 12B (gemma4_unified) chat template injects this empty thought scaffold into the
# generation prompt when add_generation_prompt=True and enable_thinking=False. It makes
# the tokenized prompt no longer a strict prefix of prompt+completion, which silently
# misaligns completion-only masking (the strict-prefix invariant in tokenization.py).
# This text-only pipeline never uses a thought channel, so we render it away for
# consistency across scoring and generation. E4B's template has no such scaffold.
_THOUGHT_SCAFFOLD = "{{- '<|channel>thought\\n<channel|>' -}}"


def local_device_map() -> dict[str, int] | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return {"": int(os.environ.get("LOCAL_RANK", "0"))}


def load_processor(config: BackendConfig) -> Any:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
        config.model.model_id,
        revision=config.model.revision,
        token=os.environ.get("HF_TOKEN"),
    )
    if not hasattr(processor, "apply_chat_template") or not hasattr(processor, "parse_response"):
        raise ModelSetupError("Gemma 4 processor lacks required chat-template/response parsing methods")
    _apply_text_only_template(processor)
    return processor


def _apply_text_only_template(processor: Any) -> None:
    """Render the empty thought scaffold out of the chat template, in place.

    Applied to every processor load so scoring and generation share one
    rendering. On E4B the scaffold is absent and this is a no-op. If the
    template ever changes shape so the substring no longer matches, the strip
    silently no-ops; callers that supervise completions must prove the
    strict-prefix invariant per row (see ``tokenization.prompt_and_full_ids``).
    """
    template = getattr(processor, "chat_template", None) or getattr(
        getattr(processor, "tokenizer", None), "chat_template", None
    )
    if not template:
        raise ModelSetupError("Gemma 4 processor has no chat template to normalize")
    stripped = template.replace(_THOUGHT_SCAFFOLD, "")
    processor.chat_template = stripped
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.chat_template = stripped


def load_quantized_base(config: BackendConfig) -> Any:
    import torch
    from transformers import AutoModelForMultimodalLM, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise ModelSetupError("4-bit Gemma 4 loading requires a CUDA GPU")
    quantization_config = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
        load_in_4bit=config.quantization.load_in_4bit,
        bnb_4bit_quant_type=config.quantization.quant_type,
        bnb_4bit_use_double_quant=config.quantization.double_quant,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        config.model.model_id,
        revision=config.model.revision,
        token=os.environ.get("HF_TOKEN"),
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        device_map=local_device_map(),
    )
    # Gemma 4 keeps use_cache on the nested text config; the top-level config does
    # not own the field.
    text_config = getattr(model.config, "text_config", None)
    (text_config if text_config is not None else model.config).use_cache = False
    return model
