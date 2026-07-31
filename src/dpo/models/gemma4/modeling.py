"""Loading and preparing the pinned Gemma 4 checkpoint for QLoRA work.

Everything here is runtime shape: the quantized base, the processor with its
chat template normalized, the k-bit training preparation, and the guard that
keeps every trainable parameter inside the text language model.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
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
    strict-prefix invariant per row (see ``tokenization.prompt_and_full_encodings``).
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


def load_base_model(config: BackendConfig) -> Any:
    """The pinned checkpoint, quantized to nf4 or left in bfloat16.

    Unquantized is not the default but it is a supported configuration: the
    audio track needs it, because nf4 leaves the language model unable to read
    the audio soft tokens (measured; see configs/gemma4/e4b-audio.toml). Passing no
    quantization_config is what makes ``dtype`` alone govern the weights.
    """
    import torch
    from transformers import AutoModelForMultimodalLM, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise ModelSetupError("Gemma 4 loading requires a CUDA GPU")
    quantization_config = (
        BitsAndBytesConfig(  # type: ignore[no-untyped-call]
            load_in_4bit=True,
            bnb_4bit_quant_type=config.quantization.quant_type,
            bnb_4bit_use_double_quant=config.quantization.double_quant,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if config.quantization.load_in_4bit
        else None
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


def prepare_base_model(model: Any, *, gradient_checkpointing: bool, quantized: bool) -> Any:
    """Freeze the base and arm checkpointing, before any LoRA attach.

    The 4-bit path is peft's k-bit preparation: non-reentrant checkpointing,
    layernorm upcasts, grad hooks. The bf16 path needs the same two effects
    without the k-bit casts, so it does them directly — freeze every base
    parameter, then make the embedding output require grad, or checkpointing
    has no graph to recompute through once the base is frozen.
    """
    if quantized:
        from peft import prepare_model_for_kbit_training

        return prepare_model_for_kbit_training(  # type: ignore[no-untyped-call]
            model,
            use_gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    for parameter in model.parameters():
        parameter.requires_grad = False
    if gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def trainable_parameter_names(model: Any) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def assert_text_only_lora_scope(model: Any, names: Iterable[str] | None = None) -> list[str]:
    """Prove every trainable parameter lives in the text language model.

    This is the runtime enforcement of the frozen-towers invariant, and it must
    run after every LoRA attach. The vision and audio towers of Gemma 4 contain
    modules peft cannot wrap (``Gemma4ClippableLinear``; peft issues #3129/#3130,
    unfixed on the pinned peft), so a targets pattern that escapes the language
    model either crashes or — worse — silently trains a tower we require frozen.
    An empty trainable set is equally a failure: it means the pattern matched
    nothing and the "training" run would be a no-op.
    """
    trainable = list(names) if names is not None else trainable_parameter_names(model)
    if not trainable:
        raise ModelSetupError("no trainable LoRA parameters were found")
    invalid = [name for name in trainable if "language_model" not in name]
    if invalid:
        preview = ", ".join(invalid[:5])
        raise ModelSetupError(f"trainable parameters escaped the text language model: {preview}")
    forbidden = [name for name in trainable if "vision" in name or "audio" in name]
    if forbidden:
        preview = ", ".join(forbidden[:5])
        raise ModelSetupError(f"vision/audio parameters became trainable: {preview}")
    return trainable
