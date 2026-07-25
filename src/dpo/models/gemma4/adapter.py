"""Typed boundary to the real Gemma 4 QLoRA backend.

Construction validates the pinned backend config and the track mapping; the
heavy load happens on first use and requires a CUDA device. Without one, every
method fails closed with ``BackendPending`` so a CPU session can never
silently substitute a different model for a real run. Scoring routes through
the shared completion-only log-probability implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any

import torch

from dpo.contracts.study_contract import CaptionContract, ContractError
from dpo.models.base import CompletionBatch, MediaBatch, ModalityIsolationError
from dpo.models.gemma4.backend_config import BackendConfig
from dpo.models.gemma4.prompt import assistant_message, prompt_messages, template_kwargs
from dpo.models.gemma4.tokenization import prompt_and_full_encodings
from dpo.models.logprob import completion_logprobs


class BackendPending(RuntimeError):
    """Raised when the real backend is unavailable; status is blocked, not proxied."""

    status = "blocked_pending_external_operation"


def _compute_dtype(model: Any) -> torch.dtype:
    """The model's floating-point compute dtype (weights are mixed with uint8)."""
    for parameter in model.parameters():
        if parameter.is_floating_point():
            dtype: torch.dtype = parameter.dtype
            return dtype
    return torch.float32


def _to_model(model: Any, encoding: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Move one processor encoding onto the model's device and compute dtype.

    The processor emits media features in float32 while the towers hold
    bfloat16 weights, so the features must be cast or the first tower
    LayerNorm raises a dtype mismatch. Integer tensors (ids, masks, position
    ids) are moved but never cast — casting them would corrupt them.
    """
    device = next(model.parameters()).device
    dtype = _compute_dtype(model)
    return {
        key: (value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device))
        for key, value in encoding.items()
    }


def _response_text(processor: Any, tokens: torch.Tensor) -> str:
    """The caption a generation actually asserts, with its channels resolved.

    Gemma 4 answers on a channel-structured response: it opens a thought
    channel and then gives the answer. Decoding with ``skip_special_tokens``
    drops the channel MARKERS but keeps their literal text, so a caption comes
    back as ``"thought\\n<answer>"`` — contaminating every candidate and every
    training target. The processor's own ``parse_response`` resolves the
    channels; the plain decode is only a fallback for a response it cannot
    parse.
    """
    decoded = str(processor.decode(tokens, skip_special_tokens=False))
    try:
        parsed = processor.parse_response(decoded)
    except Exception:  # noqa: BLE001 - a malformed response must not kill a run
        parsed = None
    if isinstance(parsed, Mapping):
        content = parsed.get("content")
        if isinstance(content, str):
            return content.strip()
    return str(processor.decode(tokens, skip_special_tokens=True)).strip()


class GemmaCaptionAdapter:
    """ModelAdapter over a pinned Gemma 4 checkpoint for one caption track.

    ``media_resolver`` maps a clip id to the media payload the processor
    consumes (a path or array for the track's modality). It is injected so the
    adapter never invents media paths and modality isolation stays testable.
    """

    def __init__(
        self,
        *,
        config: BackendConfig,
        contract: CaptionContract,
        media_resolver: Callable[[str], object],
        adapter_dir: str | None = None,
    ) -> None:
        if config.model.media_inputs != contract.track:
            raise ContractError(
                f"backend config serves media_inputs={config.model.media_inputs!r} but the"
                f" caption contract is for track {contract.track!r}"
            )
        self.config = config
        self.contract = contract
        self.media_resolver = media_resolver
        self.adapter_dir = adapter_dir
        self._model: Any = None
        self._processor: Any = None

    @property
    def track(self) -> str:
        return self.contract.track

    def _require_loaded(self) -> tuple[Any, Any]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        if not torch.cuda.is_available():
            raise BackendPending(
                "real Gemma backend requires a CUDA device; run the tiny offline backend for canary work"
            )
        from dpo.models.gemma4.modeling import load_processor, load_quantized_base

        self._processor = load_processor(self.config)
        model = load_quantized_base(self.config)
        if self.adapter_dir is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model, self.adapter_dir, adapter_name="default", is_trainable=True
            )
        self._model = model
        return self._model, self._processor

    def _messages(self, clip_id: str) -> list[dict[str, Any]]:
        media = self.media_resolver(clip_id)
        return prompt_messages(self.contract, media_reference=str(media))

    def completion_logps(self, media: MediaBatch, completions: CompletionBatch) -> torch.Tensor:
        if media.track != self.track:
            raise ModalityIsolationError(f"{self.track} adapter received a {media.track} media batch")
        model, processor = self._require_loaded()
        logps = []
        for clip_id, text in zip(media.clip_ids, completions.texts, strict=True):
            prompt_length, encoding = prompt_and_full_encodings(
                processor,
                self._messages(clip_id),
                assistant_message(text),
                template_kwargs=template_kwargs(self.contract),
            )
            # The WHOLE encoding goes to the forward pass: input ids alone would
            # silently score the completion without any media conditioning.
            tensors = _to_model(model, encoding)
            logits = model(**tensors).logits
            input_ids = tensors["input_ids"]
            completion_mask = torch.zeros_like(input_ids)
            completion_mask[0, prompt_length:] = 1
            logps.append(completion_logprobs(logits.float(), input_ids, completion_mask)[0].to("cpu"))
        return torch.stack(logps)

    def generate(
        self,
        media: MediaBatch,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int,
    ) -> list[str]:
        if media.track != self.track:
            raise ModalityIsolationError(f"{self.track} adapter received a {media.track} media batch")
        model, processor = self._require_loaded()
        torch.manual_seed(seed)
        outputs = []
        for clip_id in media.clip_ids:
            encoded = processor.apply_chat_template(
                self._messages(clip_id),
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                # The same template rendering as scoring, or a generated caption
                # would come from a different prompt than the one it is scored under.
                **template_kwargs(self.contract),
            )
            sampling: dict[str, Any] = (
                {"do_sample": True, "temperature": temperature, "top_p": top_p}
                if temperature > 0
                else {"do_sample": False}
            )
            generated = model.generate(
                **_to_model(model, encoded),
                **sampling,
                max_new_tokens=max_new_tokens,
            )
            prompt_length = int(encoded["input_ids"].shape[1])
            outputs.append(_response_text(processor, generated[0][prompt_length:]))
        return outputs

    def trainable_parameters(self) -> Iterator[torch.nn.Parameter]:
        model, _ = self._require_loaded()
        for parameter in model.parameters():
            if parameter.requires_grad:
                yield parameter

    def state_signature(self) -> str:
        import hashlib

        model, _ = self._require_loaded()
        digest = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            if "lora" not in name:
                continue
            data = tensor.detach().to("cpu")
            # bf16/fp16 tensors cannot view as numpy; upcasting to fp32 is exact.
            if data.dtype in (torch.bfloat16, torch.float16):
                data = data.to(torch.float32)
            digest.update(name.encode())
            digest.update(str(tuple(data.shape)).encode())
            digest.update(data.contiguous().numpy().tobytes())
        return f"sha256:{digest.hexdigest()}"
