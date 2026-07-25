"""The real Gemma 4 QLoRA training backend behind the matrix runner's seam.

One quantized base per track is loaded once, prepared for k-bit training, and
kept for the whole run; every policy is a named LoRA adapter attached to that
base. Attaching rather than reloading is what keeps eighteen cells inside one
24 GB device, and it is why every attach is followed by
``assert_text_only_lora_scope``: Gemma 4's vision and audio towers hold modules
the pinned peft cannot wrap (``Gemma4ClippableLinear``, peft #3129/#3130), so a
targets pattern that escapes the language model would either crash or silently
train a tower this study requires frozen.

Result-affecting settings come from the study contract — the LoRA topology is
read from ``[training.lora]``, never from the backend TOML, which owns runtime
and hardware concerns only. Scoring is never reimplemented here: policies are
``GemmaCaptionAdapter`` subclasses that resolve the currently active model and
otherwise use the adapter's media-conditioned completion-log-probability path.
"""

from __future__ import annotations

import gc
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from dpo.candidates.generation import resolve_media_files
from dpo.contracts.study_contract import ContractError, StudyContract
from dpo.core.identity import semantic_hash
from dpo.models.audio_media import build_audio_media
from dpo.models.base import MediaBatch, ModelAdapter
from dpo.models.gemma4.adapter import GemmaCaptionAdapter
from dpo.models.gemma4.backend_config import BackendConfig
from dpo.models.gemma4.modeling import (
    assert_text_only_lora_scope,
    load_processor,
    load_quantized_base,
    prepare_quantized_model,
)
from dpo.models.visual_media import build_visual_media

# The known-good language-model scope for Gemma 4 QLoRA on this study.
LANGUAGE_MODEL_TARGETS = (
    r"^model\.language_model\.layers\.[0-9]+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$"
)
# The offline tiny model's module names; a contract carrying these has not been
# adapted for a live run.
TINY_LORA_TARGETS = frozenset({"mixer", "head", "token_embedding", "position_embedding"})

_REGEX_HINT = re.compile(r"[\\^$|()\[\]]")


def resolve_lora_targets(contract: StudyContract) -> str | list[str]:
    """The contract's LoRA targets, refused when they are the tiny model's."""
    raw = contract.training["lora"]["targets"]
    values = [str(target) for target in raw] if isinstance(raw, list) else [str(raw)]
    if not values:
        raise ContractError("training.lora.targets is empty")
    offending = sorted(set(values) & TINY_LORA_TARGETS)
    if offending:
        raise ContractError(
            f"training.lora.targets={values} names the offline tiny model's modules"
            f" ({', '.join(offending)}); a Gemma run needs language-model targets."
            f' Set training.lora.targets = ["{LANGUAGE_MODEL_TARGETS}"]'
        )
    if len(values) == 1 and _REGEX_HINT.search(values[0]) is not None:
        return values[0]
    return values


@dataclass(frozen=True)
class _Attachment:
    """One attached LoRA adapter: its peft name and where it came from."""

    name: str
    track: str
    directory: Path | None


class GemmaPolicyAdapter(GemmaCaptionAdapter):
    """A ``GemmaCaptionAdapter`` bound to one attachment of the shared base.

    Scoring, generation, and parameter enumeration are inherited untouched;
    only model resolution changes: every call activates this adapter's LoRA on
    the backend's single loaded model first, so alternating between a policy
    and its reference during validation can never score the wrong weights.
    """

    def __init__(
        self,
        *,
        backend: GemmaBackend,
        config: BackendConfig,
        contract_track: str,
        attachment: _Attachment | None,
    ) -> None:
        super().__init__(
            config=config,
            contract=backend.contract.tracks[contract_track],
            media_resolver=lambda clip_id: backend.media_file(contract_track, clip_id),
        )
        self._backend = backend
        self._attachment = attachment

    @property
    def attachment(self) -> _Attachment | None:
        return self._attachment

    def _require_loaded(self) -> tuple[Any, Any]:
        return self._backend.activate(self.track, self._attachment)

    def state_signature(self) -> str:
        if self._attachment is None:
            # SEED is the pinned base checkpoint itself: identify it by the pin,
            # not by an empty LoRA state.
            return semantic_hash(
                {
                    "schema": "dpo.gemma-seed-signature/v1",
                    "model_id": self.config.model.model_id,
                    "revision": self.config.model.revision,
                    "track": self.track,
                }
            )
        return self._backend.adapter_signature(self.track, self._attachment)


@dataclass
class GemmaBackend:
    """Track-scoped quantized bases plus named LoRA attachments."""

    contract: StudyContract
    configs: Mapping[str, BackendConfig]
    media_dir: Path
    # Two attachments cover the worst case (a policy and its frozen reference);
    # anything older is deleted to give its device memory straight back.
    max_attached: int = 2
    _base: dict[str, Any] = field(default_factory=dict)
    _peft: dict[str, Any] = field(default_factory=dict)
    _processor: dict[str, Any] = field(default_factory=dict)
    _resident: dict[str, list[str]] = field(default_factory=dict)
    _sources: dict[tuple[str, str], Path] = field(default_factory=dict)
    _counter: int = 0

    # -- configuration ---------------------------------------------------------

    def config_for(self, track: str) -> BackendConfig:
        config = self.configs.get(track)
        if config is None:
            raise ContractError(
                f"no backend config serves track {track!r}; pass one --backend-config per caption track"
            )
        return config

    def processor(self, track: str) -> Any:
        processor = self._processor.get(track)
        if processor is None:
            processor = load_processor(self.config_for(track))
            self._processor[track] = processor
        return processor

    def base(self, track: str) -> Any:
        model = self._base.get(track)
        if model is None:
            config = self.config_for(track)
            model = prepare_quantized_model(
                load_quantized_base(config),
                gradient_checkpointing=config.runtime.gradient_checkpointing,
            )
            self._base[track] = model
        return model

    def media_file(self, track: str, clip_id: str) -> Path:
        return resolve_media_files(self.media_dir, [clip_id], track=track)[clip_id]

    def close(self) -> None:
        """Drop every loaded model and free the device memory they hold.

        One CLI command owns one backend for its whole process, so production
        runs never need this; it exists so a caller that builds several
        backends in one process (tests, notebooks) can reclaim VRAM instead of
        exhausting the device on the second load.
        """
        self._peft.clear()
        self._base.clear()
        self._processor.clear()
        self._resident.clear()
        self._sources.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _lora_config(self) -> Any:
        from peft import LoraConfig, TaskType

        lora = self.contract.training["lora"]
        return LoraConfig(
            r=int(str(lora["rank"])),
            lora_alpha=int(str(lora["alpha"])),
            lora_dropout=float(str(lora["dropout"])),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=resolve_lora_targets(self.contract),
        )

    # -- attachments -----------------------------------------------------------

    def _model(self, track: str) -> Any:
        peft_model = self._peft.get(track)
        return self.base(track) if peft_model is None else peft_model

    @staticmethod
    def _repair_active(peft_model: Any) -> None:
        """Keep ``active_adapter`` naming an adapter that still exists.

        Every peft forward resolves ``peft_config[active_adapter]``, including
        the SEED pass that runs with the adapter layers disabled. Deleting the
        adapter that happened to be active leaves that name dangling, so the
        next forward raises KeyError instead of scoring the base model.
        """
        names = list(peft_model.peft_config)
        if names and peft_model.active_adapter not in peft_model.peft_config:
            peft_model.set_adapter(names[-1])

    def _touch(self, track: str, name: str) -> None:
        resident = self._resident.setdefault(track, [])
        if name in resident:
            resident.remove(name)
        resident.append(name)
        while len(resident) > self.max_attached:
            self._detach(track, resident[0])

    def _detach(self, track: str, name: str) -> None:
        resident = self._resident.setdefault(track, [])
        if name not in resident:
            return
        resident.remove(name)
        peft_model = self._peft.get(track)
        if peft_model is not None and name in peft_model.peft_config:
            peft_model.delete_adapter(name)
            # peft leaves ``active_adapter`` naming a deleted adapter, and the
            # next attribute lookup then raises KeyError. Hand activation to a
            # surviving adapter, or fall back to the bare base.
            survivors = [other for other in resident if other in peft_model.peft_config]
            if survivors:
                peft_model.set_adapter(survivors[-1])
            else:
                peft_model.base_model.disable_adapter_layers()
        # The wrapper itself stays for the lifetime of the track: re-wrapping a
        # base that already carries LoRA layers would double-inject them.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _attach(self, track: str, attachment: _Attachment) -> None:
        """Materialize one named LoRA on the track's base, guard, and record it."""
        from peft import get_peft_model

        peft_model = self._peft.get(track)
        if peft_model is None:
            if attachment.directory is None:
                peft_model = get_peft_model(self.base(track), self._lora_config(), attachment.name)
            else:
                from peft import PeftModel

                peft_model = PeftModel.from_pretrained(
                    self.base(track),
                    str(attachment.directory),
                    adapter_name=attachment.name,
                    is_trainable=True,
                )
            self._peft[track] = peft_model
        elif attachment.directory is None:
            peft_model.add_adapter(attachment.name, self._lora_config())
        else:
            peft_model.load_adapter(
                str(attachment.directory), adapter_name=attachment.name, is_trainable=True
            )
        peft_model = self._peft[track]
        peft_model.set_adapter(attachment.name)
        # The frozen-towers invariant, enforced at runtime after every attach.
        assert_text_only_lora_scope(peft_model)
        self._touch(track, attachment.name)

    def activate(self, track: str, attachment: _Attachment | None) -> tuple[Any, Any]:
        """Make ``attachment`` the live weights of the track and return (model, processor)."""
        peft_model = self._peft.get(track)
        if attachment is None:
            if peft_model is not None:
                if not peft_model.peft_config:
                    # No adapter survives to name; the pristine base IS the SEED.
                    self._peft.pop(track, None)
                    self._resident.pop(track, None)
                    return self.base(track), self.processor(track)
                self._repair_active(peft_model)
                peft_model.base_model.disable_adapter_layers()
            return self._model(track), self.processor(track)
        if attachment.name not in self._resident.setdefault(track, []):
            source = self._sources.get((track, attachment.name))
            if source is None:
                raise ContractError(
                    f"adapter {attachment.name!r} was released and has no saved checkpoint to restore"
                )
            self._attach(track, _Attachment(attachment.name, track, source))
        else:
            peft_model = self._peft[track]
            peft_model.set_adapter(attachment.name)
            self._touch(track, attachment.name)
        peft_model = self._peft[track]
        peft_model.base_model.enable_adapter_layers()
        return peft_model, self.processor(track)

    def adapter_signature(self, track: str, attachment: _Attachment) -> str:
        """Content signature of exactly this attachment's LoRA weights."""
        import hashlib

        from peft import get_peft_model_state_dict

        self.activate(track, attachment)
        digest = hashlib.sha256()
        state = get_peft_model_state_dict(  # type: ignore[no-untyped-call]
            self._peft[track], adapter_name=attachment.name
        )
        for name, tensor in sorted(state.items()):
            data = tensor.detach().to("cpu")
            if data.dtype in (torch.bfloat16, torch.float16):
                data = data.to(torch.float32)
            digest.update(name.encode())
            digest.update(str(tuple(data.shape)).encode())
            digest.update(data.contiguous().numpy().tobytes())
        return f"sha256:{digest.hexdigest()}"

    # -- TrainingBackend -------------------------------------------------------

    def seed_adapter(self, track: str) -> ModelAdapter:
        return GemmaPolicyAdapter(
            backend=self, config=self.config_for(track), contract_track=track, attachment=None
        )

    def clone_trainable(self, track: str, source: ModelAdapter | None) -> ModelAdapter:
        self._counter += 1
        name = f"policy{self._counter:03d}"
        directory: Path | None = None
        if source is not None:
            if not isinstance(source, GemmaPolicyAdapter) or source.attachment is None:
                raise ContractError("the Gemma backend can only warm-start from one of its checkpoints")
            directory = self._sources.get((track, source.attachment.name))
            if directory is None:
                raise ContractError("the warm-start source has no saved checkpoint to copy")
        attachment = _Attachment(name=name, track=track, directory=directory)
        self._attach(track, attachment)
        if directory is not None:
            self._sources[(track, name)] = directory
        return GemmaPolicyAdapter(
            backend=self, config=self.config_for(track), contract_track=track, attachment=attachment
        )

    def save(self, adapter: ModelAdapter, directory: Path) -> None:
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file

        if not isinstance(adapter, GemmaPolicyAdapter) or adapter.attachment is None:
            raise ContractError("the Gemma backend can only save one of its attached policies")
        attachment = adapter.attachment
        track = attachment.track
        self.activate(track, attachment)
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        state = get_peft_model_state_dict(  # type: ignore[no-untyped-call]
            self._peft[track], adapter_name=attachment.name
        )
        save_file(
            {key: value.detach().to("cpu").contiguous() for key, value in state.items()},
            str(destination / "adapter_model.safetensors"),
        )
        self._peft[track].peft_config[attachment.name].save_pretrained(str(destination))
        self._sources[(track, attachment.name)] = destination

    def load(self, track: str, directory: Path) -> ModelAdapter:
        self._counter += 1
        name = f"loaded{self._counter:03d}"
        attachment = _Attachment(name=name, track=track, directory=Path(directory))
        self._attach(track, attachment)
        self._sources[(track, name)] = Path(directory)
        return GemmaPolicyAdapter(
            backend=self, config=self.config_for(track), contract_track=track, attachment=attachment
        )

    def release(self, adapter: ModelAdapter) -> None:
        if not isinstance(adapter, GemmaPolicyAdapter):
            return
        attachment = adapter.attachment
        if attachment is None:
            return
        self._detach(attachment.track, attachment.name)

    def media_batch(self, track: str, clip_ids: Sequence[str]) -> MediaBatch:
        """Clip identity for the media-resolving adapter; the model reads files.

        The Gemma adapter conditions on the media file its resolver returns, so
        the batch carries identity and the modality-isolation checks only.
        """
        count = len(clip_ids)
        if track == "audio":
            return build_audio_media(list(clip_ids), torch.zeros(count, 1600))
        return build_visual_media(list(clip_ids), torch.zeros(count, 4, 8))

    def count_completion_tokens(self, track: str, text: str) -> int:
        """The real tokenizer's completion length, for the contract budget."""
        processor = self.processor(track)
        tokenizer = getattr(processor, "tokenizer", processor)
        return int(len(tokenizer(text)["input_ids"]))
