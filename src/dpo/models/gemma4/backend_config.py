"""Validated backend configuration for real Gemma 4 QLoRA runs.

The backend config owns runtime and hardware concerns only: the pinned model
checkpoint, quantization, and memory-saving toggles. Everything
result-affecting — learning rates, epochs, batch geometry, LoRA topology,
seeds, precision — lives in the study contract, which pins this file's hash
for provenance; a backend config cannot express an experiment.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a backend configuration is unsafe or incomplete."""


def _table(document: dict[str, Any], name: str) -> dict[str, Any]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"missing [{name}] table")
    return value


def _boolean(values: dict[str, Any], field: str, *, default: bool = False) -> bool:
    value = values.get(field, default)
    if type(value) is not bool:
        raise ConfigError(f"{field} must be an explicit boolean")
    return value


@dataclass(frozen=True)
class ModelSettings:
    model_id: str
    revision: str
    media_inputs: str  # "visual" | "audio"

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ModelSettings:
        settings = cls(
            model_id=str(values.get("model_id", "")),
            revision=str(values.get("revision", "")),
            media_inputs=str(values.get("media_inputs", "")),
        )
        if not settings.model_id.startswith("google/gemma-4-"):
            raise ConfigError("model.model_id must be an official google/gemma-4-* checkpoint")
        if re.fullmatch(r"[0-9a-f]{40}", settings.revision) is None:
            raise ConfigError("model.revision must be a pinned 40-character commit SHA")
        if settings.media_inputs not in {"visual", "audio"}:
            raise ConfigError("model.media_inputs must be 'visual' or 'audio' (one track per backend)")
        return settings


@dataclass(frozen=True)
class QuantizationSettings:
    load_in_4bit: bool
    quant_type: str
    double_quant: bool
    compute_dtype: str

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> QuantizationSettings:
        settings = cls(
            load_in_4bit=_boolean(values, "load_in_4bit"),
            quant_type=str(values.get("quant_type", "")),
            double_quant=_boolean(values, "double_quant"),
            compute_dtype=str(values.get("compute_dtype", "")),
        )
        # 4-bit is the default and what the 24 GB budget was measured against,
        # but it is no longer mandatory: nf4 destroys Gemma 4's audio captioning
        # (measured — see configs/gemma4/e4b-audio.toml), and bf16 is the only known
        # working audio configuration. A backend that opts out states so
        # explicitly and carries its own memory budget.
        if settings.load_in_4bit and settings.quant_type != "nf4":
            raise ConfigError("quantization.quant_type must be nf4 when load_in_4bit is set")
        if settings.compute_dtype != "bfloat16":
            raise ConfigError("quantization.compute_dtype must be bfloat16 on the RTX 3090 setup")
        return settings


@dataclass(frozen=True)
class RuntimeSettings:
    gradient_checkpointing: bool

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> RuntimeSettings:
        settings = cls(gradient_checkpointing=_boolean(values, "gradient_checkpointing"))
        if not settings.gradient_checkpointing:
            raise ConfigError("gradient checkpointing is required on the 24 GB reference setup")
        return settings


@dataclass(frozen=True)
class BackendConfig:
    source_path: Path
    model: ModelSettings
    quantization: QuantizationSettings
    runtime: RuntimeSettings


def load_config(path: str | Path) -> BackendConfig:
    source_path = Path(path).resolve()
    with source_path.open("rb") as handle:
        document = tomllib.load(handle)
    return _config_from_document(document, source_path)


def load_config_text(text: str, *, source_name: str = "artifact-config.toml") -> BackendConfig:
    """Parse an artifact-resolved config without materializing an arbitrary input path."""
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError("artifact Gemma config is invalid TOML") from exc
    return _config_from_document(document, Path(source_name))


def _config_from_document(document: dict[str, Any], source_path: Path) -> BackendConfig:
    return BackendConfig(
        source_path=source_path,
        model=ModelSettings.from_dict(_table(document, "model")),
        quantization=QuantizationSettings.from_dict(_table(document, "quantization")),
        runtime=RuntimeSettings.from_dict(_table(document, "runtime")),
    )
