"""Live-backend resolution shared by train, select, and study export."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dpo.candidates.generation import (
    AUDIO_MEDIA_SUFFIXES,
    GEMMA_IMPLEMENTATION,
    TINY_IMPLEMENTATION,
    VIDEO_MEDIA_SUFFIXES,
    resolve_media_files,
    verify_backend_pin,
)
from dpo.cli._shared import _emit
from dpo.contracts.study_contract import (
    StudyContract,
)
from dpo.core.artifacts import (
    ArtifactError,
)
from dpo.core.identity import semantic_hash, sha256_file
from dpo.pipeline.live_runner import (
    TinyBackend,
    TrainingBackend,
)
from dpo.pipeline.run_matrix import DEFAULT_MEDIA_DIM


@dataclass(frozen=True)
class _BackendChoice:
    """The resolved live-training backend and the inputs it needs."""

    implementation: str
    configs: dict[str, Any]
    media_dir: Path | None

    @property
    def name(self) -> str:
        return "tiny" if self.implementation == TINY_IMPLEMENTATION else "gemma"


def _resolve_backend(
    contract: StudyContract, arguments: argparse.Namespace, *, command: str
) -> _BackendChoice | None:
    """Gate the backend before any store mutation; None means blocked (exit 3)."""
    implementation = str(contract.raw["models"]["seed"]["implementation"])
    backend_configs = [str(path) for path in (arguments.backend_config or [])]
    media_dir = arguments.media_dir
    if implementation == TINY_IMPLEMENTATION:
        if backend_configs or media_dir:
            raise ArtifactError("the tiny seed backend takes no --backend-config/--media-dir")
        return _BackendChoice(implementation=implementation, configs={}, media_dir=None)
    if implementation != GEMMA_IMPLEMENTATION:
        _emit(
            {
                "status": "blocked_pending_external_operation",
                "command": command,
                "gate": f"seed-model implementation {implementation!r} has no wired training backend",
                "side_effects": False,
            }
        )
        return None
    if not backend_configs or not media_dir:
        raise ArtifactError("the Gemma seed backend requires --backend-config and --media-dir")
    from dpo.models.gemma4.backend_config import load_config
    from dpo.models.gemma4.training_backend import resolve_lora_targets

    # A contract that cannot express a Gemma LoRA scope fails here, on any
    # machine, rather than after an operator has found a GPU.
    resolve_lora_targets(contract)
    import torch

    if not torch.cuda.is_available():
        _emit(
            {
                "status": "blocked_pending_external_operation",
                "command": command,
                "gate": "Gemma training and scoring require a CUDA device",
                "side_effects": False,
            }
        )
        return None
    configs: dict[str, Any] = {}
    for path in backend_configs:
        config = load_config(path)
        track = config.model.media_inputs
        verify_backend_pin(contract, track=track, backend_config_path=Path(path))
        if track in configs:
            raise ArtifactError(f"two backend configs serve track {track!r}")
        configs[track] = config
    # The matrix trains the tracks the CONTRACT declares, not every track that
    # exists, so a single-track study needs exactly one backend config.
    absent = [track for track in contract.tracks if track not in configs]
    if absent:
        raise ArtifactError(
            f"no --backend-config serves track {absent[0]!r}; the matrix trains every declared track"
        )
    extra = [track for track in configs if track not in contract.tracks]
    if extra:
        raise ArtifactError(
            f"--backend-config serves track {extra[0]!r}, which this contract does not declare"
        )
    return _BackendChoice(implementation=implementation, configs=configs, media_dir=Path(str(media_dir)))


def _build_backend(contract: StudyContract, choice: _BackendChoice) -> TrainingBackend:
    if choice.implementation == TINY_IMPLEMENTATION:
        return TinyBackend(contract=contract)
    from dpo.models.gemma4.training_backend import GemmaBackend

    assert choice.media_dir is not None
    return GemmaBackend(contract=contract, configs=choice.configs, media_dir=choice.media_dir)


def _require_media_coverage(choice: _BackendChoice, clips_by_track: Mapping[str, set[str]]) -> None:
    if choice.media_dir is None:
        return
    for track, clip_ids in sorted(clips_by_track.items()):
        resolve_media_files(choice.media_dir, sorted(clip_ids), track=track)


def _selection_identity(contract: StudyContract, choice: _BackendChoice) -> dict[str, str]:
    """The processor/preprocessing identities the lock manifest freezes.

    Tiny: the byte tokenizer and the synthetic media generator, exactly as the
    canary records them. Gemma: the pinned checkpoint plus the hash of the
    backend config that produced the processor, and the media-file convention
    (one directory, one file per clip by suffix) that preprocessing consists of.
    """
    if choice.implementation == TINY_IMPLEMENTATION:
        return {
            "processor_hash": semantic_hash({"processor": "tiny-byte/v1"}),
            "preprocessing_hash": semantic_hash({"media": "synthetic/v1", "media_dim": DEFAULT_MEDIA_DIM}),
        }
    return {
        "processor_hash": semantic_hash(
            {
                "processor": "gemma4-chat-template/v1",
                "tracks": {
                    track: {
                        "model_id": config.model.model_id,
                        "revision": config.model.revision,
                        "backend_config_hash": sha256_file(config.source_path),
                    }
                    for track, config in sorted(choice.configs.items())
                },
            }
        ),
        "preprocessing_hash": semantic_hash(
            {
                "media": "media-directory/v1",
                "audio_suffixes": list(AUDIO_MEDIA_SUFFIXES),
                "video_suffixes": list(VIDEO_MEDIA_SUFFIXES),
            }
        ),
    }
