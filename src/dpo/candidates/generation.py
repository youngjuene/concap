"""C0 candidate generation: the contract-pinned seed model over each clip.

Two seed-model implementations are wired:

* ``dpo.models.tiny.TinyAdapter`` — the offline backend over synthetic media.
  Two operationalizations keep its byte-level outputs inside the record
  contracts without weakening any screen: **hex transcription** (raw
  generations routinely contain control characters the untrusted-text screen
  rightly rejects; every generation is transcribed to its space-separated
  lowercase hex byte form, bijective with the model output, and the caption
  contract's compliance screens still judge — and typically flag — them), and
  **controlled errors** as mismatched captions (the greedy generation of a
  different clip of the same split, scanning deterministic offsets until the
  mismatch differs from every caption already generated for the clip).

* ``dpo.models.gemma4.adapter.GemmaCaptionAdapter`` — the real backend over
  media files resolved from a local directory (same suffix conventions as the
  annotation server). Captions are the model's text verbatim; controlled
  errors use the same mismatched-greedy construction. Requires CUDA; the
  caller gates and verifies the backend-config pin before invoking.

Per-sample decoding seeds derive deterministically from
``(generation_seed, clip_id, mixture-entry name, sample index)`` via
``semantic_hash``, so samples within one clip stay distinct and the whole
pool regenerates bit-identically for a fixed model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from dpo.candidates.candidate_records import (
    CandidateError,
    CandidateRecord,
    CollectionPolicy,
    GenerationConfig,
    build_candidate_records,
)
from dpo.contracts.study_contract import StudyContract
from dpo.core.identity import semantic_hash, sha256_file
from dpo.models.tiny import TinyAdapter, synthetic_media

TINY_IMPLEMENTATION = "dpo.models.tiny.TinyAdapter"
GEMMA_IMPLEMENTATION = "dpo.models.gemma4.adapter.GemmaCaptionAdapter"
MEDIA_DIM = 16

# One media-file convention shared with the annotation server.
AUDIO_MEDIA_SUFFIXES = (".wav", ".mp3", ".m4a", ".ogg", ".flac")
VIDEO_MEDIA_SUFFIXES = (".mp4", ".webm", ".mov")

# (clip_id, temperature, top_p, seed) -> final caption text.
GenerateOne = Callable[[str, float, float, int], str]


def derive_generation_seed(generation_seed: int, clip_id: str, entry_name: str, sample_index: int) -> int:
    """Deterministic per-sample decoding seed; distinct per (clip, entry, sample)."""
    digest = semantic_hash(
        {
            "generation_seed": generation_seed,
            "clip": clip_id,
            "entry": entry_name,
            "sample": sample_index,
        }
    ).removeprefix("sha256:")
    return int(digest[:12], 16)


def transcribe_generation(raw: str, *, seed: int) -> str:
    """Hex byte transcription of one raw tiny generation (see the module docstring)."""
    data = raw.encode("utf-8")
    if not data:
        return f"empty generation {seed:x}"
    return " ".join(f"{byte:02x}" for byte in data)


def _mismatched_caption(
    position: int, ordered: Sequence[str], greedy_texts: dict[str, str], exclude: set[str]
) -> str:
    """The first offset clip's greedy caption that collides with nothing local."""
    total = len(ordered)
    for offset in range(1, total):
        text = greedy_texts[ordered[(position + offset) % total]]
        if text not in exclude:
            return text
    raise CandidateError(
        "every greedy caption in the split collides with the clip's own candidates;"
        " cannot construct a mismatched controlled error"
    )


def _collect_records(
    contract: StudyContract,
    *,
    track: str,
    clip_ids: Sequence[str],
    policy: CollectionPolicy,
    generate_one: GenerateOne,
) -> tuple[CandidateRecord, ...]:
    """The shared C0 mixture loop over one already-constructed seed backend."""
    generation_seed = int(str(contract.candidates["generation_seed"]))
    max_new_tokens = int(str(contract.candidates["max_new_tokens"]))
    ordered = sorted(set(clip_ids))
    mixture = contract.decoding_mixture
    needs_mismatch = any(entry.name == "controlled_error" for entry in mixture)
    greedy_texts: dict[str, str] = {}
    if needs_mismatch:
        if len(ordered) < 2:
            raise CandidateError("controlled errors require at least two clips in the split")
        for clip_id in ordered:
            seed = derive_generation_seed(generation_seed, clip_id, "greedy", 0)
            greedy_texts[clip_id] = generate_one(clip_id, 0.0, 1.0, seed)
    records: list[CandidateRecord] = []
    for position, clip_id in enumerate(ordered):
        generations: list[tuple[str, str, GenerationConfig]] = []
        for entry in mixture:
            for sample_index in range(entry.count):
                seed = derive_generation_seed(generation_seed, clip_id, entry.name, sample_index)
                config = GenerationConfig(
                    temperature=entry.temperature,
                    top_p=entry.top_p,
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                )
                if entry.name == "controlled_error":
                    exclude = {text for _, text, _ in generations} | {greedy_texts[clip_id]}
                    text = _mismatched_caption(position, ordered, greedy_texts, exclude)
                else:
                    text = generate_one(clip_id, entry.temperature, entry.top_p, seed)
                generations.append((entry.name, text, config))
        records.extend(
            build_candidate_records(clip_id=clip_id, track=track, policy=policy, generations=generations)
        )
    return tuple(records)


def generate_c0_candidates(
    contract: StudyContract, *, track: str, clip_ids: Sequence[str]
) -> tuple[CandidateRecord, ...]:
    """Generate the full C0 candidate set with the offline tiny seed model."""
    seed_model = contract.raw["models"]["seed"]
    if str(seed_model["implementation"]) != TINY_IMPLEMENTATION:
        raise CandidateError("offline C0 generation requires the tiny seed-model implementation")
    max_new_tokens = int(str(contract.candidates["max_new_tokens"]))
    adapter = TinyAdapter(
        track=track,
        prompt=contract.tracks[track].prompt,
        media_dim=MEDIA_DIM,
        seed=int(str(seed_model["init_seed"])),
    )

    def generate_one(clip_id: str, temperature: float, top_p: float, seed: int) -> str:
        raw = adapter.generate(
            synthetic_media(track, [clip_id], media_dim=MEDIA_DIM),
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )[0]
        return transcribe_generation(raw, seed=seed)

    policy = CollectionPolicy(policy_id="C0", checkpoint_hash=str(seed_model["lock_hash"]))
    return _collect_records(
        contract, track=track, clip_ids=clip_ids, policy=policy, generate_one=generate_one
    )


def resolve_media_files(media_dir: Path, clip_ids: Sequence[str], *, track: str) -> dict[str, Path]:
    """clip_id -> media file, by the shared suffix convention; fails on any gap."""
    suffixes = AUDIO_MEDIA_SUFFIXES if track == "audio" else VIDEO_MEDIA_SUFFIXES
    resolved: dict[str, Path] = {}
    missing: list[str] = []
    for clip_id in sorted(set(clip_ids)):
        for suffix in suffixes:
            candidate = media_dir / f"{clip_id}{suffix}"
            if candidate.is_file():
                resolved[clip_id] = candidate
                break
        else:
            missing.append(clip_id)
    if missing:
        raise CandidateError(
            f"media directory {str(media_dir)!r} has no {track} file for clip {missing[0]!r}"
            f" (expected one of {', '.join(suffixes)}); {len(missing)} clip(s) missing"
        )
    return resolved


def verify_backend_pin(contract: StudyContract, *, track: str, backend_config_path: Path) -> None:
    """Enforce the contract's [backends] hash pin for this track when present."""
    backends = contract.raw.get("backends")
    if not isinstance(backends, Mapping):
        return
    entry = backends.get(track)
    if not isinstance(entry, Mapping):
        return
    pinned = str(entry["config_hash"])
    actual = sha256_file(backend_config_path)
    if actual != pinned:
        raise CandidateError(
            f"backend config {str(backend_config_path)!r} hashes to {actual} but the contract"
            f" pins backends.{track}.config_hash = {pinned}"
        )


def generate_c0_candidates_gemma(
    contract: StudyContract,
    *,
    track: str,
    clip_ids: Sequence[str],
    backend_config_path: Path,
    media_dir: Path,
) -> tuple[CandidateRecord, ...]:
    """Generate the full C0 candidate set with the real Gemma seed backend.

    The caller must have gated on CUDA availability and the seed-model
    implementation; this function verifies the backend pin, the track
    mapping, and full media coverage before any model load.
    """
    import torch

    from dpo.models.audio_media import build_audio_media
    from dpo.models.base import MediaBatch
    from dpo.models.gemma4.adapter import GemmaCaptionAdapter
    from dpo.models.gemma4.backend_config import load_config
    from dpo.models.visual_media import build_visual_media

    seed_model = contract.raw["models"]["seed"]
    if str(seed_model["implementation"]) != GEMMA_IMPLEMENTATION:
        raise CandidateError("gemma C0 generation requires the Gemma seed-model implementation")
    verify_backend_pin(contract, track=track, backend_config_path=backend_config_path)
    config = load_config(backend_config_path)
    if config.model.media_inputs != track:
        raise CandidateError(
            f"backend config serves media_inputs={config.model.media_inputs!r};"
            f" the requested track is {track!r}"
        )
    files = resolve_media_files(media_dir, clip_ids, track=track)
    adapter = GemmaCaptionAdapter(
        config=config,
        contract=contract.tracks[track],
        media_resolver=lambda clip_id: files[clip_id],
    )
    max_new_tokens = int(str(contract.candidates["max_new_tokens"]))

    def placeholder_media(clip_id: str) -> MediaBatch:
        # The Gemma adapter consumes media through its resolver; the batch
        # exists to carry clip identity and the modality-isolation checks.
        if track == "audio":
            return build_audio_media([clip_id], torch.zeros(1, 1600))
        return build_visual_media([clip_id], torch.zeros(1, 4, 8))

    def generate_one(clip_id: str, temperature: float, top_p: float, seed: int) -> str:
        text = adapter.generate(
            placeholder_media(clip_id),
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )[0].strip()
        return text if text else f"empty generation {seed:x}"

    policy = CollectionPolicy(policy_id="C0", checkpoint_hash=str(seed_model["lock_hash"]))
    return _collect_records(
        contract, track=track, clip_ids=clip_ids, policy=policy, generate_one=generate_one
    )
