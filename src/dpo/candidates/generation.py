"""Offline C0 candidate generation with the deterministic tiny seed model.

The offline collection policy C0 runs the contract-pinned ``TinyAdapter``
seed model over each clip's synthetic media and freezes the outputs as
candidate captions, one candidate per decoding-mixture sample. Two
operationalizations keep the byte-level outputs inside the record contracts
without weakening any screen:

* **Hex transcription** — the tiny model emits raw bytes, which routinely
  include control characters that candidate records rightly reject as
  untrusted text. Every generation is transcribed to its space-separated
  lowercase hex byte form (bijective with the model output), so records stay
  valid while the caption contract's compliance screens still judge — and
  typically flag — them; those flags feed the audit, never a hard error.
* **Controlled errors** — the ``controlled_error`` mixture kind is
  operationalized offline as a mismatched caption: the greedy
  (temperature-0) generation of a DIFFERENT clip of the same split. The
  mismatch source is found by scanning deterministic offsets from the clip's
  position in the sorted split until the mismatched text differs from every
  caption already generated for the clip, so identical greedy outputs across
  clips can never produce a duplicate candidate.

Per-sample decoding seeds derive deterministically from
``(generation_seed, clip_id, mixture-entry name, sample index)`` via
``semantic_hash``, so samples within one clip stay distinct and the whole
pool regenerates bit-identically.
"""

from __future__ import annotations

from collections.abc import Sequence

from dpo.candidates.candidate_records import (
    CandidateError,
    CandidateRecord,
    CollectionPolicy,
    GenerationConfig,
    build_candidate_records,
)
from dpo.contracts.study_contract import StudyContract
from dpo.core.identity import semantic_hash
from dpo.models.tiny import TinyAdapter, synthetic_media

TINY_IMPLEMENTATION = "dpo.models.tiny.TinyAdapter"
MEDIA_DIM = 16


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
    """Hex byte transcription of one raw generation (see the module docstring)."""
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


def generate_c0_candidates(
    contract: StudyContract, *, track: str, clip_ids: Sequence[str]
) -> tuple[CandidateRecord, ...]:
    """Generate the full C0 candidate set for one (track, split-clip) collection."""
    seed_model = contract.raw["models"]["seed"]
    if str(seed_model["implementation"]) != TINY_IMPLEMENTATION:
        raise CandidateError("offline C0 generation requires the tiny seed-model implementation")
    generation_seed = int(str(contract.candidates["generation_seed"]))
    max_new_tokens = int(str(contract.candidates["max_new_tokens"]))
    adapter = TinyAdapter(
        track=track,
        prompt=contract.tracks[track].prompt,
        media_dim=MEDIA_DIM,
        seed=int(str(seed_model["init_seed"])),
    )
    policy = CollectionPolicy(policy_id="C0", checkpoint_hash=str(seed_model["lock_hash"]))
    ordered = sorted(set(clip_ids))
    mixture = contract.decoding_mixture
    needs_mismatch = any(entry.name == "controlled_error" for entry in mixture)
    greedy_texts: dict[str, str] = {}
    if needs_mismatch:
        if len(ordered) < 2:
            raise CandidateError("controlled errors require at least two clips in the split")
        for clip_id in ordered:
            seed = derive_generation_seed(generation_seed, clip_id, "greedy", 0)
            raw = adapter.generate(
                synthetic_media(track, [clip_id], media_dim=MEDIA_DIM),
                temperature=0.0,
                top_p=1.0,
                max_new_tokens=max_new_tokens,
                seed=seed,
            )[0]
            greedy_texts[clip_id] = transcribe_generation(raw, seed=seed)
    records: list[CandidateRecord] = []
    for position, clip_id in enumerate(ordered):
        media = synthetic_media(track, [clip_id], media_dim=MEDIA_DIM)
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
                    raw = adapter.generate(
                        media,
                        temperature=entry.temperature,
                        top_p=entry.top_p,
                        max_new_tokens=max_new_tokens,
                        seed=seed,
                    )[0]
                    text = transcribe_generation(raw, seed=seed)
                generations.append((entry.name, text, config))
        records.extend(
            build_candidate_records(clip_id=clip_id, track=track, policy=policy, generations=generations)
        )
    return tuple(records)
