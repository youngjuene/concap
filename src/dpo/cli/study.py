"""`dpo study`: caption the held-out study split for the human study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dpo.candidates.freeze import parse_frozen_pool
from dpo.candidates.generation import (
    resolve_media_files,
)
from dpo.cli._backend import _build_backend, _require_media_coverage, _resolve_backend
from dpo.cli._shared import (
    _emit,
    _find_manifest,
    _operation,
    _registry_shard_ids,
    _registry_shard_rows,
    _require_types,
)
from dpo.contracts.study_contract import (
    load_contract,
)
from dpo.core.access import ROLE_SCOPES, ProtectedAccessAuthority
from dpo.core.artifacts import (
    ArtifactError,
)
from dpo.core.identity import sha256_file
from dpo.evaluation.congruency import LadderRung, generate_ladders, ladder_for
from dpo.models.gemma4.adapter import GemmaCaptionAdapter
from dpo.models.gemma4.prompt import stimulus_messages
from dpo.pipeline.live_runner import (
    load_checkpoint_policies,
)
from dpo.pipeline.lock import parse_lock_manifest
from dpo.pipeline.study_stage import StudyError, publish_study_export


def _study_export(arguments: argparse.Namespace) -> int:
    contract = load_contract(arguments.contract)
    choice = _resolve_backend(contract, arguments, command="study export")
    if choice is None:
        return 3
    operation = _operation(arguments, with_training_cells=True)
    _require_types(
        operation,
        {
            "dpo.lock-manifest/v1",
            "dpo.selection-report/v1",
            "dpo.validation-report/v1",
            "dpo.clip-registry/v1",
            "dpo.frozen-candidate-pool/v1",
        },
        minimum=5,
    )
    track = str(arguments.track)
    if track not in contract.tracks:
        raise ArtifactError(
            f"this contract does not declare track {track!r}; it declares {sorted(contract.tracks)}"
        )
    lock_manifest = _find_manifest(operation, "dpo.lock-manifest/v1")
    selection_manifest = _find_manifest(operation, "dpo.selection-report/v1")
    validation_manifest = _find_manifest(operation, "dpo.validation-report/v1")
    registry_manifest = _find_manifest(operation, "dpo.clip-registry/v1")
    pool_manifest = _find_manifest(operation, "dpo.frozen-candidate-pool/v1")

    lock = parse_lock_manifest(json.loads(operation.store.read_payload(lock_manifest.artifact_id)))
    selection = json.loads(operation.store.read_payload(selection_manifest.artifact_id))
    # Selection already ranks experiments by their winner's validation accuracy,
    # so the overall winner is the first ranked experiment's selected variant.
    # The accuracy itself lives in the validation report, not the selection one.
    experiment_id = str(selection["ranking"][track][0])
    variant_id = str(selection["selected_variants"][track][experiment_id])
    validation = json.loads(operation.store.read_payload(validation_manifest.artifact_id))
    accuracy = float(validation["accuracy"][track][experiment_id][variant_id])

    training_pool = parse_frozen_pool(operation.store.read_payload(pool_manifest.artifact_id))

    # Every precondition BEFORE the fence opens. A missing checkpoint or absent
    # media is recoverable, but reserving first leaves an open reservation and a
    # study.sqlite in a workspace where no study clip was ever read. Shard
    # membership metadata names the role and clip without opening any payload.
    shard_ids = _registry_shard_ids(
        operation.store, registry_manifest, lambda entry: str(entry["role"]) == "study"
    )
    if not shard_ids:
        raise ArtifactError("the clip registry has no study-role clips to caption")
    clip_ids = sorted(shard_ids)
    _require_media_coverage(choice, {track: set(clip_ids)})

    backend = _build_backend(contract, choice)
    policies, _ = load_checkpoint_policies(backend, arguments.checkpoint_dir)
    canonical_seed = int(str(contract.training["canonical_seed"]))
    key = (experiment_id, variant_id, track, canonical_seed)
    if key not in policies:
        raise ArtifactError(
            f"checkpoint directory has no policy for the selected cell {key};"
            " run `dpo train run` with the same --checkpoint-dir first"
        )

    # One reservation per locked configuration: reserve() is idempotent for a
    # matching (scope, hash, roles), so a crashed run retries, while a new lock
    # forces a new fence and captions from two configurations cannot be mixed.
    authority = ProtectedAccessAuthority(operation.store.root / "study.sqlite")
    capability = authority.reserve(scope=ROLE_SCOPES["study"], semantic_hash=lock.lock_id, roles={"study"})
    shard_rows = _registry_shard_rows(
        operation.store,
        registry_manifest,
        lambda entry: str(entry["role"]) == "study",
        capability=capability,
        authority=authority,
        semantic_hash_for_read=lock.lock_id,
    )
    # The protected read earns its place rather than being ceremonial: it proves
    # the media about to be captioned is a derivative this corpus registered, so
    # a study cannot ship captions generated from files the registry never saw.
    if choice.media_dir is not None:
        files = resolve_media_files(choice.media_dir, clip_ids, track=track)
        for clip_id in clip_ids:
            registered = {str(value) for value in shard_rows[clip_id][1]["derivative_hashes"]}
            actual = sha256_file(files[clip_id])
            if actual not in registered:
                raise StudyError(
                    f"clip {clip_id!r}: media at {files[clip_id]} hashes to {actual}, which the"
                    " clip registry does not list as a derivative of this clip"
                )
    # One caption per (clip, rung). The bottom rung is the contract's own audio
    # prompt on audio alone — the caption this study's trained model actually
    # produces; every rung above it also sees the video, because congruency with
    # the real frame is not expressible by a model that has never seen it.
    rungs = ladder_for(contract.tracks[track].prompt)
    policy = policies[key]
    if not isinstance(policy, GemmaCaptionAdapter):
        raise StudyError(
            "congruency ladders need the Gemma backend's stimulus generation;"
            f" this policy is {type(policy).__name__}"
        )
    if choice.media_dir is None:
        raise StudyError("congruency ladders need --media-dir to resolve audio and video per clip")
    audio_files = resolve_media_files(choice.media_dir, clip_ids, track="audio")
    video_files = resolve_media_files(choice.media_dir, clip_ids, track="visual")

    def _generate(clip_id: str, rung: LadderRung) -> str:
        messages = stimulus_messages(
            rung.instruction,
            audio_reference=str(audio_files[clip_id]),
            video_reference=(str(video_files[clip_id]) if rung.conditioning == "audio+video" else None),
        )
        return policy.generate_stimulus(
            messages,
            temperature=float(str(contract.validation["temperature"])),
            top_p=float(str(contract.validation["top_p"])),
            max_new_tokens=int(str(contract.validation["max_new_tokens"])),
            seed=canonical_seed,
        )

    ladders = generate_ladders(clip_ids, rungs, _generate)
    document, artifact_id = publish_study_export(
        operation.publisher(),
        contract,
        track=track,
        experiment_id=experiment_id,
        variant_id=variant_id,
        validation_accuracy=accuracy,
        ladders=ladders,
        rungs=rungs,
        training_pool=training_pool,
        lock_artifact_id=lock_manifest.artifact_id,
        shard_artifact_ids={clip_id: shard for clip_id, (shard, _) in shard_rows.items()},
        decoding=dict(contract.validation),
    )
    authority.complete(capability)
    _emit(
        {
            "status": "published",
            "operation": "study-export",
            "artifact_id": artifact_id,
            "track": track,
            "experiment_id": experiment_id,
            "variant_id": variant_id,
            "validation_accuracy": accuracy,
            "clips": len(clip_ids),
            "ladder_summary": document["ladder_summary"],
            "training_candidate_reuse_rate": document["training_candidate_reuse_rate"],
            "lock_id": lock.lock_id,
        }
    )
    return 0


def _study_serve(arguments: argparse.Namespace) -> int:
    from dpo.userstudy.app import run_study_app

    run_study_app(
        export_path=Path(arguments.export),
        media_dir=Path(arguments.media_dir),
        out_path=Path(arguments.out),
        host=arguments.host,
        port=arguments.port,
    )
    return 0
