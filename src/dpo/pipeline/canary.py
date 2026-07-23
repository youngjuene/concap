"""The offline end-to-end canary: every stage, real artifacts, no live data.

One command proves the executable boundary of the whole pipeline on synthetic
fixtures — contract lock through corpus, candidates, annotation, views, the
nine-condition matrix, validation and selection, and the configuration lock —
all content-addressed. A warm rerun in the same workspace reuses every
artifact and makes zero provider calls.

The stage bodies live in the shared ``*_stage`` modules; this module holds
only the synthetic fixture generators, the orchestration that calls the
shared stages in pipeline order, and the canary's own assertions (the
integrity oracles).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from dpo.annotation.aggregate import PairAggregate
from dpo.annotation.raw_annotations import RawAnnotation
from dpo.candidates.audit import (
    CandidateAudit,
    ResolvedCandidateAudit,
    audit_candidate,
    content_tokens,
    resolve_audits,
)
from dpo.candidates.candidate_records import (
    CandidateRecord,
    CollectionPolicy,
    GenerationConfig,
    build_candidate_records,
)
from dpo.candidates.freeze import FrozenCandidatePool
from dpo.contracts.study_contract import (
    EXPERIMENT_IDS,
    TRACKS,
    DecodingMixture,
    StudyContract,
    load_contract,
)
from dpo.core.artifacts import ArtifactStore, ArtifactTampered, ParentEdge, ProtectedExposure
from dpo.core.identity import repo_lock_hash, semantic_hash, sha256_bytes
from dpo.data.split import ClipInput
from dpo.models.base import MediaBatch
from dpo.models.tiny import synthetic_media
from dpo.pipeline.annotation_stage import ingest_annotations
from dpo.pipeline.candidate_stage import publish_frozen_pool
from dpo.pipeline.corpus_stage import publish_corpus_ingest, publish_lock_splits
from dpo.pipeline.experiments import expand_experiment
from dpo.pipeline.publishing import ArtifactPublisher
from dpo.pipeline.run_matrix import DEFAULT_MEDIA_DIM, OfflineMatrixRunner
from dpo.pipeline.selection_stage import publish_selection
from dpo.pipeline.training_stage import publish_training_matrix
from dpo.pipeline.view_stage import TrackViews, publish_track_views

REPORT_TYPE = "dpo.canary-report/v1"


class CanaryError(RuntimeError):
    """Raised when the offline milestone cannot complete."""


@dataclass(frozen=True)
class CanaryResult:
    artifact_id: str
    report: dict[str, object]
    cached: bool
    provider_calls: int


# Per-clip word banks. Content words are distinct across clips (bank index is
# clip-derived) so the cross-split near-duplicate audit has real signal.
# fmt: off
_VISUAL_SUBJECTS = (
    "cyclist", "gardener", "vendor", "skater", "painter", "runner", "climber", "juggler",
    "farmer", "barista", "tailor", "potter", "surfer", "welder", "librarian", "beekeeper",
)
_VISUAL_ACTIONS = (
    "crosses", "waters", "arranges", "circles", "sketches", "passes", "scales", "balances",
    "harvests", "pours", "measures", "shapes", "rides", "joins", "shelves", "inspects",
)
_VISUAL_OBJECTS = (
    "junction", "seedlings", "stall", "ramp", "mural", "track", "wall", "pins",
    "rows", "cups", "fabric", "wheel", "wave", "beams", "cart", "hive",
)
_AUDIO_EVENTS = (
    "footsteps", "kettle", "doorbell", "engine", "typewriter", "rainfall", "sander", "zipper",
    "stapler", "clock", "faucet", "grinder", "printer", "shutter", "drill", "mixer",
)
_AUDIO_QUALITIES = (
    "repeated", "distant", "sudden", "steady", "faint", "sharp", "slow", "rapid",
    "gentle", "hollow", "crisp", "uneven", "brief", "layered", "muffled", "rhythmic",
)
# fmt: on


def _bank_index(clip_id: str, salt: str, size: int) -> int:
    digest = semantic_hash({"clip": clip_id, "salt": salt}).removeprefix("sha256:")
    return int(digest[:8], 16) % size


def _clip_terms(clip_id: str, track: str) -> dict[str, str]:
    if track == "visual":
        subject = _VISUAL_SUBJECTS[_bank_index(clip_id, "subject", len(_VISUAL_SUBJECTS))]
        action = _VISUAL_ACTIONS[_bank_index(clip_id, "action", len(_VISUAL_ACTIONS))]
        target = _VISUAL_OBJECTS[_bank_index(clip_id, "object", len(_VISUAL_OBJECTS))]
        object_index = _bank_index(clip_id, "object", len(_VISUAL_OBJECTS))
        wrong = _VISUAL_OBJECTS[(object_index + 5) % len(_VISUAL_OBJECTS)]
        return {"first": subject, "second": action, "third": target, "wrong": wrong}
    event = _AUDIO_EVENTS[_bank_index(clip_id, "event", len(_AUDIO_EVENTS))]
    quality = _AUDIO_QUALITIES[_bank_index(clip_id, "quality", len(_AUDIO_QUALITIES))]
    other = _AUDIO_EVENTS[(_bank_index(clip_id, "event", len(_AUDIO_EVENTS)) + 7) % len(_AUDIO_EVENTS)]
    wrong = _AUDIO_EVENTS[(_bank_index(clip_id, "event", len(_AUDIO_EVENTS)) + 3) % len(_AUDIO_EVENTS)]
    return {"first": quality, "second": event, "third": other, "wrong": wrong}


def _candidate_bank(clip_id: str, track: str) -> dict[str, list[str]]:
    terms = _clip_terms(clip_id, track)
    first, second, third, wrong = terms["first"], terms["second"], terms["third"], terms["wrong"]
    # One clip-unique content token per caption bounds worst-case cross-clip
    # lexical overlap below the near-duplicate threshold even when the small
    # word banks collide across clips.
    site = "site" + semantic_hash({"site": clip_id}).removeprefix("sha256:")[:6]
    if track == "visual":
        return {
            "greedy": [f"The {first} {second} the {third} at {site}."],
            "sample": [
                f"At {site}, the {first} slowly {second} the {third}.",
                f"A {first} moves toward the {third} at {site}.",
            ],
            "controlled_error": [f"The {first} {second} the {third} with a {wrong} nearby at {site}."],
        }
    return {
        "greedy": [f"A {first} {second} continues while a {third} stays behind at {site}."],
        "sample": [
            f"Near {site}, a {third} follows the {first} {second}.",
            f"A single {second} starts and fades near {site}.",
        ],
        "controlled_error": [f"A {first} {second} continues while a {wrong} outside interrupts at {site}."],
    }


def _candidate_generations(
    clip_id: str,
    track: str,
    *,
    mixture: tuple[DecodingMixture, ...],
    max_new_tokens: int,
    seed: int,
) -> list[tuple[str, str, GenerationConfig]]:
    """Synthetic candidates shaped by the contract's frozen decoding mixture."""
    bank = _candidate_bank(clip_id, track)
    generations: list[tuple[str, str, GenerationConfig]] = []
    for entry in mixture:
        texts = bank.get(entry.name)
        if texts is None or entry.count > len(texts):
            available = 0 if texts is None else len(texts)
            raise CanaryError(
                f"the synthetic fixture provides {available} {entry.name!r} candidates;"
                f" the contract mixture requests {entry.count}"
            )
        config = GenerationConfig(
            temperature=entry.temperature,
            top_p=entry.top_p,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        generations.extend((entry.name, text, config) for text in texts[: entry.count])
    return generations


def _corpus_fixture(contract: StudyContract) -> list[ClipInput]:
    clips = []
    for source in range(14):
        for part in range(2 if source % 3 == 0 else 1):
            clip_id = f"clip-{source:02d}{chr(ord('a') + part)}"
            clips.append(
                ClipInput(
                    clip_id=clip_id,
                    source_video_id=f"src-{source:02d}",
                    media_hash=sha256_bytes(f"media:{clip_id}".encode()),
                    start_ms=part * 9000,
                    end_ms=part * 9000 + 8000,
                    derivative_hashes=(
                        sha256_bytes(f"visual:{clip_id}".encode()),
                        sha256_bytes(f"audio:{clip_id}".encode()),
                    ),
                )
            )
    return clips


def _annotator_hash(index: int) -> str:
    return sha256_bytes(f"annotator-{index}".encode())


def _quality_score(candidate: CandidateRecord, audit: ResolvedCandidateAudit) -> float:
    """Deterministic synthetic preference signal for the fixture annotators.

    The fixture's controlled-error captions contain the clip's known wrong
    term; a synthetic "human" dispreference for that term plus the audit's
    structural gates stand in for real judgment.
    """
    wrong = _clip_terms(candidate.clip_id, candidate.track)["wrong"]
    score = 0.0
    if wrong in content_tokens(candidate.text):
        score -= 3.0
    if not audit.acceptable:
        score -= 2.0
    if audit.modality_violation:
        score -= 4.0
    return score


def _synthetic_annotations(
    pool: FrozenCandidatePool,
    audits: Mapping[str, ResolvedCandidateAudit],
    *,
    collection_version: str,
) -> tuple[list[RawAnnotation], dict[str, str]]:
    annotations: list[RawAnnotation] = []
    attention_expected: dict[str, str] = {}
    counter = 0
    # Exact left/right counterbalancing of the WINNER per annotator over the
    # decisive pairs, as a real interface would schedule it; random flips can
    # drift far enough on small fixtures to trip the position-bias exclusion.
    decisive_counts = dict.fromkeys(range(3), 0)
    for pair_index, pair in enumerate(pool.pairs):
        score_a = _quality_score(pool.candidate(pair.candidate_a), audits[pair.candidate_a])
        score_b = _quality_score(pool.candidate(pair.candidate_b), audits[pair.candidate_b])
        decisive = abs(score_a - score_b) >= 0.5
        for annotator in range(3):
            if decisive:
                winner = pair.candidate_a if score_a > score_b else pair.candidate_b
                loser = pair.candidate_b if winner == pair.candidate_a else pair.candidate_a
                winner_left = decisive_counts[annotator] % 2 == 0
                decisive_counts[annotator] += 1
                display = (winner, loser) if winner_left else (loser, winner)
                choice = "a_better" if winner_left else "b_better"
                subtype = None
                strength: int | None = 4 if abs(score_a - score_b) >= 3 else 3
            else:
                flip = (pair_index + annotator) % 2 == 1
                display = (
                    (pair.candidate_b, pair.candidate_a) if flip else (pair.candidate_a, pair.candidate_b)
                )
                choice, subtype, strength = "tie", "both_acceptable", None
            # fmt: off
            annotation = RawAnnotation(
                annotation_id=f"ann-{counter:06d}", pair_id=pair.pair_id,
                clip_id=pair.clip_id, track=pair.track,
                candidate_a=pair.candidate_a, candidate_b=pair.candidate_b,
                display_order=display, choice=choice, tie_subtype=subtype,
                preference_strength=strength, confidence=4, reason_tags=("factual_support",),
                annotator_id_hash=_annotator_hash(annotator), replay_count=1,
                response_time_ms=8000, collection_version=collection_version,
                presentation="muted_video" if pair.track == "visual" else "audio_only",
            )
            # fmt: on
            annotations.append(annotation)
            counter += 1
            if annotator == 0 and pair_index % 8 == 0:
                annotations.append(
                    replace(
                        annotation,
                        annotation_id=f"ann-{counter:06d}",
                        response_time_ms=8200,
                        repeat_of=annotation.annotation_id,
                    )
                )
                counter += 1
    for check_index in range(max(1, len(pool.pairs) // 20)):
        pair = pool.pairs[check_index % len(pool.pairs)]
        pair_id = f"attn-{pool.track}-{check_index:03d}"
        attention_expected[pair_id] = "a_better"
        for annotator in range(3):
            # fmt: off
            annotations.append(
                RawAnnotation(
                    annotation_id=f"ann-{counter:06d}", pair_id=pair_id,
                    clip_id=pair.clip_id, track=pair.track,
                    candidate_a=pair.candidate_a, candidate_b=pair.candidate_b,
                    display_order=(pair.candidate_a, pair.candidate_b),
                    choice="a_better", tie_subtype=None, preference_strength=5, confidence=5,
                    reason_tags=(), annotator_id_hash=_annotator_hash(annotator), replay_count=1,
                    response_time_ms=6000, collection_version=collection_version,
                    presentation="muted_video" if pair.track == "visual" else "audio_only",
                    is_attention_check=True,
                )
            )
            # fmt: on
            counter += 1
    return annotations, attention_expected


def _synthetic_media(track: str, clip_ids: Sequence[str]) -> MediaBatch:
    """The canary's media fixture for scoring and validation metrics."""
    return synthetic_media(track, clip_ids, media_dim=DEFAULT_MEDIA_DIM)


@dataclass
class _TrackData:
    pools: dict[str, FrozenCandidatePool]
    audits: dict[str, dict[str, ResolvedCandidateAudit]]
    annotations: dict[str, tuple[RawAnnotation, ...]]
    aggregates: dict[str, tuple[PairAggregate, ...]]
    pool_artifacts: dict[str, str]
    view_artifacts: dict[str, str]


def run_canary(workspace: str | Path, contract_path: str | Path) -> CanaryResult:
    contract = load_contract(contract_path)
    if contract.execution_class != "synthetic_canary":
        raise CanaryError("the offline canary requires a synthetic_canary contract")
    lock_id = repo_lock_hash()
    store = ArtifactStore.create(workspace)
    source_identity = ArtifactStore.source_identity()
    for report_id in store.find_by_type(REPORT_TYPE):
        report_manifest = store.verify(report_id)
        attributes = report_manifest.semantic.get("attributes")
        if (
            isinstance(attributes, Mapping)
            and attributes.get("contract_hash") == contract.contract_hash
            and attributes.get("lock_id") == lock_id
            and attributes.get("source_identity") == source_identity
        ):
            report = json.loads(store.read_payload(report_id))
            return CanaryResult(artifact_id=report_id, report=report, cached=True, provider_calls=0)
    publisher = ArtifactPublisher(store, contract, lock_id)
    # The trimmed default pipeline is provider-free; the field stays so the
    # CLI and Makefile zero-provider assertions keep their contract.
    provider_calls = 0

    contract_lock_id = publisher.publish(
        "dpo.contract-lock/v1",
        {"schema": "dpo.contract-lock/v1", "contract": contract.raw},
        stage="contract",
        parameters={"operation": "contract-lock"},
    )

    clips = _corpus_fixture(contract)
    ingest_id = publish_corpus_ingest(publisher, clips)
    _, manifest, shard_ids = publish_lock_splits(publisher, contract, clips, ingest_artifact_id=ingest_id)

    role_of = manifest.role_of
    train_clips = list(manifest.assignments["train"])
    validation_clips = list(manifest.assignments["validation"])
    test_clips = list(manifest.assignments["test"])

    policy = CollectionPolicy(
        policy_id="C0",
        checkpoint_hash=str(contract.raw["models"]["seed"]["lock_hash"]),
    )
    candidate_max_new_tokens = int(str(contract.candidates["max_new_tokens"]))
    generation_seed = int(str(contract.candidates["generation_seed"]))
    track_data: dict[str, _TrackData] = {}
    for track in TRACKS:
        pools: dict[str, FrozenCandidatePool] = {}
        audits_by_split: dict[str, dict[str, ResolvedCandidateAudit]] = {}
        annotations_by_split: dict[str, tuple[RawAnnotation, ...]] = {}
        aggregates_by_split: dict[str, tuple[PairAggregate, ...]] = {}
        pool_artifact_ids: dict[str, str] = {}
        by_split_candidates: dict[str, list[CandidateRecord]] = {"train": [], "validation": []}
        by_split_audits: dict[str, list[CandidateAudit]] = {"train": [], "validation": []}
        for clip_id in (*train_clips, *validation_clips):
            role = role_of(clip_id)
            records = build_candidate_records(
                clip_id=clip_id,
                track=track,
                policy=policy,
                generations=_candidate_generations(
                    clip_id,
                    track,
                    mixture=contract.decoding_mixture,
                    max_new_tokens=candidate_max_new_tokens,
                    seed=generation_seed,
                ),
            )
            by_split_candidates[role].extend(records)
            for record in records:
                by_split_audits[role].append(audit_candidate(record, contract=contract.tracks[track]))
        for split in ("train", "validation"):
            resolved_audits = dict(resolve_audits(by_split_audits[split], []))
            # fmt: off
            pool, pool_artifact_ids[split] = publish_frozen_pool(
                publisher, contract, track=track, split=split,
                candidates=by_split_candidates[split], audits=resolved_audits,
                shard_artifact_ids=shard_ids,
                dataset_version="canary/v1", audit_version="audit/v1",
            )
            annotations, attention_expected = _synthetic_annotations(
                pool, resolved_audits,
                collection_version=str(contract.annotation["collection_version"]),
            )
            retained, aggregates, _ = ingest_annotations(
                publisher, contract, track=track, split=split,
                pool=pool, pool_artifact_id=pool_artifact_ids[split],
                annotations=annotations, attention_expected=attention_expected,
            )
            # fmt: on
            pools[split] = pool
            audits_by_split[split] = resolved_audits
            annotations_by_split[split] = retained
            aggregates_by_split[split] = aggregates
        track_data[track] = _TrackData(
            pools=pools,
            audits=audits_by_split,
            annotations=annotations_by_split,
            aggregates=aggregates_by_split,
            pool_artifacts=pool_artifact_ids,
            view_artifacts={},
        )

    # Derived views, weighting, noise calibration, flips, and leakage audit.
    views_by_track: dict[str, TrackViews] = {}
    for track in TRACKS:
        data = track_data[track]
        # fmt: off
        views = publish_track_views(
            publisher, contract, track=track, manifest=manifest, clips=clips,
            train_pool=data.pools["train"], validation_pool=data.pools["validation"],
            train_pool_artifact_id=data.pool_artifacts["train"],
            validation_pool_artifact_id=data.pool_artifacts["validation"],
            train_aggregates=data.aggregates["train"],
            validation_aggregates=data.aggregates["validation"],
            train_annotations=data.annotations["train"], train_audits=data.audits["train"],
        )
        # fmt: on
        if not views.strict_pairs or not views.sft_rows or not views.validation_pairs:
            raise CanaryError(f"track {track!r} produced an empty training or validation view")
        data.view_artifacts.update(views.artifact_ids)
        views_by_track[track] = views

    # The nine-condition matrix on the tiny backend, canonical seed only (the
    # canary proves the executable boundary; the full 3-seed sweep is a real run).
    canonical_seed = int(str(contract.training["canonical_seed"]))
    runner = OfflineMatrixRunner(
        contract=contract,
        strict_pairs={track: views_by_track[track].strict_pairs for track in TRACKS},
        metadata_pairs={track: views_by_track[track].metadata_pairs for track in TRACKS},
        sft_rows={track: views_by_track[track].sft_rows for track in TRACKS},
    )
    variants_by_experiment = {
        experiment_id: expand_experiment(contract, experiment_id) for experiment_id in EXPERIMENT_IDS
    }
    view_artifacts = {track: track_data[track].view_artifacts for track in TRACKS}
    # fmt: off
    cells, cell_artifacts = publish_training_matrix(
        publisher, contract, runner=runner, variants_by_experiment=variants_by_experiment,
        canonical_seed=canonical_seed, view_artifacts=view_artifacts,
        strict_pairs={track: views_by_track[track].strict_pairs for track in TRACKS},
    )

    # Common validation scoring, selection, and the configuration lock.
    selection = publish_selection(
        publisher, contract, variants_by_experiment=variants_by_experiment,
        canonical_seed=canonical_seed,
        validation_pairs={track: views_by_track[track].validation_pairs for track in TRACKS},
        strict_pairs={track: views_by_track[track].strict_pairs for track in TRACKS},
        policies=runner.policies,
        seed_adapters={track: runner.seed_adapter(track) for track in TRACKS},
        media_provider=_synthetic_media, view_artifacts=view_artifacts,
        cells=cells, cell_artifacts=cell_artifacts,
        processor_hash=semantic_hash({"processor": "tiny-byte/v1"}),
        preprocessing_hash=semantic_hash({"media": "synthetic/v1", "media_dim": DEFAULT_MEDIA_DIM}),
        evaluation_version="evaluation/v1",
        metric_versions={"compliance": "v1", "preference": "v1"},
        selection_note="canary selection: validation preference accuracy, lexical tie-break",
    )
    # fmt: on

    # Integrity oracles: the store must fail closed on leakage and tampering.
    try:
        publisher.publish(
            "dpo.matrix-cell/v1",
            {"schema": "dpo.matrix-cell/v1", "oracle": "leakage"},
            parents=(ParentEdge(shard_ids[test_clips[0]], "test-shard"),),
            stage="views",
            parameters={"operation": "leakage-oracle"},
            clips={test_clips[0]},
            purpose="training",
        )
    except ProtectedExposure:
        leakage_oracle = "protected_exposure_rejected"
    else:
        raise CanaryError("a training artifact with test ancestry must be rejected")
    probe_payload_path = store.payload_path(contract_lock_id)
    original = probe_payload_path.read_bytes()
    probe_payload_path.write_bytes(original[:-1] + bytes([original[-1] ^ 0xFF]))
    try:
        store.verify(contract_lock_id)
    except ArtifactTampered:
        tamper_oracle = "payload_corruption_detected"
    else:
        raise CanaryError("payload corruption must be detected")
    finally:
        probe_payload_path.write_bytes(original)
    store.verify(contract_lock_id)

    report = {
        "schema": REPORT_TYPE,
        "status": "offline_milestone_complete",
        "release": "blocked_pending_external_operation",
        "clips": len(clips),
        "splits": {split: len(manifest.assignments[split]) for split in manifest.assignments},
        "matrix_cells": len(cells),
        "validation_accuracy": selection.validation_reports,
        "selection_ranking": selection.ranking,
        "oracles": {"leakage": leakage_oracle, "tamper": tamper_oracle},
        "provider_calls": provider_calls,
    }
    report_id = publisher.publish(
        REPORT_TYPE,
        report,
        parents=(ParentEdge(selection.lock_artifact_id, "lock-manifest"),),
        stage="*",
        parameters={"operation": "canary-report"},
        attributes={
            "contract_hash": contract.contract_hash,
            "lock_id": lock_id,
            "source_identity": source_identity,
        },
    )
    return CanaryResult(artifact_id=report_id, report=report, cached=False, provider_calls=provider_calls)
