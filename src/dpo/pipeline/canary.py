"""The offline end-to-end canary: every stage, real artifacts, no live data.

One command proves the executable boundary of the whole pipeline on synthetic
fixtures: contract lock, corpus ingest, immutable splits, pinned fixture
evidence, audited claim ledgers, frozen per-split candidate pools, synthetic
annotation with repeats and attention checks, reliability screening, derived
views, leakage audit, the full nine-condition matrix on the tiny backend, validation
scoring and selection, the configuration lock, the fenced test-once read, the
blinded study export, and a Bradley-Terry analysis — all content-addressed.
A warm rerun in the same workspace reuses every artifact and makes zero
provider calls.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from dpo.analysis.bootstrap import clip_cluster_bootstrap
from dpo.analysis.bradley_terry import PairwiseOutcome, fit_bradley_terry
from dpo.annotation.aggregate import PairAggregate, aggregate_all
from dpo.annotation.raw_annotations import RawAnnotation, validate_against_pool
from dpo.annotation.reliability import build_reliability_report, retained_annotations
from dpo.candidates.audit import (
    CandidateAudit,
    ResolvedCandidateAudit,
    audit_candidate,
    resolve_audits,
)
from dpo.candidates.candidate_records import (
    CandidateRecord,
    CollectionPolicy,
    GenerationConfig,
    build_candidate_records,
)
from dpo.candidates.freeze import FrozenCandidatePool, freeze_pool
from dpo.candidates.pair_sampler import sample_pairs
from dpo.contracts.study_contract import (
    EXPERIMENT_IDS,
    TRACKS,
    DecodingMixture,
    StudyContract,
    load_contract,
)
from dpo.core.access import ProtectedAccessAuthority
from dpo.core.artifacts import (
    ArtifactStore,
    ArtifactTampered,
    ParentEdge,
    ProtectedExposure,
)
from dpo.core.identity import repo_lock_hash, semantic_hash, sha256_bytes
from dpo.data.derive_pairs import (
    MetadataPair,
    StrictPair,
    derive_pair_all,
    derive_pair_strict,
    with_weights,
)
from dpo.data.derive_sft import SftExample, derive_sft
from dpo.data.leakage_audit import run_leakage_audit
from dpo.data.noise import estimate_natural_noise, make_flip_manifest
from dpo.data.split import ClipInput
from dpo.data.weighting import pair_weights
from dpo.evaluation.compliance import GeneratedCaption, compute_compliance
from dpo.evaluation.human_study_export import build_study_export
from dpo.evaluation.preference_accuracy import ScoredPair, evaluate_preferences
from dpo.evidence.audio_evidence import parse_audio_evidence
from dpo.evidence.claim_ledger import (
    ClaimAudit,
    ClaimLedger,
    apply_audits,
    propose_claims,
)
from dpo.evidence.providers import (
    NO_RETRY_POLICY,
    AdapterIdentity,
    EvidenceRunner,
    FixtureAdapter,
    capability_schema_hash,
)
from dpo.evidence.visual_evidence import parse_visual_evidence
from dpo.models.base import CompletionBatch
from dpo.models.tiny import synthetic_media
from dpo.pipeline.corpus_stage import publish_corpus_ingest, publish_lock_splits
from dpo.pipeline.experiments import expand_experiment
from dpo.pipeline.lock import (
    LockError,
    TestOnceResolver,
    create_lock_manifest,
    test_split_semantic_hash,
)
from dpo.pipeline.publishing import ArtifactPublisher
from dpo.pipeline.run_matrix import DEFAULT_MEDIA_DIM, CellResult, OfflineMatrixRunner

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
_VISUAL_SUBJECTS = (
    "cyclist",
    "gardener",
    "vendor",
    "skater",
    "painter",
    "runner",
    "climber",
    "juggler",
    "farmer",
    "barista",
    "tailor",
    "potter",
    "surfer",
    "welder",
    "librarian",
    "beekeeper",
)
_VISUAL_ACTIONS = (
    "crosses",
    "waters",
    "arranges",
    "circles",
    "sketches",
    "passes",
    "scales",
    "balances",
    "harvests",
    "pours",
    "measures",
    "shapes",
    "rides",
    "joins",
    "shelves",
    "inspects",
)
_VISUAL_OBJECTS = (
    "junction",
    "seedlings",
    "stall",
    "ramp",
    "mural",
    "track",
    "wall",
    "pins",
    "rows",
    "cups",
    "fabric",
    "wheel",
    "wave",
    "beams",
    "cart",
    "hive",
)
_AUDIO_EVENTS = (
    "footsteps",
    "kettle",
    "doorbell",
    "engine",
    "typewriter",
    "rainfall",
    "sander",
    "zipper",
    "stapler",
    "clock",
    "faucet",
    "grinder",
    "printer",
    "shutter",
    "drill",
    "mixer",
)
_AUDIO_QUALITIES = (
    "repeated",
    "distant",
    "sudden",
    "steady",
    "faint",
    "sharp",
    "slow",
    "rapid",
    "gentle",
    "hollow",
    "crisp",
    "uneven",
    "brief",
    "layered",
    "muffled",
    "rhythmic",
)


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


def _fixture_response(clip_id: str, track: str) -> str:
    terms = _clip_terms(clip_id, track)
    if track == "visual":
        document: dict[str, object] = {
            "language": "en",
            "clip_id": clip_id,
            "frame_timestamps_ms": [0, 2000, 4000, 6000],
            "scene_boundaries_ms": [3000],
            "items": [
                {
                    "kind": "object",
                    "label": f"{terms['first']} at the {terms['third']}",
                    "start_ms": 0,
                    "end_ms": 4000,
                    "confidence": 0.9,
                    "salience": 0.9,
                },
                {
                    "kind": "action",
                    "label": f"{terms['first']} {terms['second']} the {terms['third']}",
                    "start_ms": 1000,
                    "end_ms": 6000,
                    "confidence": 0.8,
                    "salience": 0.8,
                },
                {
                    "kind": "object",
                    "label": f"a {terms['wrong']} nearby",
                    "start_ms": 5000,
                    "end_ms": 7000,
                    "confidence": 0.3,
                    "salience": 0.2,
                },
            ],
        }
    else:
        document = {
            "language": "en",
            "clip_id": clip_id,
            "speech_present": False,
            "music_present": False,
            "low_activity_spans_ms": [[6000, 7000]],
            "items": [
                {
                    "kind": "sound_event",
                    "label": f"{terms['first']} {terms['second']}",
                    "start_ms": 0,
                    "end_ms": 4000,
                    "confidence": 0.9,
                    "salience": 0.9,
                    "foreground": True,
                },
                {
                    "kind": "sound_event",
                    "label": f"a {terms['third']} behind",
                    "start_ms": 2000,
                    "end_ms": 6000,
                    "confidence": 0.7,
                    "salience": 0.5,
                    "foreground": False,
                },
                {
                    "kind": "sound_event",
                    "label": f"a {terms['wrong']} outside",
                    "start_ms": 5000,
                    "end_ms": 7000,
                    "confidence": 0.3,
                    "salience": 0.2,
                    "foreground": False,
                },
            ],
        }
    return json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _candidate_bank(clip_id: str, track: str) -> dict[str, list[str]]:
    terms = _clip_terms(clip_id, track)
    # One clip-unique content token per caption bounds worst-case cross-clip
    # lexical overlap below the near-duplicate threshold even when the small
    # word banks collide across clips.
    site = "site" + semantic_hash({"site": clip_id}).removeprefix("sha256:")[:6]
    if track == "visual":
        return {
            "greedy": [f"The {terms['first']} {terms['second']} the {terms['third']} at {site}."],
            "sample": [
                f"At {site}, the {terms['first']} slowly {terms['second']} the {terms['third']}.",
                f"A {terms['first']} moves toward the {terms['third']} at {site}.",
            ],
            "controlled_error": [
                f"The {terms['first']} {terms['second']} the {terms['third']} with a"
                f" {terms['wrong']} nearby at {site}.",
            ],
        }
    return {
        "greedy": [
            f"A {terms['first']} {terms['second']} continues while a {terms['third']} stays"
            f" behind at {site}.",
        ],
        "sample": [
            f"Near {site}, a {terms['third']} follows the {terms['first']} {terms['second']}.",
            f"A single {terms['second']} starts and fades near {site}.",
        ],
        "controlled_error": [
            f"A {terms['first']} {terms['second']} continues while a {terms['wrong']} outside"
            f" interrupts at {site}.",
        ],
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


def _quality_score(audit: ResolvedCandidateAudit) -> float:
    deterministic = audit.deterministic
    return (
        2.0 * deterministic.supported_overlap
        - 1.0 * deterministic.unsupported_token_count
        - 4.0 * deterministic.contradicted_overlap
        - 4.0 * float(deterministic.cross_modal_violation)
    )


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
        score_a = _quality_score(audits[pair.candidate_a])
        score_b = _quality_score(audits[pair.candidate_b])
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
            annotation = RawAnnotation(
                annotation_id=f"ann-{counter:06d}",
                pair_id=pair.pair_id,
                clip_id=pair.clip_id,
                track=pair.track,
                candidate_a=pair.candidate_a,
                candidate_b=pair.candidate_b,
                display_order=display,
                choice=choice,
                tie_subtype=subtype,
                preference_strength=strength,
                confidence=4,
                reason_tags=("factual_support",),
                annotator_id_hash=_annotator_hash(annotator),
                replay_count=1,
                response_time_ms=8000,
                collection_version=collection_version,
            )
            annotations.append(annotation)
            counter += 1
            if annotator == 0 and pair_index % 8 == 0:
                annotations.append(
                    RawAnnotation(
                        annotation_id=f"ann-{counter:06d}",
                        pair_id=pair.pair_id,
                        clip_id=pair.clip_id,
                        track=pair.track,
                        candidate_a=pair.candidate_a,
                        candidate_b=pair.candidate_b,
                        display_order=display,
                        choice=choice,
                        tie_subtype=subtype,
                        preference_strength=strength,
                        confidence=4,
                        reason_tags=("factual_support",),
                        annotator_id_hash=_annotator_hash(annotator),
                        replay_count=1,
                        response_time_ms=8200,
                        collection_version=collection_version,
                        repeat_of=annotation.annotation_id,
                    )
                )
                counter += 1
    for check_index in range(max(1, len(pool.pairs) // 20)):
        pair = pool.pairs[check_index % len(pool.pairs)]
        pair_id = f"attn-{pool.track}-{check_index:03d}"
        attention_expected[pair_id] = "a_better"
        for annotator in range(3):
            annotations.append(
                RawAnnotation(
                    annotation_id=f"ann-{counter:06d}",
                    pair_id=pair_id,
                    clip_id=pair.clip_id,
                    track=pair.track,
                    candidate_a=pair.candidate_a,
                    candidate_b=pair.candidate_b,
                    display_order=(pair.candidate_a, pair.candidate_b),
                    choice="a_better",
                    tie_subtype=None,
                    preference_strength=5,
                    confidence=5,
                    reason_tags=(),
                    annotator_id_hash=_annotator_hash(annotator),
                    replay_count=1,
                    response_time_ms=6000,
                    collection_version=collection_version,
                    is_attention_check=True,
                )
            )
            counter += 1
    return annotations, attention_expected


@dataclass
class _TrackData:
    pools: dict[str, FrozenCandidatePool]
    audits: dict[str, dict[str, ResolvedCandidateAudit]]
    annotations: dict[str, tuple[RawAnnotation, ...]]
    aggregates: dict[str, tuple[PairAggregate, ...]]
    ledgers: dict[str, ClaimLedger]
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
    study_clips = list(manifest.assignments["study"])

    evidence_runner = EvidenceRunner(
        store,
        contract_id=publisher.contract_id_for("evidence", None),
        prompt_contracts={track: contract.tracks[track].prompt for track in TRACKS},
    )
    policy = CollectionPolicy(
        policy_id="C0",
        checkpoint_hash=str(contract.raw["models"]["seed"]["lock_hash"]),
    )
    candidate_max_new_tokens = int(str(contract.candidates["max_new_tokens"]))
    generation_seed = int(str(contract.candidates["generation_seed"]))
    track_data: dict[str, _TrackData] = {}
    for track in TRACKS:
        capability = "visual_evidence" if track == "visual" else "audio_evidence"
        derivative_index = 0 if track == "visual" else 1
        model_section = contract.raw["models"][f"evidence_{track}"]
        pools: dict[str, FrozenCandidatePool] = {}
        audits_by_split: dict[str, dict[str, ResolvedCandidateAudit]] = {}
        annotations_by_split: dict[str, tuple[RawAnnotation, ...]] = {}
        aggregates_by_split: dict[str, tuple[PairAggregate, ...]] = {}
        ledgers: dict[str, ClaimLedger] = {}
        pool_artifact_ids: dict[str, str] = {}
        by_split_candidates: dict[str, list[CandidateRecord]] = {"train": [], "validation": []}
        by_split_audits: dict[str, list[CandidateAudit]] = {"train": [], "validation": []}
        ledger_artifact_ids: dict[str, str] = {}
        for clip_id in (*train_clips, *validation_clips):
            role = role_of(clip_id)
            clip = next(item for item in clips if item.clip_id == clip_id)
            adapter = FixtureAdapter(
                AdapterIdentity(
                    capability=capability,
                    implementation=str(model_section["implementation"]),
                    model=str(model_section["model_id"]),
                    revision=str(model_section["revision"]),
                    prompt_hash=sha256_bytes(contract.tracks[track].prompt.encode()),
                    schema_hash=capability_schema_hash(capability),
                    lock_hash=str(model_section["lock_hash"]),
                    credential_env="DPO_FIXTURE",
                ),
                _fixture_response(clip_id, track),
            )
            run = evidence_runner.run(
                adapter,
                clip_id=clip_id,
                track=track,
                media_hash=clip.derivative_hashes[derivative_index],
                seed=int(str(contract.corpus["split_seed"])),
                parser_version="parser/v1",
                decoding={"temperature": 0.0, "top_p": 1.0, "max_tokens": 256},
                retry_policy=NO_RETRY_POLICY,
                role_exposure={role},
            )
            provider_calls += adapter.calls
            parsed = (
                parse_visual_evidence(dict(run.parsed), provider=adapter.identity.model)
                if track == "visual"
                else parse_audio_evidence(dict(run.parsed), provider=adapter.identity.model)
            )
            proposed = propose_claims(parsed, track=track)
            ledger = ClaimLedger(
                clip_id=clip_id,
                track=track,
                claims=proposed,
                prohibited_cross_modal_claims=(),
                audit_version="audit/v1",
            )
            terms = _clip_terms(clip_id, track)
            decisions = []
            for claim in ledger.claims:
                status = "contradicted" if terms["wrong"] in claim.canonical_form else "supported"
                decisions.append(ClaimAudit(claim_id=claim.claim_id, auditor_id="auditor-01", status=status))
            ledger = apply_audits(ledger, decisions)
            ledgers[clip_id] = ledger
            ledger_artifact_ids[clip_id] = publisher.publish(
                "dpo.claim-ledger/v1",
                ledger.document(),
                parents=(ParentEdge(run.parsed_artifact_id, "parsed-evidence"),),
                stage="claims",
                parameters={"operation": "claims", "clip_id": clip_id, "track": track},
                row_count=len(ledger.claims),
                clips={clip_id},
                role_exposure={role},
            )
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
                by_split_audits[role].append(
                    audit_candidate(record, contract=contract.tracks[track], ledger=ledger)
                )
        for split in ("train", "validation"):
            resolved_audits = dict(resolve_audits(by_split_audits[split], []))
            pairs = sample_pairs(
                by_split_candidates[split],
                resolved_audits,
                per_clip_min=int(str(contract.pairs["per_clip_min"])),
                per_clip_max=int(str(contract.pairs["per_clip_max"])),
                seed=int(str(contract.pairs["seed"])),
                near_duplicate_jaccard=float(str(contract.pairs["near_duplicate_jaccard"])),
            )
            pool = freeze_pool(
                dataset_version="canary/v1",
                track=track,
                evidence_audit_version="audit/v1",
                candidates=by_split_candidates[split],
                pairs=pairs,
            )
            split_clips = {candidate.clip_id for candidate in pool.candidates}
            pool_artifact_ids[split] = publisher.publish(
                "dpo.frozen-candidate-pool/v1",
                pool.document(),
                parents=tuple(
                    ParentEdge(ledger_artifact_ids[clip_id], "claim-ledger")
                    for clip_id in sorted(split_clips)
                ),
                stage="candidates",
                parameters={"operation": "candidates-freeze", "track": track, "split": split},
                row_count=len(pool.pairs),
                clips=split_clips,
                role_exposure={split},
                attributes={"pool_hash": pool.pool_hash, "track": track, "split": split},
            )
            annotations, attention_expected = _synthetic_annotations(
                pool,
                resolved_audits,
                collection_version=str(contract.annotation["collection_version"]),
            )
            validate_against_pool(annotations, pool)
            reliability = build_reliability_report(
                annotations,
                attention_expected=attention_expected,
                min_response_ms=int(str(contract.annotation["min_response_ms"])),
                max_position_bias=float(str(contract.annotation["max_position_bias"])),
                min_attention_pass=float(str(contract.annotation["min_attention_pass"])),
            )
            annotations_artifact = publisher.publish(
                "dpo.raw-annotations/v1",
                {
                    "schema": "dpo.raw-annotations/v1",
                    "rows": [annotation.document() for annotation in annotations],
                },
                parents=(ParentEdge(pool_artifact_ids[split], "frozen-pool"),),
                stage="annotation",
                parameters={"operation": "annotation-ingest", "track": track, "split": split},
                row_count=len(annotations),
                clips=split_clips,
                role_exposure={split},
            )
            publisher.publish(
                "dpo.reliability-report/v1",
                reliability.document(),
                parents=(ParentEdge(annotations_artifact, "raw-annotations"),),
                stage="annotation",
                parameters={"operation": "reliability", "track": track, "split": split},
                clips=split_clips,
                role_exposure={split},
            )
            retained = retained_annotations(annotations, reliability)
            aggregates = aggregate_all(
                retained, minimum_judgments=int(str(contract.annotation["judgments_per_pair"]))
            )
            pools[split] = pool
            audits_by_split[split] = resolved_audits
            annotations_by_split[split] = tuple(retained)
            aggregates_by_split[split] = aggregates
        track_data[track] = _TrackData(
            pools=pools,
            audits=audits_by_split,
            annotations=annotations_by_split,
            aggregates=aggregates_by_split,
            ledgers=ledgers,
            pool_artifacts=pool_artifact_ids,
            view_artifacts={},
        )

    # Derived views, weighting, noise calibration, flips, and leakage audit.
    strict_by_track: dict[str, tuple[StrictPair, ...]] = {}
    meta_by_track: dict[str, tuple[MetadataPair, ...]] = {}
    sft_by_track: dict[str, tuple[SftExample, ...]] = {}
    validation_pairs_by_track: dict[str, tuple[StrictPair, ...]] = {}
    views_config = contract.views
    for track in TRACKS:
        data = track_data[track]
        strict = derive_pair_strict(
            data.aggregates["train"],
            data.pools["train"],
            manifest,
            split="train",
            min_agreement=float(str(views_config["pair_strict"]["min_agreement"])),
            min_strength=float(str(views_config["pair_strict"]["min_strength"])),
        )
        weights = pair_weights(
            strict,
            strategy=str(views_config["weighting"]["strategy"]),
            cap_per_clip=int(str(views_config["weighting"]["cap_per_clip"])),
        )
        strict = with_weights(strict, weights)
        meta = derive_pair_all(data.aggregates["train"], data.pools["train"], manifest, split="train")
        sft_rows = derive_sft(
            data.aggregates["train"],
            data.annotations["train"],
            data.pools["train"],
            manifest,
            data.audits["train"],
            split="train",
            min_agreement=float(str(views_config["sft"]["min_agreement"])),
            min_confidence=float(str(views_config["sft"]["min_confidence"])),
        )
        validation_pairs = derive_pair_strict(
            data.aggregates["validation"],
            data.pools["validation"],
            manifest,
            split="validation",
            min_agreement=float(str(views_config["pair_strict"]["min_agreement"])),
            min_strength=float(str(views_config["pair_strict"]["min_strength"])),
        )
        if not strict or not sft_rows or not validation_pairs:
            raise CanaryError(f"track {track!r} produced an empty training or validation view")
        combined_pool = freeze_pool(
            dataset_version="canary-combined/v1",
            track=track,
            evidence_audit_version="audit/v1",
            candidates=(*data.pools["train"].candidates, *data.pools["validation"].candidates),
            pairs=(*data.pools["train"].pairs, *data.pools["validation"].pairs),
        )
        run_leakage_audit(
            manifest=manifest,
            clips=clips,
            pool=combined_pool,
            sft_rows=sft_rows,
            enforce=True,
        )
        noise = estimate_natural_noise(data.annotations["train"], data.aggregates["train"])
        data.view_artifacts["sft"] = publisher.publish(
            "dpo.sft-view/v1",
            {"schema": "dpo.sft-view/v1", "rows": [row.document() for row in sft_rows]},
            parents=(ParentEdge(data.pool_artifacts["train"], "frozen-pool"),),
            stage="views",
            parameters={"operation": "derive-sft", "track": track},
            row_count=len(sft_rows),
            clips={row.clip_id for row in sft_rows},
            role_exposure={"train"},
        )
        data.view_artifacts["pair_strict"] = publisher.publish(
            "dpo.pair-strict-view/v1",
            {"schema": "dpo.pair-strict-view/v1", "rows": [row.document() for row in strict]},
            parents=(ParentEdge(data.pool_artifacts["train"], "frozen-pool"),),
            stage="views",
            parameters={"operation": "derive-pairs-strict", "track": track},
            row_count=len(strict),
            clips={row.clip_id for row in strict},
            role_exposure={"train"},
        )
        data.view_artifacts["pair_all"] = publisher.publish(
            "dpo.pair-all-view/v1",
            {"schema": "dpo.pair-all-view/v1", "rows": [row.document() for row in meta]},
            parents=(ParentEdge(data.pool_artifacts["train"], "frozen-pool"),),
            stage="views",
            parameters={"operation": "derive-pairs-all", "track": track},
            row_count=len(meta),
            clips={row.clip_id for row in meta},
            role_exposure={"train"},
        )
        data.view_artifacts["validation_pairs"] = publisher.publish(
            "dpo.validation-pairs/v1",
            {
                "schema": "dpo.validation-pairs/v1",
                "rows": [row.document() for row in validation_pairs],
            },
            parents=(ParentEdge(data.pool_artifacts["validation"], "frozen-pool"),),
            stage="views",
            parameters={"operation": "derive-validation-pairs", "track": track},
            row_count=len(validation_pairs),
            clips={row.clip_id for row in validation_pairs},
            role_exposure={"validation"},
        )
        publisher.publish(
            "dpo.noise-calibration/v1",
            noise.document(),
            parents=(ParentEdge(data.view_artifacts["pair_strict"], "pair-view"),),
            stage="views",
            parameters={"operation": "noise-calibration", "track": track},
            clips={row.clip_id for row in strict},
            role_exposure={"train"},
        )
        for rate in [float(str(rate)) for rate in contract.robustness["flip_rates"]]:
            flip_manifest = make_flip_manifest(
                strict, flip_rate=rate, seed=int(str(contract.robustness["flip_seed"]))
            )
            publisher.publish(
                "dpo.flip-manifest/v1",
                flip_manifest.document(),
                parents=(ParentEdge(data.view_artifacts["pair_strict"], "pair-view"),),
                stage="views",
                parameters={"operation": "flip-manifest", "track": track, "rate": rate},
                clips={row.clip_id for row in strict},
                role_exposure={"train"},
            )
        strict_by_track[track] = strict
        meta_by_track[track] = meta
        sft_by_track[track] = sft_rows
        validation_pairs_by_track[track] = validation_pairs

    # The nine-condition matrix on the tiny backend, canonical seed only (the
    # canary proves the executable boundary; the full 3-seed sweep is a real run).
    canonical_seed = int(str(contract.training["canonical_seed"]))
    runner = OfflineMatrixRunner(
        contract=contract,
        strict_pairs={track: tuple(strict_by_track[track]) for track in TRACKS},
        metadata_pairs={track: tuple(meta_by_track[track]) for track in TRACKS},
        sft_rows={track: tuple(sft_by_track[track]) for track in TRACKS},
    )
    variants_by_experiment = {
        experiment_id: expand_experiment(contract, experiment_id) for experiment_id in EXPERIMENT_IDS
    }
    cells: dict[tuple[str, str, str], CellResult] = {}
    cell_artifacts: dict[tuple[str, str, str], str] = {}
    for experiment_id in EXPERIMENT_IDS:
        for variant in variants_by_experiment[experiment_id]:
            for track in TRACKS:
                cell = runner.run_cell(experiment_id, track=track, seed=canonical_seed, variant=variant)
                cells[(experiment_id, variant.variant_id, track)] = cell
                parent_edges: list[ParentEdge] = []
                data = track_data[track]
                if experiment_id == "SFT":
                    parent_edges.append(ParentEdge(data.view_artifacts["sft"], "training-view"))
                elif experiment_id == "SFT_DPO":
                    parent_edges.append(ParentEdge(data.view_artifacts["sft"], "warm-start-view"))
                    parent_edges.append(ParentEdge(data.view_artifacts["pair_strict"], "training-view"))
                elif cell.trained:
                    view_key = "pair_all" if cell.training_view == "pair_all" else "pair_strict"
                    parent_edges.append(ParentEdge(data.view_artifacts[view_key], "training-view"))
                cell_clips = {row.clip_id for row in strict_by_track[track]} if cell.trained else set()
                cell_artifacts[(experiment_id, variant.variant_id, track)] = publisher.publish(
                    "dpo.matrix-cell/v1",
                    cell.document(),
                    parents=tuple(parent_edges),
                    stage="train",
                    # Per-variant cache identity: the cell keys on exactly what it
                    # trained from, so extending a sweep axis leaves every sibling
                    # variant's request identity untouched.
                    slice_override={
                        "training": dict(contract.raw["training"]),
                        "models_seed": dict(contract.raw["models"]["seed"]),
                        "track_contract": dict(contract.raw["tracks"][track]),
                        "experiment": {
                            "experiment_id": experiment_id,
                            "variant_id": variant.variant_id,
                            "hyperparameters": dict(cell.hyperparameters),
                            "training_view": cell.training_view,
                        },
                    },
                    parameters={
                        "operation": "train-cell",
                        "experiment_id": experiment_id,
                        "variant": variant.variant_id,
                        "track": track,
                        "seed": canonical_seed,
                    },
                    seed=canonical_seed,
                    clips=cell_clips,
                    role_exposure={"train"} if cell_clips else None,
                    fit_exposure=cell_clips or None,
                    purpose="training" if cell.trained else "research",
                )

    # Common validation scoring: every experiment variant is scored identically.
    validation_reports: dict[str, dict[str, dict[str, float]]] = {}
    for track in TRACKS:
        seed_adapter = runner.seed_adapter(track)
        sft_adapter = runner.policies[("SFT", "base", track, canonical_seed)]
        report_by_experiment: dict[str, dict[str, float]] = {}
        for experiment_id in EXPERIMENT_IDS:
            report_by_variant: dict[str, float] = {}
            for variant in variants_by_experiment[experiment_id]:
                policy_adapter = runner.policies[(experiment_id, variant.variant_id, track, canonical_seed)]
                reference_adapter = sft_adapter if experiment_id == "SFT_DPO" else seed_adapter
                scored = []
                for validation_pair in validation_pairs_by_track[track]:
                    media = synthetic_media(track, [validation_pair.clip_id], media_dim=DEFAULT_MEDIA_DIM)
                    chosen_batch = CompletionBatch(texts=(validation_pair.chosen_text,))
                    rejected_batch = CompletionBatch(texts=(validation_pair.rejected_text,))
                    with torch.no_grad():
                        policy_chosen = float(policy_adapter.completion_logps(media, chosen_batch)[0])
                        policy_rejected = float(policy_adapter.completion_logps(media, rejected_batch)[0])
                        ref_chosen = float(reference_adapter.completion_logps(media, chosen_batch)[0])
                        ref_rejected = float(reference_adapter.completion_logps(media, rejected_batch)[0])
                    scored.append(
                        ScoredPair(
                            pair_id=validation_pair.pair_id,
                            clip_id=validation_pair.clip_id,
                            track=track,
                            policy_chosen_logp=policy_chosen,
                            policy_rejected_logp=policy_rejected,
                            ref_chosen_logp=ref_chosen,
                            ref_rejected_logp=ref_rejected,
                            difficulty=validation_pair.difficulty,
                            agreement=validation_pair.agreement,
                        )
                    )
                # SEED and SFT carry no trained beta; score them at beta=1.0
                # (accuracy is beta-invariant; beta only scales the probability fields).
                hyper = variant.hyperparameters
                beta = float(str(hyper["beta"])) if "beta" in hyper else 1.0
                preference_report = evaluate_preferences(scored, beta=beta)
                report_by_variant[variant.variant_id] = preference_report.accuracy
            report_by_experiment[experiment_id] = report_by_variant
        validation_reports[track] = report_by_experiment
    validation_artifact = publisher.publish(
        "dpo.validation-report/v1",
        {"schema": "dpo.validation-report/v1", "accuracy": validation_reports},
        parents=tuple(
            ParentEdge(track_data[track].view_artifacts["validation_pairs"], "validation-pairs")
            for track in TRACKS
        )
        + tuple(ParentEdge(artifact_id, "matrix-cell") for artifact_id in cell_artifacts.values()),
        stage="validate",
        parameters={"operation": "validate"},
        clips={row.clip_id for track in TRACKS for row in validation_pairs_by_track[track]}
        | {row.clip_id for track in TRACKS for row in strict_by_track[track]},
        role_exposure={"train", "validation"},
        selection_exposure={row.clip_id for track in TRACKS for row in validation_pairs_by_track[track]},
    )
    # Selection: one winning variant per experiment and track (max validation
    # accuracy, lexical variant-id tie-break), then experiments ranked by their
    # winner. Only selected variants reach the lock, the test, and the study.
    selected_variants: dict[str, dict[str, str]] = {}
    for track in TRACKS:
        selected_variants[track] = {}
        for experiment_id in EXPERIMENT_IDS:
            per_variant = validation_reports[track][experiment_id]
            selected_variants[track][experiment_id] = sorted(
                per_variant, key=lambda variant_id: (-per_variant[variant_id], variant_id)
            )[0]
    ranking = {
        track: sorted(
            EXPERIMENT_IDS,
            key=lambda experiment_id: (
                -validation_reports[track][experiment_id][selected_variants[track][experiment_id]],
                experiment_id,
            ),
        )
        for track in TRACKS
    }
    selection_artifact = publisher.publish(
        "dpo.selection-report/v1",
        {
            "schema": "dpo.selection-report/v1",
            "ranking": ranking,
            "selected_variants": selected_variants,
            "selected_hyperparameters": {
                track: {
                    experiment_id: dict(
                        cells[(experiment_id, selected_variants[track][experiment_id], track)].hyperparameters
                    )
                    for experiment_id in EXPERIMENT_IDS
                }
                for track in TRACKS
            },
            "canonical_seed": canonical_seed,
            "note": "canary selection: validation preference accuracy, lexical tie-break",
        },
        parents=(ParentEdge(validation_artifact, "validation-report"),),
        stage="validate",
        parameters={"operation": "selection"},
    )

    lock_manifest = create_lock_manifest(
        contract,
        checkpoint_hashes={
            experiment_id: {
                track: cells[
                    (experiment_id, selected_variants[track][experiment_id], track)
                ].checkpoint_signature
                for track in TRACKS
            }
            for experiment_id in EXPERIMENT_IDS
        },
        processor_hash=semantic_hash({"processor": "tiny-byte/v1"}),
        preprocessing_hash=semantic_hash({"media": "synthetic/v1", "media_dim": DEFAULT_MEDIA_DIM}),
        evaluation_version="evaluation/v1",
        metric_versions={"compliance": "v1", "preference": "v1", "factuality": "v1"},
        study_interface_version="study-export/v1",
        selection_report_hash=selection_artifact,
    )
    lock_artifact = publisher.publish(
        "dpo.lock-manifest/v1",
        lock_manifest.document,
        parents=(ParentEdge(selection_artifact, "selection-report"),),
        stage="lock",
        parameters={"operation": "lock"},
        attributes={"lock_id": lock_manifest.lock_id},
    )

    # Test-once: reserve, fence, one official protected read, generate, finalize.
    resolver = TestOnceResolver(str(Path(workspace) / "test.sqlite"))
    test_split_hash = test_split_semantic_hash(test_clips)
    reservation = resolver.reserve(lock_manifest, test_split_hash=test_split_hash)
    resumed = resolver.reserve(lock_manifest, test_split_hash=test_split_hash)
    if resumed != reservation:
        raise CanaryError("an identical test reservation must resume, not fork")
    try:
        resolver.reserve(lock_manifest, test_split_hash=semantic_hash({"other": True}))
    except LockError:
        pass
    else:
        raise CanaryError("a different test semantic must be rejected after reservation")
    resolver.mark_first_read(reservation)
    authority = ProtectedAccessAuthority(Path(workspace) / "test.sqlite")
    test_capability = authority.reserve(
        scope="confirmatory-test", semantic_hash=reservation.semantic_hash, roles={"test"}
    )
    for clip_id in test_clips:
        store.read_payload(
            shard_ids[clip_id],
            capability=test_capability,
            authority=authority,
            semantic_hash=reservation.semantic_hash,
        )
    reservation_artifact = publisher.publish(
        "dpo.test-reservation/v1",
        {"schema": "dpo.test-reservation/v1", **resolver.authority_record(reservation)},
        parents=(ParentEdge(lock_artifact, "lock-manifest"),)
        + tuple(ParentEdge(shard_ids[clip_id], "test-shard") for clip_id in test_clips),
        stage="test",
        parameters={"operation": "test-reserve"},
        clips=set(test_clips),
        role_exposure={"test"},
        test_exposure=set(test_clips),
    )
    test_metrics: dict[str, dict[str, dict[str, object]]] = {}
    validation_decoding = contract.validation
    for track in TRACKS:
        per_model: dict[str, dict[str, object]] = {}
        media = synthetic_media(track, test_clips, media_dim=DEFAULT_MEDIA_DIM)
        for experiment_id in EXPERIMENT_IDS:
            policy_adapter = runner.policies[
                (experiment_id, selected_variants[track][experiment_id], track, canonical_seed)
            ]
            texts = policy_adapter.generate(
                media,
                temperature=float(str(validation_decoding["temperature"])),
                top_p=float(str(validation_decoding["top_p"])),
                max_new_tokens=int(str(validation_decoding["max_new_tokens"])),
                seed=canonical_seed,
            )
            captions = tuple(
                GeneratedCaption(clip_id=clip_id, text=text or "(empty)")
                for clip_id, text in zip(test_clips, texts, strict=True)
            )
            per_model[experiment_id] = compute_compliance(captions, contract.tracks[track]).document()
        test_metrics[track] = per_model
    test_metrics_artifact = publisher.publish(
        "dpo.test-metrics/v1",
        {"schema": "dpo.test-metrics/v1", "compliance": test_metrics},
        parents=(ParentEdge(reservation_artifact, "test-reservation"),),
        stage="test",
        parameters={"operation": "test-metrics"},
        clips=set(test_clips),
        role_exposure={"test"},
        test_exposure=set(test_clips),
    )
    resolver.complete(reservation)
    authority.complete(test_capability)
    finalization_artifact = publisher.publish(
        "dpo.test-finalization/v1",
        {"schema": "dpo.test-finalization/v1", **resolver.authority_record(reservation)},
        parents=(
            ParentEdge(reservation_artifact, "test-reservation"),
            ParentEdge(test_metrics_artifact, "test-metrics"),
        ),
        stage="test",
        parameters={"operation": "test-finalize"},
        clips=set(test_clips),
        role_exposure={"test"},
    )

    # Blinded study export and a synthetic analysis pass.
    study_documents = {}
    outcomes: list[PairwiseOutcome] = []
    for track in TRACKS:
        media = synthetic_media(track, study_clips, media_dim=DEFAULT_MEDIA_DIM)
        captions_by_model: dict[str, dict[str, str]] = {}
        for experiment_id in EXPERIMENT_IDS:
            policy_adapter = runner.policies[
                (experiment_id, selected_variants[track][experiment_id], track, canonical_seed)
            ]
            # The one frozen decoding budget (contract [validation]) serves every
            # model in every evaluation surface, the study export included.
            texts = policy_adapter.generate(
                media,
                temperature=float(str(validation_decoding["temperature"])),
                top_p=float(str(validation_decoding["top_p"])),
                max_new_tokens=int(str(validation_decoding["max_new_tokens"])),
                seed=canonical_seed,
            )
            captions_by_model[experiment_id] = {
                clip_id: (text or "(empty)") for clip_id, text in zip(study_clips, texts, strict=True)
            }
        export = build_study_export(
            track=track,
            captions=captions_by_model,
            media_hashes={
                clip.clip_id: clip.media_hash for clip in clips if clip.clip_id in set(study_clips)
            },
            exposures_per_pair=int(str(contract.study["exposures_per_model_pair"])),
        )
        study_documents[track] = export.public_documents()
        for task in export.tasks:
            identities = export.randomization[task.task_id]
            flip = int(semantic_hash({"outcome": task.task_id}).removeprefix("sha256:")[:2], 16) % 3
            winner = None if flip == 2 else str(identities["model_a"] if flip == 0 else identities["model_b"])
            outcomes.append(
                PairwiseOutcome(
                    model_a=str(identities["model_a"]),
                    model_b=str(identities["model_b"]),
                    winner=winner,
                    clip_id=task.clip_id,
                )
            )
    study_export_artifact = publisher.publish(
        "dpo.study-export/v1",
        {"schema": "dpo.study-export/v1", "tracks": study_documents},
        parents=(ParentEdge(lock_artifact, "lock-manifest"),)
        + tuple(ParentEdge(shard_ids[clip_id], "study-shard") for clip_id in study_clips),
        stage="study",
        parameters={"operation": "study-export"},
        clips=set(study_clips),
        role_exposure={"study"},
    )
    fit = fit_bradley_terry(outcomes)
    top_model = fit.ranking()[0]
    point, interval = clip_cluster_bootstrap(
        outcomes,
        lambda outcome: outcome.clip_id,
        lambda rows: sum(1.0 for row in rows if row.winner == top_model) / len(rows),
        samples=int(str(contract.analysis["bootstrap_samples"])),
        seed=int(str(contract.analysis["seed"])),
    )
    analysis_artifact = publisher.publish(
        "dpo.analysis-report/v1",
        {
            "schema": "dpo.analysis-report/v1",
            "bradley_terry": fit.document(),
            "top_model_win_rate": {"point": point, "interval_95": list(interval)},
        },
        parents=(ParentEdge(study_export_artifact, "study-export"),),
        stage="analysis",
        parameters={"operation": "analysis"},
        clips=set(study_clips),
        role_exposure={"study"},
    )

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
        "study_recruitment": "blocked_pending_external_operation",
        "clips": len(clips),
        "splits": {split: len(manifest.assignments[split]) for split in manifest.assignments},
        "matrix_cells": len(cells),
        "validation_accuracy": validation_reports,
        "selection_ranking": ranking,
        "oracles": {"leakage": leakage_oracle, "tamper": tamper_oracle},
        "provider_calls": provider_calls,
    }
    report_id = publisher.publish(
        REPORT_TYPE,
        report,
        parents=(
            ParentEdge(lock_artifact, "lock-manifest"),
            ParentEdge(finalization_artifact, "test-finalization"),
            ParentEdge(analysis_artifact, "analysis-report"),
        ),
        stage="*",
        parameters={"operation": "canary-report"},
        attributes={
            "contract_hash": contract.contract_hash,
            "lock_id": lock_id,
            "source_identity": source_identity,
        },
    )
    return CanaryResult(artifact_id=report_id, report=report, cached=False, provider_calls=provider_calls)
