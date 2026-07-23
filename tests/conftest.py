"""Shared fixtures: a validated contract and a small frozen preference world."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from dpo.annotation.aggregate import PairAggregate, aggregate_all
from dpo.annotation.raw_annotations import RawAnnotation
from dpo.candidates.audit import ResolvedCandidateAudit, audit_candidate, resolve_audits
from dpo.candidates.candidate_records import CollectionPolicy, GenerationConfig, build_candidate_records
from dpo.candidates.freeze import FrozenCandidatePool, freeze_pool
from dpo.candidates.pair_sampler import sample_pairs
from dpo.contracts.study_contract import StudyContract, load_contract
from dpo.core.identity import sha256_bytes
from dpo.data.derive_pairs import derive_pair_all, derive_pair_strict
from dpo.data.derive_sft import derive_sft
from dpo.data.split import ClipInput, SplitManifest, assign_splits
from dpo.evidence.claim_ledger import Claim, ClaimLedger
from dpo.pipeline.run_matrix import OfflineMatrixRunner

REPO_ROOT = Path(__file__).parents[1]
CANARY_CONTRACT = REPO_ROOT / "configs" / "study" / "canary.toml"

_SUBJECTS = ("cyclist", "gardener", "vendor", "skater", "painter", "runner", "climber", "juggler")
_OBJECTS = ("junction", "seedlings", "stall", "ramp", "mural", "track", "wall", "pins")


@pytest.fixture(scope="session")
def contract() -> StudyContract:
    return load_contract(CANARY_CONTRACT)


@dataclass(frozen=True)
class PreferenceWorld:
    contract: StudyContract
    clips: tuple[ClipInput, ...]
    manifest: SplitManifest
    ledgers: dict[str, ClaimLedger]
    pool: FrozenCandidatePool
    audits: dict[str, ResolvedCandidateAudit]
    annotations: tuple[RawAnnotation, ...]
    aggregates: tuple[PairAggregate, ...]


def build_world(contract: StudyContract, *, track: str = "visual") -> PreferenceWorld:
    clips = tuple(
        ClipInput(
            clip_id=f"clip-{index:03d}",
            source_video_id=f"src-{index // 2:02d}",
            media_hash=sha256_bytes(f"media-{index}".encode()),
            start_ms=0,
            end_ms=8000,
            derivative_hashes=(sha256_bytes(f"deriv-{index}".encode()),),
        )
        for index in range(24)
    )
    manifest = assign_splits(
        clips, seed=int(str(contract.corpus["split_seed"])), fractions=contract.split_fractions
    )
    policy = CollectionPolicy(policy_id="C0", checkpoint_hash=sha256_bytes(b"c0"))
    config = GenerationConfig(temperature=0.7, top_p=0.9, max_new_tokens=48, seed=1)
    train_clips = list(manifest.assignments["train"])[:6]
    candidates = []
    audit_rows = []
    ledgers: dict[str, ClaimLedger] = {}
    for rank, clip_id in enumerate(train_clips):
        subject = _SUBJECTS[rank % len(_SUBJECTS)]
        target = _OBJECTS[rank % len(_OBJECTS)]
        site = f"site{rank:02d}x"
        ledger = ClaimLedger(
            clip_id=clip_id,
            track=track,
            audit_version="audit/v1",
            prohibited_cross_modal_claims=(),
            claims=(
                Claim(
                    claim_id=f"claim-{rank}-a",
                    type="object",
                    canonical_form=f"{subject} at the {target}",
                    start_ms=0,
                    end_ms=4000,
                    support_sources=("detector",),
                    support_confidence=0.9,
                    human_status="supported",
                ),
                Claim(
                    claim_id=f"claim-{rank}-b",
                    type="action",
                    canonical_form=f"{subject} crossing slowly",
                    start_ms=0,
                    end_ms=6000,
                    support_sources=("detector",),
                    support_confidence=0.7,
                    human_status="contradicted",
                ),
            ),
        )
        ledgers[clip_id] = ledger
        records = build_candidate_records(
            clip_id=clip_id,
            track=track,
            policy=policy,
            generations=[
                ("greedy", f"The {subject} waits at the {target} near {site}.", config),
                ("sample", f"Near {site}, the {subject} pauses by the {target}.", config),
                ("sample", f"A {subject} moves past the {target} toward {site}.", config),
                ("controlled_error", f"The {subject} crossing slowly leaves the {target} at {site}.", config),
            ],
        )
        candidates.extend(records)
        for record in records:
            audit_rows.append(audit_candidate(record, contract=contract.tracks[track], ledger=ledger))
    audits = dict(resolve_audits(audit_rows, []))
    pairs = sample_pairs(
        candidates,
        audits,
        per_clip_min=int(str(contract.pairs["per_clip_min"])),
        per_clip_max=int(str(contract.pairs["per_clip_max"])),
        seed=int(str(contract.pairs["seed"])),
        near_duplicate_jaccard=float(str(contract.pairs["near_duplicate_jaccard"])),
    )
    pool = freeze_pool(
        dataset_version="test/v1",
        track=track,
        evidence_audit_version="audit/v1",
        candidates=candidates,
        pairs=pairs,
    )
    annotations = []
    counter = 0
    for index, pair in enumerate(pool.pairs):
        for annotator in range(3):
            flip = (index + annotator) % 2 == 1
            display = (pair.candidate_b, pair.candidate_a) if flip else (pair.candidate_a, pair.candidate_b)
            winner = min(pair.candidate_a, pair.candidate_b)
            choice = "a_better" if display[0] == winner else "b_better"
            annotations.append(
                RawAnnotation(
                    annotation_id=f"ann-{counter:05d}",
                    pair_id=pair.pair_id,
                    clip_id=pair.clip_id,
                    track=track,
                    candidate_a=pair.candidate_a,
                    candidate_b=pair.candidate_b,
                    display_order=display,
                    choice=choice,
                    tie_subtype=None,
                    preference_strength=4,
                    confidence=4,
                    reason_tags=("factual_support",),
                    annotator_id_hash=sha256_bytes(f"annotator-{annotator}".encode()),
                    replay_count=1,
                    response_time_ms=9000,
                    collection_version="1.0",
                )
            )
            counter += 1
    aggregates = aggregate_all(annotations, minimum_judgments=3)
    return PreferenceWorld(
        contract=contract,
        clips=clips,
        manifest=manifest,
        ledgers=ledgers,
        pool=pool,
        audits=audits,
        annotations=tuple(annotations),
        aggregates=aggregates,
    )


@pytest.fixture(scope="session")
def world(contract: StudyContract) -> PreferenceWorld:
    return build_world(contract)


def build_offline_runner(
    world: PreferenceWorld, contract: StudyContract | None = None
) -> OfflineMatrixRunner:
    """One matrix runner over the world's train views, thresholds from the contract.

    Shared by the golden harness and the matrix tests so both always exercise
    the same derived training views — a threshold change cannot silently
    diverge them. ``contract`` overrides the world's contract (e.g. a sweep
    mutation of the canary contract) while reusing the world's frozen data.
    """
    active = contract if contract is not None else world.contract
    views = active.views
    sft_gate = views["sft"]
    strict_gate = views["pair_strict"]
    strict = derive_pair_strict(
        world.aggregates,
        world.pool,
        world.manifest,
        split="train",
        min_agreement=float(str(strict_gate["min_agreement"])),
        min_strength=float(str(strict_gate["min_strength"])),
    )
    meta = derive_pair_all(world.aggregates, world.pool, world.manifest, split="train")
    sft_rows = derive_sft(
        world.aggregates,
        world.annotations,
        world.pool,
        world.manifest,
        world.audits,
        split="train",
        min_agreement=float(str(sft_gate["min_agreement"])),
        min_confidence=float(str(sft_gate["min_confidence"])),
    )
    return OfflineMatrixRunner(
        contract=active,
        strict_pairs={"visual": strict},
        metadata_pairs={"visual": meta},
        sft_rows={"visual": sft_rows},
    )
