"""The frozen candidate pool: what becomes immutable when annotation begins."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from dpo.candidates.candidate_records import CandidateError, CandidateRecord, parse_candidate_record
from dpo.candidates.pair_sampler import PairRecord, parse_pair_record
from dpo.contracts.study_contract import TRACKS
from dpo.core.identity import semantic_hash


@dataclass(frozen=True)
class FrozenCandidatePool:
    """Candidate text, ids, pair mapping, source checkpoint, decoding config,
    and evidence-audit version, frozen under one content hash.

    Any change to any of these requires a new ``dataset_version`` and a new
    collection round; the pool hash is what annotation and view artifacts pin.
    """

    dataset_version: str
    track: str
    evidence_audit_version: str
    candidates: tuple[CandidateRecord, ...]
    pairs: tuple[PairRecord, ...]

    def __post_init__(self) -> None:
        if self.track not in TRACKS:
            raise CandidateError(f"pool track must be one of {sorted(TRACKS)}")
        if not self.dataset_version.strip() or not self.evidence_audit_version.strip():
            raise CandidateError("pool requires dataset_version and evidence_audit_version")
        candidate_ids = set()
        checkpoints = set()
        for candidate in self.candidates:
            if candidate.track != self.track:
                raise CandidateError("pool candidates must all belong to the pool track")
            if candidate.candidate_id in candidate_ids:
                raise CandidateError(f"duplicate candidate id {candidate.candidate_id!r}")
            candidate_ids.add(candidate.candidate_id)
            checkpoints.add(candidate.checkpoint_hash)
        if len(checkpoints) > 1:
            raise CandidateError("pool candidates must come from one frozen C0 checkpoint")
        pair_ids = set()
        for pair in self.pairs:
            if pair.track != self.track:
                raise CandidateError("pool pairs must all belong to the pool track")
            if pair.pair_id in pair_ids:
                raise CandidateError(f"duplicate pair id {pair.pair_id!r}")
            pair_ids.add(pair.pair_id)
            for candidate_id in (pair.candidate_a, pair.candidate_b):
                if candidate_id not in candidate_ids:
                    raise CandidateError(
                        f"pair {pair.pair_id!r} references unknown candidate {candidate_id!r}"
                    )

    @property
    def pool_hash(self) -> str:
        return semantic_hash(self.document())

    def candidate(self, candidate_id: str) -> CandidateRecord:
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        raise CandidateError(f"unknown candidate id {candidate_id!r}")

    def pair(self, pair_id: str) -> PairRecord:
        for pair in self.pairs:
            if pair.pair_id == pair_id:
                return pair
        raise CandidateError(f"unknown pair id {pair_id!r}")

    def document(self) -> dict[str, object]:
        return {
            "schema": "dpo.frozen-candidate-pool/v1",
            "dataset_version": self.dataset_version,
            "track": self.track,
            "evidence_audit_version": self.evidence_audit_version,
            "candidates": [candidate.document() for candidate in self.candidates],
            "pairs": [pair.document() for pair in self.pairs],
        }


def freeze_pool(
    *,
    dataset_version: str,
    track: str,
    evidence_audit_version: str,
    candidates: Sequence[CandidateRecord],
    pairs: Sequence[PairRecord],
) -> FrozenCandidatePool:
    return FrozenCandidatePool(
        dataset_version=dataset_version,
        track=track,
        evidence_audit_version=evidence_audit_version,
        candidates=tuple(sorted(candidates, key=lambda record: record.candidate_id)),
        pairs=tuple(sorted(pairs, key=lambda record: record.pair_id)),
    )


def parse_frozen_pool(payload: bytes | str | Mapping[str, object]) -> FrozenCandidatePool:
    if isinstance(payload, (bytes, str)):
        try:
            document = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise CandidateError("frozen pool payload is not valid JSON") from exc
    else:
        document = dict(payload)
    if not isinstance(document, dict) or set(document) != {
        "schema",
        "dataset_version",
        "track",
        "evidence_audit_version",
        "candidates",
        "pairs",
    }:
        raise CandidateError("frozen pool payload has an unexpected field set")
    if document["schema"] != "dpo.frozen-candidate-pool/v1":
        raise CandidateError("frozen pool schema is unknown")
    candidates_value = document["candidates"]
    pairs_value = document["pairs"]
    if not isinstance(candidates_value, list) or not isinstance(pairs_value, list):
        raise CandidateError("frozen pool candidates/pairs must be arrays")
    pool = FrozenCandidatePool(
        dataset_version=str(document["dataset_version"]),
        track=str(document["track"]),
        evidence_audit_version=str(document["evidence_audit_version"]),
        candidates=tuple(
            parse_candidate_record(value) for value in candidates_value if isinstance(value, Mapping)
        ),
        pairs=tuple(parse_pair_record(value) for value in pairs_value if isinstance(value, Mapping)),
    )
    if len(pool.candidates) != len(candidates_value) or len(pool.pairs) != len(pairs_value):
        raise CandidateError("frozen pool contains non-object candidate/pair entries")
    return pool


def assert_pool_unchanged(pool: FrozenCandidatePool, *, expected_pool_hash: str) -> None:
    """The freeze condition: after annotation begins, drift is a hard error."""
    if pool.pool_hash != expected_pool_hash:
        raise CandidateError(
            "frozen candidate pool drifted after collection started; a change to candidate text,"
            " ids, pair mapping, checkpoint, decoding, or audit version requires a new dataset"
            " version and a new collection round"
        )
