"""Confound-aware pair sampling over the audited candidate pool.

The sampler considers semantic distance, length difference, modality-purity
difference, lexical overlap, candidate source, and estimated difficulty.
Near-duplicate captions are excluded, length confounds are flagged rather
than silently kept, and category assignment is deterministic so a frozen pool
always regenerates the same pairs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

from dpo.candidates.audit import ResolvedCandidateAudit, content_tokens
from dpo.candidates.candidate_records import CandidateError, CandidateRecord
from dpo.contracts.captions import caption_words
from dpo.contracts.study_contract import PAIR_CATEGORIES
from dpo.core.identity import semantic_hash


@dataclass(frozen=True)
class PairFeatures:
    lexical_overlap: float
    semantic_distance: float
    length_difference: int
    modality_purity_difference: int
    length_confounded: bool
    difficulty: float

    def document(self) -> dict[str, object]:
        return {
            "lexical_overlap": self.lexical_overlap,
            "semantic_distance": self.semantic_distance,
            "length_difference": self.length_difference,
            "modality_purity_difference": self.modality_purity_difference,
            "length_confounded": self.length_confounded,
            "difficulty": self.difficulty,
        }


@dataclass(frozen=True)
class PairRecord:
    """One unordered candidate pair. Orientation is by candidate id, never by preference."""

    pair_id: str
    clip_id: str
    track: str
    candidate_a: str
    candidate_b: str
    category: str
    features: PairFeatures

    def __post_init__(self) -> None:
        if self.candidate_a >= self.candidate_b:
            raise CandidateError("pair candidates must be in canonical id order (a < b)")
        if self.category not in PAIR_CATEGORIES:
            raise CandidateError(f"pair category must be one of {sorted(PAIR_CATEGORIES)}")

    def document(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "clip_id": self.clip_id,
            "track": self.track,
            "candidate_a": self.candidate_a,
            "candidate_b": self.candidate_b,
            "category": self.category,
            "features": self.features.document(),
        }


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _categorize(
    features: PairFeatures,
    first_audit: ResolvedCandidateAudit,
    second_audit: ResolvedCandidateAudit,
) -> str:
    if features.modality_purity_difference > 0:
        return "factuality_contrast"
    if features.length_confounded:
        return "specificity_contrast"
    if first_audit.acceptable != second_audit.acceptable:
        return "quality_contrast"
    if features.lexical_overlap >= 0.6 and features.length_difference <= 2:
        return "ambiguous_near_tie"
    return "style_contrast"


def pair_features(
    first: CandidateRecord,
    second: CandidateRecord,
    first_audit: ResolvedCandidateAudit,
    second_audit: ResolvedCandidateAudit,
    *,
    length_confound_words: int = 5,
) -> PairFeatures:
    first_tokens = content_tokens(first.text)
    second_tokens = content_tokens(second.text)
    overlap = _jaccard(first_tokens, second_tokens)
    length_difference = abs(len(caption_words(first.text)) - len(caption_words(second.text)))
    purity_difference = abs(int(first_audit.modality_violation) - int(second_audit.modality_violation))
    # A pair is hard when the candidates are lexically close and neither the
    # modality-purity screen nor a length gap separates them: separation is
    # 1.0 on any purity difference, otherwise length_difference / 10 capped at
    # 1.0, and difficulty = overlap * (1 - separation), clamped to [0, 1].
    separation = min(1.0, purity_difference + length_difference / 10)
    difficulty = max(0.0, min(1.0, overlap * (1.0 - separation)))
    return PairFeatures(
        lexical_overlap=overlap,
        semantic_distance=1.0 - overlap,
        length_difference=length_difference,
        modality_purity_difference=purity_difference,
        length_confounded=length_difference >= length_confound_words,
        difficulty=difficulty,
    )


def sample_pairs(
    candidates: Sequence[CandidateRecord],
    audits: Mapping[str, ResolvedCandidateAudit],
    *,
    per_clip_min: int,
    per_clip_max: int,
    seed: int,
    near_duplicate_jaccard: float,
) -> tuple[PairRecord, ...]:
    """Deterministically sample pairs per clip with category diversity."""
    if per_clip_min < 1 or per_clip_max < per_clip_min:
        raise CandidateError("pair sampling bounds must satisfy 1 <= per_clip_min <= per_clip_max")
    by_clip: dict[str, list[CandidateRecord]] = {}
    for candidate in candidates:
        if candidate.candidate_id not in audits:
            raise CandidateError(f"candidate {candidate.candidate_id!r} has no resolved audit")
        by_clip.setdefault(candidate.clip_id, []).append(candidate)
    pairs: list[PairRecord] = []
    for clip_id in sorted(by_clip):
        clip_candidates = sorted(by_clip[clip_id], key=lambda record: record.candidate_id)
        tracks = {record.track for record in clip_candidates}
        if len(tracks) != 1:
            raise CandidateError(f"clip {clip_id!r} mixes candidate tracks; pairs are single-track")
        track = next(iter(tracks))
        candidate_pairs: list[PairRecord] = []
        for first, second in combinations(clip_candidates, 2):
            first_audit = audits[first.candidate_id]
            second_audit = audits[second.candidate_id]
            features = pair_features(first, second, first_audit, second_audit)
            if features.lexical_overlap >= near_duplicate_jaccard:
                continue
            category = _categorize(features, first_audit, second_audit)
            pair_hash = semantic_hash(
                {
                    "clip_id": clip_id,
                    "candidate_a": first.candidate_id,
                    "candidate_b": second.candidate_id,
                }
            ).removeprefix("sha256:")[:12]
            candidate_pairs.append(
                PairRecord(
                    pair_id=f"pair-{pair_hash}",
                    clip_id=clip_id,
                    track=track,
                    candidate_a=first.candidate_id,
                    candidate_b=second.candidate_id,
                    category=category,
                    features=features,
                )
            )
        if len(candidate_pairs) < per_clip_min:
            raise CandidateError(
                f"clip {clip_id!r} yields {len(candidate_pairs)} non-duplicate pairs;"
                f" the contract requires at least {per_clip_min}"
            )
        # Deterministic diversity: round-robin over categories, both orders
        # seeded by the contract pair seed.
        ordering = sorted(
            candidate_pairs,
            key=lambda pair: semantic_hash({"seed": seed, "pair": pair.pair_id}),
        )
        by_category: dict[str, list[PairRecord]] = {}
        for pair in ordering:
            by_category.setdefault(pair.category, []).append(pair)
        category_order = sorted(
            by_category,
            key=lambda category: semantic_hash({"seed": seed, "clip": clip_id, "category": category}),
        )
        selected: list[PairRecord] = []
        while len(selected) < per_clip_max:
            progressed = False
            for category in category_order:
                bucket = by_category[category]
                if bucket and len(selected) < per_clip_max:
                    selected.append(bucket.pop(0))
                    progressed = True
            if not progressed:
                break
        pairs.extend(sorted(selected, key=lambda pair: pair.pair_id))
    return tuple(pairs)


def parse_pair_record(value: Mapping[str, object]) -> PairRecord:
    expected = {"pair_id", "clip_id", "track", "candidate_a", "candidate_b", "category", "features"}
    if set(value) != expected:
        raise CandidateError("pair record has an unexpected field set")
    features_value = value["features"]
    if not isinstance(features_value, Mapping) or set(features_value) != {
        "lexical_overlap",
        "semantic_distance",
        "length_difference",
        "modality_purity_difference",
        "length_confounded",
        "difficulty",
    }:
        raise CandidateError("pair features have an unexpected field set")

    def _num(key: str) -> float:
        item = features_value[key]
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise CandidateError(f"pair feature {key} must be a number")
        return float(item)

    def _int(key: str) -> int:
        item = features_value[key]
        if type(item) is not int:
            raise CandidateError(f"pair feature {key} must be an integer")
        return int(item)

    confounded = features_value["length_confounded"]
    if type(confounded) is not bool:
        raise CandidateError("pair feature length_confounded must be a boolean")
    return PairRecord(
        pair_id=str(value["pair_id"]),
        clip_id=str(value["clip_id"]),
        track=str(value["track"]),
        candidate_a=str(value["candidate_a"]),
        candidate_b=str(value["candidate_b"]),
        category=str(value["category"]),
        features=PairFeatures(
            lexical_overlap=_num("lexical_overlap"),
            semantic_distance=_num("semantic_distance"),
            length_difference=_int("length_difference"),
            modality_purity_difference=_int("modality_purity_difference"),
            length_confounded=bool(confounded),
            difficulty=_num("difficulty"),
        ),
    )
