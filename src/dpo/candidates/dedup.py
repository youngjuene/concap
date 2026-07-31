"""Cross-pool candidate deduplication: enforce the leakage audit early.

``audit_text_leakage`` vetoes, at views-derive time, any candidate text that
near-duplicates a candidate on the other side of a split boundary — after the
annotation hours are already spent. This transform applies the SAME rule to a
pair of frozen pools before anyone annotates: every candidate participating in
a cross-pool near-duplicate collision is dropped, from BOTH sides, and the
surviving candidates are re-paired and re-frozen as new pools.

The threshold is the audit's own; both read one constant, so the filter and
the gate cannot drift apart. No contract knob is involved for the same reason
the audit itself has none: this is code-owned integrity enforcement, not a
result-affecting lever a study may tune.
"""

from __future__ import annotations

from dataclasses import dataclass

from dpo.candidates.audit import content_tokens
from dpo.candidates.candidate_records import CandidateError, CandidateRecord
from dpo.candidates.freeze import FrozenCandidatePool
from dpo.data.leakage_audit import CROSS_SPLIT_JACCARD, _jaccard


@dataclass(frozen=True)
class DedupOutcome:
    kept_a: tuple[CandidateRecord, ...]
    kept_b: tuple[CandidateRecord, ...]
    dropped: tuple[tuple[str, str, str, str], ...]  # (pool, clip_id, candidate_id, text)

    def document(self) -> dict[str, object]:
        return {
            "dropped": [
                {"pool": pool, "clip_id": clip_id, "candidate_id": candidate_id, "text": text}
                for pool, clip_id, candidate_id, text in self.dropped
            ]
        }


def cross_pool_dedup(
    pool_a: FrozenCandidatePool,
    pool_b: FrozenCandidatePool,
    *,
    per_clip_min: int,
    threshold: float = CROSS_SPLIT_JACCARD,
) -> DedupOutcome:
    """Drop every candidate in a cross-pool near-duplicate collision, both sides.

    Within-pool duplicates are deliberately untouched: they never cross a split
    boundary, so the audit does not veto them — and removing them is not even
    representable here, because on this corpus it would push most clips below
    ``per_clip_min`` (measured: 34 of 48 clips).
    """
    tokens = {
        candidate.candidate_id: content_tokens(candidate.text)
        for pool in (pool_a, pool_b)
        for candidate in pool.candidates
    }
    drop: set[str] = set()
    for left in pool_a.candidates:
        for right in pool_b.candidates:
            if left.clip_id == right.clip_id:
                # One clip lives in exactly one split; a same-clip pairing across
                # pools would mean the pools already violate the split manifest.
                raise CandidateError(f"clip {left.clip_id!r} appears in both pools; splits are broken")
            if _jaccard(tokens[left.candidate_id], tokens[right.candidate_id]) >= threshold:
                drop.add(left.candidate_id)
                drop.add(right.candidate_id)

    kept_a = tuple(candidate for candidate in pool_a.candidates if candidate.candidate_id not in drop)
    kept_b = tuple(candidate for candidate in pool_b.candidates if candidate.candidate_id not in drop)
    for pool, kept in ((pool_a, kept_a), (pool_b, kept_b)):
        counts: dict[str, int] = {}
        for candidate in kept:
            counts[candidate.clip_id] = counts.get(candidate.clip_id, 0) + 1
        for candidate in pool.candidates:
            if counts.get(candidate.clip_id, 0) < per_clip_min:
                raise CandidateError(
                    f"dedup would leave clip {candidate.clip_id!r} with"
                    f" {counts.get(candidate.clip_id, 0)} candidates, below per_clip_min={per_clip_min};"
                    " regenerate candidates for this clip instead of cutting"
                )
    rows: list[tuple[str, str, str, str]] = []
    for label, pool in (("a", pool_a), ("b", pool_b)):
        for candidate in pool.candidates:
            if candidate.candidate_id in drop:
                rows.append((label, candidate.clip_id, candidate.candidate_id, candidate.text))
    return DedupOutcome(kept_a=kept_a, kept_b=kept_b, dropped=tuple(sorted(rows)))
