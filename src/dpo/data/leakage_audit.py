"""Leakage rules: group integrity, split provenance, and text-level leakage.

Splits happen before pair construction; all candidates of a clip stay in the
clip's split; all clips of a source video share one split; near-duplicate and
paraphrase candidate text across splits is flagged; validation-endorsed
completions never appear in SFT training rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from dpo.candidates.audit import content_tokens
from dpo.candidates.freeze import FrozenCandidatePool
from dpo.contracts.study_contract import SPLITS
from dpo.data.derive_sft import SftExample
from dpo.data.split import ClipInput, SplitManifest


@dataclass(frozen=True)
class LeakageReport:
    violations: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.violations

    def document(self) -> dict[str, object]:
        return {
            "schema": "dpo.leakage-report/v1",
            "passed": self.passed,
            "violations": list(self.violations),
        }


class LeakageError(ValueError):
    """Raised when a leakage audit is asked to enforce and finds violations."""


# One threshold for the gate and for any transform that pre-enforces it
# (candidates dedup); a shared constant is what keeps them from drifting.
CROSS_SPLIT_JACCARD = 0.8


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def audit_corpus(manifest: SplitManifest, clips: Sequence[ClipInput]) -> list[str]:
    violations: list[str] = []
    by_id = {clip.clip_id: clip for clip in clips}
    assigned = [clip_id for split in SPLITS for clip_id in manifest.assignments[split]]
    if len(assigned) != len(set(assigned)):
        violations.append("a clip is assigned to more than one split")
    missing = sorted(set(by_id) - set(assigned))
    if missing:
        violations.append(f"clip {missing[0]!r} has no split assignment")
    source_split: dict[str, str] = {}
    group_split: dict[str, str] = {}
    for split in SPLITS:
        for clip_id in manifest.assignments[split]:
            clip = by_id.get(clip_id)
            if clip is None:
                violations.append(f"manifest assigns unknown clip {clip_id!r}")
                continue
            previous = source_split.setdefault(clip.source_video_id, split)
            if previous != split:
                violations.append(
                    f"source video {clip.source_video_id!r} spans splits {previous!r} and {split!r}"
                )
            group = manifest.groups.get(clip_id)
            if group is None:
                violations.append(f"clip {clip_id!r} has no group id")
            else:
                previous_group = group_split.setdefault(group, split)
                if previous_group != split:
                    violations.append(f"group {group!r} spans splits {previous_group!r} and {split!r}")
    return violations


def audit_pool(pool: FrozenCandidatePool, manifest: SplitManifest) -> list[str]:
    violations: list[str] = []
    for pair in pool.pairs:
        for candidate_id in (pair.candidate_a, pair.candidate_b):
            candidate = pool.candidate(candidate_id)
            if candidate.clip_id != pair.clip_id:
                violations.append(f"pair {pair.pair_id!r} crosses clips; pairs must be within-clip")
    clip_split: dict[str, str] = {}
    for candidate in pool.candidates:
        try:
            clip_split[candidate.clip_id] = manifest.role_of(candidate.clip_id)
        except Exception:  # noqa: BLE001 - reported, not raised, in audit mode
            violations.append(f"candidate clip {candidate.clip_id!r} is not in the split manifest")
    return violations


def audit_text_leakage(
    pool: FrozenCandidatePool,
    manifest: SplitManifest,
    *,
    jaccard_threshold: float = CROSS_SPLIT_JACCARD,
) -> list[str]:
    """Flag near-duplicate/paraphrase candidate text across split boundaries."""
    violations: list[str] = []
    tokenized: list[tuple[str, str, frozenset[str]]] = []
    for candidate in pool.candidates:
        split = manifest.role_of(candidate.clip_id)
        tokenized.append((candidate.candidate_id, split, content_tokens(candidate.text)))
    for index, (left_id, left_split, left_tokens) in enumerate(tokenized):
        for right_id, right_split, right_tokens in tokenized[index + 1 :]:
            if left_split == right_split:
                continue
            if _jaccard(left_tokens, right_tokens) >= jaccard_threshold:
                violations.append(
                    f"candidates {left_id!r} ({left_split}) and {right_id!r} ({right_split})"
                    " are near-duplicates across a split boundary"
                )
    return violations


def audit_sft_rows(
    rows: Sequence[SftExample],
    manifest: SplitManifest,
    *,
    training_split: str = "train",
) -> list[str]:
    violations: list[str] = []
    for row in rows:
        split = manifest.role_of(row.clip_id)
        if split != training_split:
            violations.append(
                f"SFT example {row.example_id!r} draws its completion from split {split!r};"
                " validation/test/study-endorsed completions never enter SFT training"
            )
    return violations


def run_leakage_audit(
    *,
    manifest: SplitManifest,
    clips: Sequence[ClipInput],
    pool: FrozenCandidatePool | None = None,
    sft_rows: Sequence[SftExample] = (),
    jaccard_threshold: float = CROSS_SPLIT_JACCARD,
    enforce: bool = False,
) -> LeakageReport:
    violations = audit_corpus(manifest, clips)
    if pool is not None:
        violations.extend(audit_pool(pool, manifest))
        violations.extend(audit_text_leakage(pool, manifest, jaccard_threshold=jaccard_threshold))
    if sft_rows:
        violations.extend(audit_sft_rows(sft_rows, manifest))
    report = LeakageReport(violations=tuple(violations))
    if enforce and not report.passed:
        raise LeakageError(f"leakage audit failed: {report.violations[0]}")
    return report
