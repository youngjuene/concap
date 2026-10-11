"""Recorded gate outcomes that later stages look up (PRD §8 ``AnalysisDecision``).

A gate outcome is a record, not a mood: which gate, pass, fail, hold or waived, under
which rule, on which run manifest and model profile, decided by whom and when.
Steps that depend on a gate find the record and refuse without it. AC-20: the
lattice generator and the bundle builder refuse to run unless a passing G0
decision exists for a run manifest with the current model profile.

``waived`` (2026-10-11) records that the researcher skipped a gate's human
step on purpose. It is never a pass: a step accepts it only when its caller
says so explicitly (``accept``), so the record of what was and was not
checked survives.

The PRD's own §8 field list was not available when this was written. The fields
here follow the brief (gate, outcome, run-manifest hash) and the schema is
versioned, so aligning it later never silently reinterprets an old decision.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from diff.study_protocol import sha256_of

DECISION_SCHEMA: Final = "diff.analysis-decision/v1"
OUTCOMES: Final = ("pass", "fail", "hold", "waived")
# What the live phase-2 path and the phase-1 bundle accept from G0 (researcher, 2026-10-11).
G0_PASS_OR_WAIVER: Final = ("pass", "waived")


@dataclass(frozen=True)
class AnalysisDecision:
    gate: str
    outcome: str
    rule: Mapping[str, Any]
    evidence: Mapping[str, Any]
    run_manifest_sha256: str
    model_profile_id: str
    decided_by: str
    decided_at: str
    notes: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    def body(self) -> dict[str, Any]:
        return {
            "schema": DECISION_SCHEMA,
            "gate": self.gate,
            "outcome": self.outcome,
            "rule": dict(self.rule),
            "evidence": dict(self.evidence),
            "run_manifest_sha256": self.run_manifest_sha256,
            "model_profile_id": self.model_profile_id,
            "decided_by": self.decided_by,
            "decided_at": self.decided_at,
            "notes": self.notes,
            "extra": dict(self.extra),
        }

    @property
    def decision_id(self) -> str:
        return "dec-" + sha256_of(self.body())[:24]

    def record(self) -> dict[str, Any]:
        return {"decision_id": self.decision_id, **self.body()}


def decision_from_record(record: Mapping[str, Any]) -> AnalysisDecision:
    if record.get("schema") != DECISION_SCHEMA:
        raise ValueError(f"expected a {DECISION_SCHEMA} record")
    if record.get("outcome") not in OUTCOMES:
        raise ValueError(f"outcome must be one of {list(OUTCOMES)}")
    decision = AnalysisDecision(
        gate=str(record["gate"]),
        outcome=str(record["outcome"]),
        rule=dict(record["rule"]),
        evidence=dict(record["evidence"]),
        run_manifest_sha256=str(record["run_manifest_sha256"]),
        model_profile_id=str(record["model_profile_id"]),
        decided_by=str(record["decided_by"]),
        decided_at=str(record["decided_at"]),
        notes=str(record.get("notes", "")),
        extra=dict(record.get("extra", {})),
    )
    if record.get("decision_id") != decision.decision_id:
        raise ValueError("decision_id does not match the record's content")
    return decision


def gate_problems(
    record: Mapping[str, Any] | None,
    *,
    gate: str,
    model_profile_id: str,
    accept: tuple[str, ...] = ("pass",),
) -> list[str]:
    """Why ``record`` does not license a step that needs ``gate`` (empty when it does).

    ``accept`` lists the outcomes the step takes; only "pass" unless the caller
    names more, such as a recorded waiver.
    """
    if record is None:
        return [f"no {gate} decision was supplied; a passing {gate} decision is required"]
    try:
        decision = decision_from_record(record)
    except (KeyError, TypeError, ValueError) as exc:
        return [f"the {gate} decision record is invalid: {exc}"]
    problems = []
    if decision.gate != gate:
        problems.append(f"the decision is for {decision.gate}, not {gate}")
    if decision.outcome not in accept:
        wanted = " or ".join(repr(outcome) for outcome in accept)
        problems.append(f"the {gate} decision is {decision.outcome!r}, not {wanted}")
    if decision.model_profile_id != model_profile_id:
        problems.append(
            f"the {gate} decision covers {decision.model_profile_id}, "
            f"not the current profile {model_profile_id}"
        )
    if not decision.run_manifest_sha256:
        problems.append(f"the {gate} decision names no run manifest")
    return problems
