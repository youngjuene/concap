"""Generation jobs and plans: one cue × one condition, or one cue × one cell.

A job's identity is the SHA-256 of everything that determines its output: the
source content, the rendered prompt and its versions, the model profile, the
explicit call arguments and the caps. It never includes timestamps, paths or
the whole protocol hash, so re-running a plan reuses terminal jobs and an
unrelated protocol edit (a gate threshold, say) renames nothing.

``original`` is the control: the source track repeated exactly, so it has no
job. Every ``dream_k*`` condition is Dream at parallelism k — a fixed canvas
with ``steps = canvas ÷ k`` — and ``qwen`` is the autoregressive baseline with
the same canvas as its ``max_new_tokens``.

Freezing happens in the generator environment because it needs each model's
own tokenizer and chat template. It adds each prompt's token-ID hash and length
and applies the prompt guards: prompt plus canvas within the 2,048-token
context, no mask token in a Dream prompt (the mask search covers the prompt, so
that position would be generated), and no BOS anywhere.

Phase-2 lattice plans additionally need a passing G0 decision for the current
Dream profile (AC-20, :mod:`diff.analysis_decisions`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from diff.analysis_decisions import gate_problems
from diff.caption_sources import CaptionSource, bundle_problems
from diff.model_profile import DREAM_7B, QWEN_7B, ModelProfile, profile
from diff.output_checks import CHECKS_VERSION
from diff.rewrite_prompts import RenderedPrompt, render_live, render_phase1, render_phase2
from diff.study_protocol import (
    ALG,
    ALG_TEMP,
    BOS_IDS,
    CONTROL_CONDITION,
    ENDOFTEXT_ID,
    EPS,
    IM_END_ID,
    K_CONDITIONS,
    MASK_ID,
    PHASE1_CONDITIONS,
    PROTOCOL_SHA256,
    QWEN_CONDITION,
    TEMPERATURE,
    TIMEOUT_SECONDS,
    TOP_P,
    canvas_tokens,
    lattice_cells,
    sha256_of,
    sha256_tokens,
    steps_for,
)

JOB_SCHEMA: Final = "diff.job/v2"
PLAN_SCHEMA: Final = "diff.job-plan/v2"
FROZEN_SCHEMA: Final = "diff.frozen-plan/v2"
PURPOSES: Final = ("development", "participant", "g0", "smoke")
MATERIAL_FOR: Final[dict[str, tuple[str, ...]]] = {
    "development": ("development",),
    "participant": ("participant",),
    "g0": ("development",),
    "smoke": ("synthetic",),
}


class PlanError(ValueError):
    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def dream_call_settings(phase: int, k: int) -> dict[str, Any]:
    """Every keyword the Dream call contract requires, except the per-call attention mask."""
    canvas = canvas_tokens(phase)
    return {
        "max_new_tokens": canvas,
        "steps": steps_for(phase, k),
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": None,
        "alg": ALG,
        "alg_temp": ALG_TEMP,
        "eps": EPS,
        "mask_token_id": MASK_ID,
        "output_history": True,
        "return_dict_in_generate": True,
    }


def qwen_call_settings(phase: int) -> dict[str, Any]:
    """Greedy, with the shipped repetition penalty neutralised; temperature/top_p/top_k stay unset."""
    return {
        "max_new_tokens": canvas_tokens(phase),
        "do_sample": False,
        "repetition_penalty": 1.0,
        "eos_token_id": [IM_END_ID, ENDOFTEXT_ID],
    }


def condition_settings(phase: int, condition: str) -> tuple[ModelProfile, dict[str, Any]]:
    if condition == QWEN_CONDITION:
        return QWEN_7B, qwen_call_settings(phase)
    if condition in K_CONDITIONS:
        return DREAM_7B, dream_call_settings(phase, K_CONDITIONS[condition])
    raise PlanError([f"no generation for condition {condition!r}"])


@dataclass(frozen=True)
class Job:
    phase: int
    purpose: str
    source_id: str
    source_sha256: str
    unit_key: str
    cue_index: int
    condition: str
    cell: Mapping[str, Any] | None
    profile_id: str
    runner: str
    prompt: RenderedPrompt
    generate: Mapping[str, Any]

    @property
    def caps(self) -> dict[str, Any]:
        caps: dict[str, Any] = {
            "canvas_tokens": self.generate["max_new_tokens"],
            "timeout_s": TIMEOUT_SECONDS,
        }
        if self.runner == "dream":
            caps.update(steps=self.generate["steps"], expected_forward_calls=self.generate["steps"])
        return caps

    @property
    def checks_profile(self) -> str:
        return f"p{self.phase}-v{CHECKS_VERSION}"

    def spec(self) -> dict[str, Any]:
        return {
            "schema": JOB_SCHEMA,
            "phase": self.phase,
            "purpose": self.purpose,
            "source_id": self.source_id,
            "source_sha256": self.source_sha256,
            "unit_key": self.unit_key,
            "cue_index": self.cue_index,
            "condition": self.condition,
            "cell": dict(self.cell) if self.cell is not None else None,
            "model": {"profile_id": self.profile_id, "runner": self.runner},
            "prompt": self.prompt.record(),
            "generate": dict(self.generate),
            "caps": self.caps,
            "checks_profile": self.checks_profile,
        }

    @property
    def job_id(self) -> str:
        return "job-" + sha256_of(self.spec())[:32]

    @property
    def seed(self) -> int | None:
        """Dream's sampling seed: derived from the job's identity, so a retry is identical (v4)."""
        return int(self.job_id[4:12], 16) if self.runner == "dream" else None

    def record(self) -> dict[str, Any]:
        return {"job_id": self.job_id, **self.spec()}


def job_from_record(record: Mapping[str, Any]) -> Job:
    prompt = record["prompt"]
    job = Job(
        phase=int(record["phase"]),
        purpose=str(record["purpose"]),
        source_id=str(record["source_id"]),
        source_sha256=str(record["source_sha256"]),
        unit_key=str(record["unit_key"]),
        cue_index=int(record["cue_index"]),
        condition=str(record["condition"]),
        cell=record["cell"],
        profile_id=str(record["model"]["profile_id"]),
        runner=str(record["model"]["runner"]),
        prompt=RenderedPrompt(
            template_id=str(prompt["template_id"]),
            template_version=int(prompt["template_version"]),
            template_sha256=str(prompt["template_sha256"]),
            evidence_renderer_version=int(prompt["evidence_renderer_version"]),
            context_rules_version=prompt["context_rules_version"],
            text=str(prompt["text"]),
            system_message=str(prompt["system_message"]),
        ),
        generate=dict(record["generate"]),
    )
    if record.get("job_id") != job.job_id or prompt.get("text_sha256") != job.prompt.text_sha256:
        raise PlanError([f"{record.get('job_id')}: recorded identity does not match its content"])
    return job


@dataclass(frozen=True)
class Plan:
    phase: int
    purpose: str
    jobs: tuple[Job, ...]

    @property
    def plan_id(self) -> str:
        return "plan-" + sha256_of(sorted(job.job_id for job in self.jobs))[:32]

    def record(self) -> dict[str, Any]:
        return {
            "schema": PLAN_SCHEMA,
            "plan_id": self.plan_id,
            "phase": self.phase,
            "purpose": self.purpose,
            "protocol_sha256": PROTOCOL_SHA256,
            "jobs": [job.record() for job in self.jobs],
        }


def plan_from_record(record: Mapping[str, Any]) -> Plan:
    if record.get("schema") != PLAN_SCHEMA:
        raise PlanError([f"expected {PLAN_SCHEMA}"])
    plan = Plan(
        phase=int(record["phase"]),
        purpose=str(record["purpose"]),
        jobs=tuple(job_from_record(job) for job in record["jobs"]),
    )
    if plan.plan_id != record.get("plan_id"):
        raise PlanError(["plan_id does not match its jobs"])
    return plan


def plannable(sources: Iterable[CaptionSource], phase: int, purpose: str) -> list[CaptionSource]:
    """The sources a plan of this purpose may read, or every reason it may not."""
    if purpose not in PURPOSES:
        raise PlanError([f"purpose must be one of {list(PURPOSES)}"])
    chosen = sorted((s for s in sources if s.phase == phase), key=lambda s: (s.unit_key, s.cue_index))
    problems = []
    if not chosen:
        problems.append(f"no phase-{phase} sources")
    for source in chosen:
        where = f"{source.source_id} ({source.unit_key} cue {source.cue_index})"
        if source.material not in MATERIAL_FOR[purpose]:
            problems.append(f"{where}: {source.material} material cannot serve a {purpose} plan")
        if source.status == "rejected":
            problems.append(f"{where}: rejected sources are never generated from")
        if not source.events:
            problems.append(f"{where}: no evidence events")
        if purpose == "participant":
            problems.extend(bundle_problems(source))
    if problems:
        raise PlanError(problems)
    return chosen


def build_phase1_plan(sources: Iterable[CaptionSource], conditions: Sequence[str], purpose: str) -> Plan:
    unknown = [condition for condition in conditions if condition not in PHASE1_CONDITIONS]
    if unknown or len(set(conditions)) != len(conditions):
        raise PlanError(
            [f"conditions must be distinct values from {list(PHASE1_CONDITIONS)}; got {conditions}"]
        )
    jobs = []
    for source in plannable(sources, 1, purpose):
        prompt = render_phase1(source)
        for condition in conditions:
            if condition == CONTROL_CONDITION:
                continue
            model, generate = condition_settings(1, condition)
            jobs.append(make_job(source, purpose, condition, None, model, prompt, generate))
    return Plan(phase=1, purpose=purpose, jobs=tuple(jobs))


def build_phase2_plan(
    sources: Iterable[CaptionSource],
    condition: str,
    lattice: str,
    purpose: str,
    g0_decision: Mapping[str, Any] | None,
) -> Plan:
    """The lattice generator: one Dream decoding condition for the whole phase (AC-20 gated)."""
    refusals = gate_problems(g0_decision, gate="G0", model_profile_id=DREAM_7B.profile_id)
    if refusals:
        raise PlanError(["lattice generation refused (AC-20): " + problem for problem in refusals])
    if condition not in K_CONDITIONS:
        raise PlanError([f"phase 2 is generated by one Dream condition from {list(K_CONDITIONS)}"])
    cells = lattice_cells(lattice)
    jobs = []
    for source in plannable(sources, 2, purpose):
        for cell in cells:
            prompt = render_phase2(source, int(cell["quality_limit"]), str(cell["context"]))
            model, generate = condition_settings(2, condition)
            jobs.append(make_job(source, purpose, condition, cell, model, prompt, generate))
    return Plan(phase=2, purpose=purpose, jobs=tuple(jobs))


def make_job(
    source: CaptionSource,
    purpose: str,
    condition: str,
    cell: Mapping[str, Any] | None,
    model: ModelProfile,
    prompt: RenderedPrompt,
    generate: Mapping[str, Any],
    *,
    phase: int | None = None,
) -> Job:
    return Job(
        phase=source.phase if phase is None else phase,
        purpose=purpose,
        source_id=source.source_id,
        source_sha256=source.source_sha256,
        unit_key=source.unit_key,
        cue_index=source.cue_index,
        condition=condition,
        cell=dict(cell) if cell is not None else None,
        profile_id=model.profile_id,
        runner=model.runner,
        prompt=prompt,
        generate=dict(generate),
    )


LIVE_CONDITION: Final = "dream_k2"  # researcher, 2026-10-11: about 8 s per caption at canvas 128


def live_job(source: CaptionSource, texture: int, context: str, guidance: str, purpose: str) -> Job:
    """One live phase-2 job: the cue's starting caption under the snapped steps and the guidance.

    The guidance is in the prompt, so it is part of the job's identity and its seed.
    """
    if purpose not in ("participant", "smoke"):
        raise PlanError([f"a live job serves a participant session or a smoke run, not {purpose!r}"])
    model, generate = condition_settings(2, LIVE_CONDITION)
    cell = {"cell_id": f"t{texture}-{context}", "quality_limit": texture, "context": context}
    return make_job(
        source,
        purpose,
        LIVE_CONDITION,
        cell,
        model,
        render_live(source, texture, context, guidance),
        generate,
        phase=2,
    )


def prompt_guard_problems(job: Job, token_ids: Sequence[int]) -> list[str]:
    """Every refusal a rendered prompt earns before any model call."""
    model = profile(job.profile_id)
    canvas = int(job.generate["max_new_tokens"])
    problems = []
    if len(token_ids) + canvas > model.context_limit:
        problems.append(
            f"{job.job_id}: {len(token_ids)} prompt tokens + a {canvas}-token canvas exceed the "
            f"{model.context_limit}-token context"
        )
    if job.runner == "dream" and MASK_ID in token_ids:
        problems.append(
            f"{job.job_id}: the prompt contains the mask token {MASK_ID}, which would be generated into"
        )
    present = sorted({token for token in token_ids if token in BOS_IDS})
    if present:
        problems.append(f"{job.job_id}: the rendered prompt contains BOS token(s) {present}")
    return problems


@dataclass(frozen=True)
class FrozenJob:
    job: Job
    token_ids_sha256: str
    token_count: int

    def record(self) -> dict[str, Any]:
        return {
            **self.job.record(),
            "token_ids_sha256": self.token_ids_sha256,
            "token_count": self.token_count,
        }


@dataclass(frozen=True)
class FrozenPlan:
    plan: Plan
    jobs: tuple[FrozenJob, ...]
    tokenizers: Mapping[str, Mapping[str, Any]]

    def body(self) -> dict[str, Any]:
        return {
            "schema": FROZEN_SCHEMA,
            "plan_id": self.plan.plan_id,
            "phase": self.plan.phase,
            "purpose": self.plan.purpose,
            "protocol_sha256": PROTOCOL_SHA256,
            "tokenizers": {key: dict(value) for key, value in sorted(self.tokenizers.items())},
            "jobs": [job.record() for job in self.jobs],
        }

    @property
    def frozen_sha256(self) -> str:
        return sha256_of(self.body())

    def record(self) -> dict[str, Any]:
        return {**self.body(), "frozen_sha256": self.frozen_sha256}

    def by_id(self) -> dict[str, FrozenJob]:
        return {frozen.job.job_id: frozen for frozen in self.jobs}


def freeze_plan(
    plan: Plan,
    tokenize: Callable[[Job], Sequence[int]],
    tokenizers: Mapping[str, Mapping[str, Any]],
) -> FrozenPlan:
    """Add token-ID hashes and lengths; refuse the plan if any prompt fails a guard."""
    missing = sorted({job.profile_id for job in plan.jobs} - set(tokenizers))
    if missing:
        raise PlanError([f"no tokenizer record for {missing}"])
    frozen, problems = [], []
    for job in plan.jobs:
        ids = [int(token) for token in tokenize(job)]
        problems.extend(prompt_guard_problems(job, ids))
        frozen.append(FrozenJob(job, sha256_tokens(ids), len(ids)))
    if problems:
        raise PlanError(problems)
    return FrozenPlan(plan, tuple(frozen), dict(tokenizers))


def frozen_plan_from_record(record: Mapping[str, Any]) -> FrozenPlan:
    if record.get("schema") != FROZEN_SCHEMA:
        raise PlanError([f"expected {FROZEN_SCHEMA}"])
    jobs = tuple(
        FrozenJob(job_from_record(entry), str(entry["token_ids_sha256"]), int(entry["token_count"]))
        for entry in record["jobs"]
    )
    plan = Plan(phase=int(record["phase"]), purpose=str(record["purpose"]), jobs=tuple(f.job for f in jobs))
    frozen = FrozenPlan(plan, jobs, dict(record["tokenizers"]))
    if plan.plan_id != record.get("plan_id") or frozen.frozen_sha256 != record.get("frozen_sha256"):
        raise PlanError(["frozen plan identity does not match its content"])
    return frozen


def source_token_report(sources: Iterable[CaptionSource], count: Callable[[str], int]) -> dict[str, Any]:
    """CV-08: measured source token counts against each phase's canvas."""
    report: dict[str, Any] = {}
    chosen = list(sources)
    for phase in (1, 2):
        counts = sorted(count(source.text) for source in chosen if source.phase == phase)
        if not counts:
            continue
        canvas = canvas_tokens(phase)
        report[f"phase{phase}"] = {
            "sources": len(counts),
            "max_tokens": counts[-1],
            "median_tokens": counts[len(counts) // 2],
            "canvas_tokens": canvas,
            "over_canvas": sum(1 for value in counts if value > canvas),
        }
    return report
