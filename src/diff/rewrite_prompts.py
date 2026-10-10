"""Versioned Korean generation prompts and the evidence rendering they embed.

Both templates are the brief's development drafts, stored verbatim. A rendered
prompt carries three identities: the template's hash, the rendered text's
hash, and — added at plan freeze in the generator environment — the hash of the
token IDs after the model's own chat template (:mod:`diff.job_plan`).

The evidence rendering is deterministic and versioned: events in their
recorded order, each with source and category, count, action, qualities in
their Korean final forms, location, relation and certainty, so "종류, 개수,
순서, 확실성" in the phase-1 instruction each have a field to come from. Missing
optional fields are omitted rather than written as "없음", which a model could
copy into a caption.

Both models receive the same explicit system message, "You are a helpful
assistant." — Dream's template default; Qwen's own default adds an identity
line — so the rendered token IDs differ only by tokenizer and template.

The three ``CONTEXT_RULES`` sentences, the ``k = 0`` wording and the rendering
of an empty quality list are drafts awaiting the researcher's approval (plan
Q10); changing any of them bumps its version and therefore every job hash.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from diff.caption_sources import CATEGORIES_KO, CERTAINTY_KO, COUNTS_KO, QUALITIES, CaptionSource, SoundEvent
from diff.study_protocol import CONTEXT_STEPS, SYSTEM_MESSAGE, sha256_text


@dataclass(frozen=True)
class PromptTemplate:
    template_id: str
    version: int
    text: str

    @property
    def sha256(self) -> str:
        return sha256_text(self.text)


PHASE1_REWRITE: Final = PromptTemplate(
    template_id="rewrite-p1",
    version=1,
    text=(
        "다음 소리 자막을 다시 써 주세요. 들리는 소리의 종류, 개수, 순서, 확실성의 정도를 바꾸지 말고 "
        "새로운 소리를 더하지 마세요. 근거에 없는 음원을 화면 속 대상과 잇지 마세요. "
        "원래 자막처럼 끝맺고, 96자 이내의 한 문장으로 쓰세요. 다시 쓴 자막만 출력하세요.\n"
        "자막: {cue_text}\n"
        "근거: {cue_evidence}"
    ),
)
PHASE2_CELL: Final = PromptTemplate(
    template_id="lattice-p2",
    version=1,
    text=(
        "아래 근거만 써서 이 5초 구간의 소리 자막을 한 문장으로 쓰세요. "
        "음향 특성 표현은 다음 목록에서 {k}개까지만 고르세요. {qualities}. {context_rule} "
        "근거에 없는 소리는 쓰지 마세요. 160자 이내로 쓰고 자막만 출력하세요.\n"
        "근거: {cue_evidence}"
    ),
)
EVIDENCE_RENDERER_VERSION: Final = 1
CONTEXT_RULES_VERSION: Final = 1
CONTEXT_RULES: Final[dict[str, str]] = {
    "broad": (
        "음원은 사람 소리, 동물 소리, 교통·기계 소리, 음악 소리, 자연 소리 같은 넓은 범주로만 쓰고 "
        "구체적인 음원 이름이나 위치는 쓰지 마세요."
    ),
    "specific": "근거에 있는 음원 이름을 쓰되 위치나 다른 대상과의 관계는 쓰지 마세요.",
    "relation": "근거에 있는 음원 이름과 함께 근거에 있는 위치나 관계를 쓰세요.",
}
EMPTY_QUALITIES: Final = "없음"


@dataclass(frozen=True)
class RenderedPrompt:
    """A rendered user turn plus the fixed system message both models receive."""

    template_id: str
    template_version: int
    template_sha256: str
    evidence_renderer_version: int
    context_rules_version: int | None
    text: str
    system_message: str = SYSTEM_MESSAGE

    @property
    def text_sha256(self) -> str:
        return sha256_text(self.text)

    def messages(self) -> list[dict[str, str]]:
        """The chat turns to render with each model's own template."""
        return [{"role": "system", "content": self.system_message}, {"role": "user", "content": self.text}]

    def record(self) -> dict[str, object]:
        return {
            "template_id": self.template_id,
            "template_version": self.template_version,
            "template_sha256": self.template_sha256,
            "evidence_renderer_version": self.evidence_renderer_version,
            "context_rules_version": self.context_rules_version,
            "system_message": self.system_message,
            "text": self.text,
            "text_sha256": self.text_sha256,
        }


def render_event(number: int, event: SoundEvent) -> str:
    parts = [f"음원: {event.source}({CATEGORIES_KO[event.category]})", f"개수: {COUNTS_KO[event.count]}"]
    if event.action:
        parts.append(f"동작: {event.action}")
    if event.qualities:
        parts.append("음향 특성: " + ", ".join(QUALITIES[key][0] for key in event.qualities))
    if event.location:
        parts.append(f"위치: {event.location}")
    if event.relation:
        parts.append(f"관계: {event.relation}")
    parts.append(f"확실성: {CERTAINTY_KO[event.certainty]}")
    return f"소리 {number} — " + " / ".join(parts)


def render_evidence(events: Sequence[SoundEvent]) -> str:
    ordered = sorted(events, key=lambda event: event.order)
    if not ordered:
        raise ValueError("A prompt needs at least one evidence event")
    return "\n".join(render_event(number, event) for number, event in enumerate(ordered, 1))


def render_qualities(source: CaptionSource) -> str:
    keys = source.evidence_terms()["qualities"]
    return ", ".join(QUALITIES[key][0] for key in keys) if keys else EMPTY_QUALITIES


def render_phase1(source: CaptionSource) -> RenderedPrompt:
    if source.phase != 1:
        raise ValueError("The rewrite prompt is for phase-1 sources")
    text = PHASE1_REWRITE.text.format(cue_text=source.text, cue_evidence=render_evidence(source.events))
    return RenderedPrompt(
        PHASE1_REWRITE.template_id,
        PHASE1_REWRITE.version,
        PHASE1_REWRITE.sha256,
        EVIDENCE_RENDERER_VERSION,
        None,
        text,
    )


def render_cell(source: CaptionSource, quality_limit: int, context: str) -> RenderedPrompt:
    """The phase-2 cell template over any source's evidence.

    G0 uses it on development-clip cues (phase-2 videos are participant
    material); :func:`render_phase2` is the lattice's own entry point.
    ``quality_limit`` fills the template's ``{k}``: how many closed-vocabulary
    qualities the caption may use. It is not the decoding factor k.
    """
    if context not in CONTEXT_STEPS:
        raise ValueError(f"Unknown context step: {context}")
    if type(quality_limit) is not int or not 0 <= quality_limit <= 4:
        raise ValueError("quality_limit must be an integer from 0 to 4")
    text = PHASE2_CELL.text.format(
        k=quality_limit,
        qualities=render_qualities(source),
        context_rule=CONTEXT_RULES[context],
        cue_evidence=render_evidence(source.events),
    )
    return RenderedPrompt(
        PHASE2_CELL.template_id,
        PHASE2_CELL.version,
        PHASE2_CELL.sha256,
        EVIDENCE_RENDERER_VERSION,
        CONTEXT_RULES_VERSION,
        text,
    )


def render_phase2(source: CaptionSource, quality_limit: int, context: str) -> RenderedPrompt:
    if source.phase != 2:
        raise ValueError("The lattice prompt is for phase-2 sources")
    return render_cell(source, quality_limit, context)
