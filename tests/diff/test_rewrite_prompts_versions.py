"""The prompt drafts are versioned and rendered deterministically; the brief's text is kept verbatim."""

from __future__ import annotations

import pytest

from diff.caption_sources import SourceSet
from diff.rewrite_prompts import (
    PHASE1_REWRITE,
    PHASE1_REWRITE_V1,
    PHASE2_CELL,
    render_cell,
    render_evidence,
    render_phase1,
    render_phase2,
)

BRIEF_PHASE1 = (
    "다음 소리 자막을 다시 써 주세요. "
    "들리는 소리의 종류, 개수, 순서, 확실성의 정도를 바꾸지 말고 "
    "새로운 소리를 더하지 마세요. "
    "근거에 없는 음원을 화면 속 대상과 잇지 마세요. "
    "원래 자막처럼 끝맺고, 96자 이내의 한 문장으로 쓰세요. "
    "다시 쓴 자막만 출력하세요.\n"
    "자막: {cue_text}\n"
    "근거: {cue_evidence}"
)
BRIEF_PHASE2 = (
    "아래 근거만 써서 이 5초 구간의 소리 자막을 한 문장으로 쓰세요. "
    "음향 특성 표현은 다음 목록에서 {k}개까지만 고르세요. "
    "{qualities}. {context_rule} "
    "근거에 없는 소리는 쓰지 마세요. "
    "160자 이내로 쓰고 자막만 출력하세요.\n"
    "근거: {cue_evidence}"
)


def test_templates_are_the_brief_verbatim() -> None:
    assert PHASE1_REWRITE_V1.text == BRIEF_PHASE1 and PHASE1_REWRITE_V1.version == 1
    assert PHASE2_CELL.text == BRIEF_PHASE2


def test_rewrite_v2_asks_for_new_wording_and_keeps_both_slots() -> None:
    # Protocol v4, 2026-10-11: version 1 came back as copies of the source caption.
    assert (PHASE1_REWRITE.template_id, PHASE1_REWRITE.version) == ("rewrite-p1", 2)
    assert PHASE1_REWRITE.sha256 != PHASE1_REWRITE_V1.sha256
    assert "그대로 옮기지 말고" in PHASE1_REWRITE.text
    assert PHASE1_REWRITE.text.endswith("자막: {cue_text}\n근거: {cue_evidence}")
    assert PHASE1_REWRITE.text.count("{") == 2  # the worked example adds no slot


def test_phase1_rendering(development_sources: SourceSet) -> None:
    source = development_sources.phase(1)[0]
    prompt = render_phase1(source)
    assert prompt.text.startswith("다음 소리 자막을 같은 뜻의 새 문장으로 바꿔 쓰세요.")
    assert prompt.template_version == 2
    assert "자막: 버스가 오른쪽 차로로 지나간다." in prompt.text
    assert "소리 1 — 음원: 버스(교통·기계 소리) / 개수: 하나 / 동작: 지나간다" in prompt.text
    assert prompt.messages() == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt.text},
    ]
    assert prompt.record()["text_sha256"] == prompt.text_sha256


def test_evidence_keeps_order_and_omits_missing_fields(development_sources: SourceSet) -> None:
    two = development_sources.phase(1)[7]  # 발소리 then 문
    rendered = render_evidence(two.events)
    assert rendered.index("발소리") < rendered.index("문(")
    assert "음향 특성" not in rendered.splitlines()[0]  # the first event has no qualities


def test_cell_rendering_fills_the_quality_limit_not_the_decoding_k(development_sources: SourceSet) -> None:
    source = development_sources.phase(2)[0]
    prompt = render_phase2(source, 2, "relation")
    assert "다음 목록에서 2개까지만 고르세요. 일정한, 음이 낮은." in prompt.text
    assert "근거에 있는 음원 이름과 함께 근거에 있는 위치나 관계를 쓰세요." in prompt.text
    assert prompt.context_rules_version == 1
    with pytest.raises(ValueError, match="phase-2"):
        render_phase2(development_sources.phase(1)[0], 2, "relation")
    assert render_cell(development_sources.phase(1)[0], 0, "broad").template_id == "lattice-p2"
    with pytest.raises(ValueError, match="quality_limit"):
        render_cell(source, 5, "broad")
