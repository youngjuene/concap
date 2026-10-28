"""Versioned Korean generation prompts and the evidence rendering they embed.

``PHASE2_CELL`` and ``PHASE1_REWRITE_V1`` are the brief's development drafts,
stored verbatim. ``PHASE1_REWRITE`` is version 3 (2026-10-11), and every
version after the brief's was chosen the same way: drafted by the assistant at
the researcher's request, compared at Dream's settings on held-out captions
only (three c2 clips excluded from G0 and the eight synthetic test cues), and
picked by a rule written down before the run.
- Version 2 targeted copying of the source caption, and won on G0's script
  check.
- Version 3 targets the prompt labels version 2 echoed (10 of 18 G0 outputs).
  It quotes the caption in 「」, has no label lines, and ends by forbidding any
  prefix. It won on the fewest outputs with a failure or a prompt leak.
- The worked examples are invented (a distant dog), so they name no study clip.

``LIVE_CELL_PROMPTS`` (``live-p2``) are the live phase-2 prompts, one per
slider cell, chosen by the same procedure on the phase-2 captions. A rendered
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
assistant.", because each chat template's own default adds an identity line
("You are Trillion…", Qwen's) — so the rendered token IDs differ only by
tokenizer and template. Live phase-2 jobs carry the session's tuner system
message instead.

The three ``CONTEXT_RULES`` sentences, the ``k = 0`` wording and the rendering
of an empty quality list are drafts awaiting the researcher's approval (plan
Q10); changing any of them bumps its version and therefore every job hash.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from diff.caption_sources import CATEGORIES_KO, CERTAINTY_KO, COUNTS_KO, QUALITIES, CaptionSource, SoundEvent
from diff.study_protocol import CONTEXT_STEPS, LIVE_TEXTURE_MAX, SYSTEM_MESSAGE, sha256_text


@dataclass(frozen=True)
class PromptTemplate:
    template_id: str
    version: int
    text: str

    @property
    def sha256(self) -> str:
        return sha256_text(self.text)


PHASE1_REWRITE_V1: Final = PromptTemplate(
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
PHASE1_REWRITE: Final = PromptTemplate(
    template_id="rewrite-p1",
    version=3,
    text=(
        "다음은 영상에서 들리는 소리를 설명한 자막입니다: 「{cue_text}」\n"
        "근거로 확인된 소리는 다음과 같습니다.\n"
        "{cue_evidence}\n"
        "예를 들어 「멀리서 개 한 마리가 짖는 소리」는 「먼 곳에서 들려오는 개 한 마리의 짖는 소리」처럼 "
        "바꿀 수 있습니다.\n"
        "이 자막을 같은 뜻의 새 문장으로 바꿔 쓰세요. 원래 문장을 그대로 옮기지 말고 단어와 어순을 바꾸세요. "
        "들리는 소리의 종류, 개수, 순서, 확실성의 정도는 그대로 두고 새로운 소리를 더하지 마세요. "
        "근거에 없는 음원을 화면 속 대상과 잇지 마세요. 끝맺는 방식은 원래 자막과 같게 하고, "
        "96자 이내의 한 문장으로 쓰세요. 바꿔 쓴 문장만 출력하고, "
        "앞에 '자막' 같은 말이나 따옴표, 번호, 설명을 붙이지 마세요."
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

# Live phase 2 (researcher, 2026-10-11): the model rewrites a cue's starting caption
# while the participant watches, under the session's tuner system message.
# Version 4 (2026-10-21): one prompt per slider cell, nine in all (researcher: "9 different
# prompts for rewriting the original caption to have noticeable semantical adjustments", then
# "polish the prompts to be kind enough to deliver as soundscape descriptors"). Each cell writes
# a short description of the sound scene. Context: broad groups the sounds into general kinds but
# keeps what they do, specific names each source and what it does, relation places the sounds in
# space around the first. Texture: how they sound, on no sound (0), the first sound (1), every
# sound (2). The cue comes last, after worked examples that name no study clip: in the bench of
# 2026-10-21 (artifacts/diff/cell-prompts, rule fixed before the runs) an example after the cue
# was copied into unrelated cues, and no example at all left 56% of the captions unchanged.
# Version 3's single prompt with a texture rule and a context rule changed too little (38 of 54
# rewrites were copies). Until the researcher approves the nine texts, participant sessions are
# refused.
# Version 5 (2026-10-27), for Claude, which writes the live captions since that day: each stop
# delivers its approved label ("none / a little / a lot", "broad kinds / specific sources / places
# and relations"). Texture 0 also forbids sound-imitating words (웅웅, 웅성, 끼익) taken from the cue,
# texture 1 allows one such word in the whole caption, and broad names no source at all (Claude kept
# 경적, 종소리 or 엔진 in a third of version 4's broad captions); the market example's broad rewrites
# drop the cart. Chosen by a rule written before the runs (artifacts/diff/cell-prompts/RULE.md, the
# Claude round): it passed the v4 rule on 30 dev cues and on 30 held-out ones.
# Version 6 (2026-10-27, the researcher: fix the broad captions that name one kind twice): broad names
# each kind once and merges what the kind's sounds do ("교통 소리가 이어지다가 잠깐씩 울린다"), not
# "또 다른 교통 소리"; version 5 did that in 12-22% of broad captions, version 6 in 0 of 90 on dev and
# on held-out (RULE.md, Claude round 2). The other six cells are version 5's text.
LIVE_PROMPTS_PROVISIONAL: Final = True
LIVE_RULES_VERSION: Final = 5
_LIVE_HEAD: Final = (
    "영상 속 소리를 시청자가 편하게 읽을 수 있는 소리 풍경 묘사로 바꿔 쓰는 일입니다. "
    "원래 자막에는 가장 두드러진 소리가 먼저 나옵니다.\n"
    "지킬 것\n"
    "- 원래 자막에 있는 소리만 쓰고, 없는 소리는 더하지 마세요. 소리의 순서는 그대로 두세요.\n"
    "- 짧고 자연스러운 우리말로, 귀에 들리는 장면을 그리듯 쓰세요.\n"
)
_LIVE_TEXTURE: Final[dict[int, str]] = {
    0: (
        "- 소리가 어떻게 들리는지 나타내는 꾸밈말과 흉내말(크게, 낮게, 날카롭게, 웅웅, 웅성, 끼익 같은 말)은 "
        "원래 자막에 있더라도 쓰지 말고, 무엇이 무엇을 하는지만 담담하게 쓰세요"
        "(예: 웅성거리는 사람들 → 이야기하는 사람들).\n"
    ),
    1: (
        "- 소리가 어떻게 들리는지 나타내는 꾸밈말이나 흉내말은 자막 전체에서 딱 하나만, "
        "가장 두드러진 첫 소리에만 쓰세요(예: 낮게, 날카롭게, 묵직하게, 희미하게, 일정하게). "
        "다른 소리에는 원래 자막에 있던 꾸밈말과 흉내말(웅웅, 웅성, 끼익 같은 말)까지 모두 빼세요.\n"
    ),
    2: (
        "- 모든 소리에 어떻게 들리는지 나타내는 꾸밈말을 붙여 질감이 느껴지게 쓰세요. 첫 소리에는 두 개"
        "(예: 낮고 묵직하게), 나머지 소리에는 하나씩 붙이세요(예: 날카롭게, 희미하게, 일정하게, 부드럽게).\n"
    ),
}
_LIVE_CONTEXT: Final[dict[str, str]] = {
    "broad": (
        "- 음원 이름은 하나도 쓰지 마세요. 경적, 엔진, 타이어, 오토바이, 종소리, 목소리, 발소리처럼 "
        "무엇이 내는 소리인지 알려 주는 이름도 쓰지 말고, 교통 소리, 사람들 소리, 음악 소리, "
        "자연의 소리, 동물 소리 같은 큰 갈래로만 부르세요. 갈래마다 이름은 한 번만 쓰세요. "
        "같은 갈래의 소리가 여럿이면 하나로 묶어 그 갈래가 하는 일을 이어 쓰고"
        "(예: 교통 소리가 이어지다가 잠깐씩 울린다), '또 다른 교통 소리'처럼 같은 갈래를 다시 부르지 마세요. "
        "무엇을 하는지는 그 갈래에 맞는 쉬운 말(오간다, 이어진다, 이야기한다, 흐른다, 분다)로만 쓰세요. "
        "위치는 쓰지 마세요.\n"
    ),
    "specific": (
        "- 음원은 원래 자막의 구체적인 이름으로 부르고, 각 소리가 무엇을 하는지 살려 쉼표로 이으세요. "
        "위치나 다른 소리와의 관계는 쓰지 마세요.\n"
    ),
    "relation": (
        "- 음원은 원래 자막의 구체적인 이름으로 부르고, 첫 소리를 앞에 두고 나머지 소리를 그 사이나 뒤, "
        "배경에 놓아 공간이 그려지는 한 문장으로 쓰세요('사이로', '너머로', '뒤로', '배경에' 같은 말). "
        "원래 자막의 위치(멀리서, 근처에서)는 살리세요.\n"
    ),
}
# Each example: an invented caption and its rewrite in every cell, (texture, context) -> text.
_LIVE_EXAMPLES: Final[tuple[tuple[str, dict[tuple[int, str], str]], ...]] = (
    (
        "공원에서 크게 웃는 아이들 소리, 어른들의 대화 소리, 멀리서 개가 짖는 소리",
        {
            (0, "broad"): "사람들이 웃고 이야기하는 소리와 동물이 우는 소리",
            (1, "broad"): "사람들이 크게 웃고 이야기하는 소리와 동물이 우는 소리",
            (2, "broad"): "사람들이 높고 크게 웃으며 이야기하는 소리와 동물이 날카롭게 우는 소리",
            (0, "specific"): "아이들이 웃는 소리, 어른들이 이야기하는 소리, 개가 짖는 소리",
            (1, "specific"): "아이들이 크게 웃는 소리, 어른들이 이야기하는 소리, 개가 짖는 소리",
            (2, "specific"): (
                "아이들이 높고 크게 웃는 소리, 어른들이 낮게 이야기하는 소리, 개가 날카롭게 짖는 소리"
            ),
            (0, "relation"): "공원의 아이들 웃음소리 사이로 어른들의 이야기가 들리고, 멀리서 개가 짖는다",
            (1, "relation"): (
                "공원에 아이들 웃음소리가 크게 퍼지는 사이로 어른들의 이야기가 들리고, 멀리서 개가 짖는다"
            ),
            (2, "relation"): (
                "공원에 높고 크게 퍼지는 웃음소리 사이로 낮은 이야기 소리가 들리고, "
                "멀리서 개가 날카롭게 짖는다"
            ),
        },
    ),
    (
        "숲속에서 계속 지저귀는 새소리, 바람에 흔들리는 나뭇잎 소리",
        {
            (0, "broad"): "동물들이 우는 소리와 바람이 스치는 자연의 소리",
            (1, "broad"): "동물들이 일정하게 우는 소리와 바람이 스치는 자연의 소리",
            (2, "broad"): "동물들이 높고 일정하게 우는 소리와 바람이 부드럽게 스치는 자연의 소리",
            (0, "specific"): "새들이 지저귀는 소리, 나뭇잎이 바람에 흔들리는 소리",
            (1, "specific"): "새들이 일정하게 지저귀는 소리, 나뭇잎이 바람에 흔들리는 소리",
            (2, "specific"): "새들이 높고 일정하게 지저귀는 소리, 나뭇잎이 부드럽게 흔들리는 소리",
            (0, "relation"): "숲속에 새들이 지저귀는 소리 뒤로 나뭇잎이 바람에 흔들린다",
            (1, "relation"): "숲속에 새들이 일정하게 지저귀는 소리 뒤로 나뭇잎이 바람에 흔들린다",
            (2, "relation"): "숲속에 새들이 높고 일정하게 지저귀는 소리 뒤로 나뭇잎이 부드럽게 흔들린다",
        },
    ),
    (
        "시장에서 상인들이 외치는 소리, 손수레 바퀴가 덜컹거리는 소리",
        {
            (0, "broad"): "사람들이 외치는 소리와 기계가 오가는 소리",
            (1, "broad"): "사람들이 크게 외치는 소리와 기계가 오가는 소리",
            (2, "broad"): "사람들이 크고 높게 외치는 소리와 기계가 거칠게 오가는 소리",
            (0, "specific"): "상인들이 외치는 소리, 손수레 바퀴가 구르는 소리",
            (1, "specific"): "상인들이 크게 외치는 소리, 손수레 바퀴가 구르는 소리",
            (2, "specific"): "상인들이 크고 높게 외치는 소리, 손수레 바퀴가 거칠게 구르는 소리",
            (0, "relation"): "시장의 상인들 외침 사이로 손수레 바퀴가 지나간다",
            (1, "relation"): "시장에 상인들 외침이 크게 퍼지는 사이로 손수레 바퀴가 지나간다",
            (2, "relation"): "시장에 크고 높게 퍼지는 외침 사이로 손수레 바퀴가 거칠게 지나간다",
        },
    ),
)
_LIVE_TAIL: Final = (
    "이제 아래 자막을 바꿔 쓰세요. 바꿔 쓴 자막만 한 줄로, 70자 이내로 출력하세요.\n원래 자막: {cue_text}"
)


def _live_cell_text(texture: int, context: str) -> str:
    examples = "".join(
        f"예시\n원래 자막: {caption}\n바꿔 쓴 자막: {rewrites[(texture, context)]}\n"
        for caption, rewrites in _LIVE_EXAMPLES
    )
    return _LIVE_HEAD + _LIVE_TEXTURE[texture] + _LIVE_CONTEXT[context] + examples + _LIVE_TAIL


LIVE_CELL_PROMPTS: Final[dict[tuple[int, str], PromptTemplate]] = {
    (texture, context): PromptTemplate(
        template_id="live-p2", version=6, text=_live_cell_text(texture, context)
    )
    for texture in range(LIVE_TEXTURE_MAX + 1)
    for context in CONTEXT_STEPS
}

# Phase 2's scene-aware cells (researcher, 2026-10-28, gate 5: the participant's phase-1 scene
# perception carried into the live captions, ``diff.phase2_scene``). A cue the model recommends
# scene-aware for this participant is written by these nine; an acousmatic one by the nine above,
# unchanged, which take nothing from the picture. Each scene cell is its acousmatic cell with the
# context rule placing the sounds in the scene on screen: the setting is the cue window's distinctive
# element, a sound whose source the masks put in view is tied to it, and the others are heard from
# "보이지 않는 곳" (not "화면 밖", as in phase 1's off-screen sentence); no unseen sound is set beside a
# seen thing. The examples' scenes are invented too. Provisional until the researcher approves them
# with the scene parts (``phase2_scene.PHASE2_SCENE_PROVISIONAL``).
_LIVE_SCENE_RULE: Final = (
    "- 이 영상은 화면과 함께 봅니다. 아래에 적힌 장면에 "
    "맞춰, 소리가 화면 속 장면에서 나는 것으로 들리게 쓰세요. "
    "화면에 음원이 보이지 않는 소리는 '보이지 않는 곳에서' "
    "들리는 것으로 쓰고, 화면 속 대상 옆이나 사이에 두지 "
    "마세요. 장면에 적히지 않은 대상이나 장소는 지어내지 마세요.\n"
)
_LIVE_SCENE_CONTEXT: Final[dict[str, str]] = {
    "broad": _LIVE_CONTEXT["broad"].replace(
        "위치는 쓰지 마세요.", "소리가 들리는 곳은 장면의 배경으로 한 번만 쓰세요."
    ),
    "specific": (
        "- 음원은 원래 자막의 구체적인 이름으로 부르고, 각 소리가 무엇을 하는지 살려 쉼표로 이으세요. "
        "소리가 들리는 곳은 장면의 배경으로 한 번만 쓰고, 다른 소리와의 관계는 쓰지 마세요.\n"
    ),
    "relation": (
        "- 음원은 원래 자막의 구체적인 이름으로 부르고, 첫 "
        "소리를 앞에 두고 나머지 소리를 그 곁이나 사이에 놓아 "
        "공간이 그려지는 한 문장으로 쓰세요. 공간은 장면의 배경과 화면에 음원이 보이는 소리로만 그리세요. "
        "원래 자막의 위치(멀리서, 근처에서)는 살리세요.\n"
    ),
}
_SCENE_LINES: Final = (
    "장면의 배경: {seen}\n화면에 음원이 보이는 소리: {visible_sounds}\n"
    "화면에 음원이 보이지 않는 소리: {unseen_sounds}\n"
)
# Each example: an invented caption, its invented scene, and its scene-aware rewrite in every cell.
_LIVE_SCENE_EXAMPLES: Final[tuple[tuple[str, dict[str, str], dict[tuple[int, str], str]], ...]] = (
    (
        _LIVE_EXAMPLES[0][0],
        {
            "seen": "나무와 풀",
            "visible_sounds": "크게 웃는 아이들 소리, 어른들의 대화 소리",
            "unseen_sounds": "멀리서 개가 짖는 소리",
        },
        {
            (
                0,
                "broad",
            ): "나무와 풀 사이에서 사람들이 웃고 이야기하는 소리, 보이지 않는 곳에서 동물이 우는 소리",
            (
                1,
                "broad",
            ): "나무와 풀 사이에서 사람들이 크게 웃고 이야기하는 소리, 보이지 않는 곳에서 동물이 우는 소리",
            (2, "broad"): (
                "나무와 풀 사이에서 사람들이 높고 크게 웃으며 이야기하는 "
                "소리, 보이지 않는 곳에서 동물이 날카롭게 우는 소리"
            ),
            (
                0,
                "specific",
            ): (
                "나무와 풀 사이에서 아이들이 웃는 소리, 어른들이 "
                "이야기하는 소리, 보이지 않는 곳에서 개가 짖는 소리"
            ),
            (1, "specific"): (
                "나무와 풀 사이에서 아이들이 크게 웃는 소리, 어른들이 "
                "이야기하는 소리, 보이지 않는 곳에서 개가 짖는 소리"
            ),
            (2, "specific"): (
                "나무와 풀 사이에서 아이들이 높고 크게 웃는 소리, 어른들이 낮게 이야기하는 소리, "
                "보이지 않는 곳에서 개가 날카롭게 짖는 소리"
            ),
            (
                0,
                "relation",
            ): (
                "나무와 풀 사이에서 웃는 아이들 곁으로 어른들의 "
                "이야기가 들리고, 멀리 보이지 않는 곳에서 개가 짖는다"
            ),
            (1, "relation"): (
                "나무와 풀 사이에 아이들 웃음소리가 크게 퍼지는 곁으로 어른들의 이야기가 들리고, "
                "멀리 보이지 않는 곳에서 개가 짖는다"
            ),
            (2, "relation"): (
                "나무와 풀 사이에 높고 크게 퍼지는 아이들 웃음소리 곁으로 어른들의 낮은 이야기가 들리고, "
                "멀리 보이지 않는 곳에서 개가 날카롭게 짖는다"
            ),
        },
    ),
    (
        _LIVE_EXAMPLES[1][0],
        {
            "seen": "나무와 풀",
            "visible_sounds": "바람에 흔들리는 나뭇잎 소리",
            "unseen_sounds": "계속 지저귀는 새소리",
        },
        {
            (
                0,
                "broad",
            ): "보이지 않는 곳에서 동물들이 우는 소리와 나무와 풀 사이로 바람이 스치는 자연의 소리",
            (
                1,
                "broad",
            ): "보이지 않는 곳에서 동물들이 일정하게 우는 소리와 나무와 풀 사이로 바람이 스치는 자연의 소리",
            (2, "broad"): (
                "보이지 않는 곳에서 동물들이 높고 일정하게 우는 소리와 "
                "나무와 풀 사이로 바람이 부드럽게 스치는 자연의 소리"
            ),
            (
                0,
                "specific",
            ): "보이지 않는 곳에서 새들이 지저귀는 소리, 나무와 풀 사이 나뭇잎이 바람에 흔들리는 소리",
            (
                1,
                "specific",
            ): (
                "보이지 않는 곳에서 새들이 일정하게 지저귀는 "
                "소리, 나무와 풀 사이 나뭇잎이 바람에 흔들리는 소리"
            ),
            (2, "specific"): (
                "보이지 않는 곳에서 새들이 높고 일정하게 지저귀는 "
                "소리, 나무와 풀 사이 나뭇잎이 부드럽게 흔들리는 소리"
            ),
            (
                0,
                "relation",
            ): "보이지 않는 곳에서 새들이 지저귀고, 눈앞의 나무와 풀 사이로 나뭇잎이 바람에 흔들린다",
            (
                1,
                "relation",
            ): (
                "보이지 않는 곳에서 새들이 일정하게 지저귀고, "
                "눈앞의 나무와 풀 사이로 나뭇잎이 바람에 흔들린다"
            ),
            (2, "relation"): (
                "보이지 않는 곳에서 새들이 높고 일정하게 지저귀고, "
                "눈앞의 나무와 풀 사이로 나뭇잎이 부드럽게 흔들린다"
            ),
        },
    ),
    (
        _LIVE_EXAMPLES[2][0],
        {
            "seen": "건물",
            "visible_sounds": "상인들이 외치는 소리, 손수레 바퀴가 덜컹거리는 소리",
            "unseen_sounds": "없음",
        },
        {
            (0, "broad"): "건물 앞에서 사람들이 외치는 소리와 기계가 오가는 소리",
            (1, "broad"): "건물 앞에서 사람들이 크게 외치는 소리와 기계가 오가는 소리",
            (2, "broad"): "건물 앞에서 사람들이 크고 높게 외치는 소리와 기계가 거칠게 오가는 소리",
            (0, "specific"): "건물 앞에서 상인들이 외치는 소리, 손수레 바퀴가 구르는 소리",
            (1, "specific"): "건물 앞에서 상인들이 크게 외치는 소리, 손수레 바퀴가 구르는 소리",
            (2, "specific"): "건물 앞에서 상인들이 크고 높게 외치는 소리, 손수레 바퀴가 거칠게 구르는 소리",
            (0, "relation"): "건물 앞 상인들의 외침 사이로 손수레 바퀴가 지나간다",
            (1, "relation"): "건물 앞에 상인들 외침이 크게 퍼지는 사이로 손수레 바퀴가 지나간다",
            (2, "relation"): "건물 앞에 크고 높게 퍼지는 외침 사이로 손수레 바퀴가 거칠게 지나간다",
        },
    ),
)
_LIVE_SCENE_TAIL: Final = (
    "이제 아래 자막을 바꿔 쓰세요. 바꿔 쓴 자막만 한 줄로, 70자 이내로 출력하세요.\n"
    + _SCENE_LINES
    + "원래 자막: {cue_text}"
)


def _live_scene_cell_text(texture: int, context: str) -> str:
    examples = "".join(
        f"예시\n{_SCENE_LINES.format(**scene)}"
        f"원래 자막: {caption}\n바꿔 쓴 자막: {rewrites[(texture, context)]}\n"
        for caption, scene, rewrites in _LIVE_SCENE_EXAMPLES
    )
    return (
        _LIVE_HEAD
        + _LIVE_TEXTURE[texture]
        + _LIVE_SCENE_CONTEXT[context]
        + _LIVE_SCENE_RULE
        + examples
        + _LIVE_SCENE_TAIL
    )


LIVE_SCENE_CELL_PROMPTS: Final[dict[tuple[int, str], PromptTemplate]] = {
    (texture, context): PromptTemplate(
        template_id="live-p2-scene", version=1, text=_live_scene_cell_text(texture, context)
    )
    for texture in range(LIVE_TEXTURE_MAX + 1)
    for context in CONTEXT_STEPS
}
LIVE_MODE_PROMPTS: Final[dict[str, dict[tuple[int, str], PromptTemplate]]] = {
    "acousmatic": LIVE_CELL_PROMPTS,
    "scene": LIVE_SCENE_CELL_PROMPTS,
}


# The live session's system message: our own "audio caption tuner" role, built
# once per session from the participant's phase-1 ratings and answers (researcher,
# 2026-10-12). The researcher writes the role text; the slots are
# {heard_families} (the sound families most attended to in phase 1, in Korean,
# or "없음") and {s1}, {s2}, {s3} (the phase-1 caption-rating means on 0-6, or
# "없음"). Until the researcher's text lands, the wording below is provisional
# and participant sessions are refused.
TUNER_PROVISIONAL: Final = True
TUNER_SYSTEM: Final = PromptTemplate(
    template_id="tuner-system",
    version=0,
    text=(
        "당신은 영상의 소리 자막을 시청자에게 맞게 다듬는 오디오 자막 조정기입니다. "
        "이 시청자가 앞 단계에서 가장 주의를 기울인 소리: {heard_families}. "
        "앞 단계 자막 평가(0-6점): 소리 표현 {s1}, 장면과의 일치 {s2}, 이해에 준 도움 {s3}."
    ),
)
TUNER_SLOTS: Final = ("heard_families", "s1", "s2", "s3")


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


# Phase 1's second caption (researcher, 2026-10-18): Trida tweaks a clip's raw
# first caption, steered by this participant's answers to that clip's first
# step. Since 2026-10-23 it takes one mode per clip, chosen by the binding model
# (diff.phase1_binding): scene-aware, which sets the sounds in the scene on screen,
# or acousmatic, which describes them as heard without the screen. Each mode has
# its own text, and its slots vary with the clip's scene (diff.phase1_tweak):
# {caption}; {sounds}, the clip's sounds from orig_text, which the prompt names and
# which are therefore R2's key (both modes keep every sound's name, so the key holds);
# {attended_sound} (A1s); {emphasis}; scene-aware: {scene_elements}, {seen}, {visible_sounds},
# {offscreen}; acousmatic: none of its own. The researcher approves the wording; until then it is
# provisional and participant sessions are refused. Version 2 (design review,
# 2026-10-24): the scene-aware prompt no longer asks for two centres (the attended
# sound and the V1 object) and ties each sound only to its own source, since an
# unrelated V1 object invited a made-up link; the acousmatic prompt makes the space
# a separate, optional request, left out when the scene is neither open nor enclosed.
# Version 3 (hypothesis team, 2026-10-24): one shared sentence places the sounds not on
# screen "here, though unseen" ({offscreen}, empty when there are none, never
# microphone wind), and the caption is held to 80 characters, since 160 take about 13 s
# to read at 12 characters per second, longer than the 10 s clip.
# Version 4 of the acousmatic text (researcher, 2026-10-26): the {space} request, a
# sentence on an open or enclosed feel drawn from the scene's sky share, is gone, so the
# acousmatic caption takes nothing from the picture.
# Scene-aware 4 and acousmatic 5 (researcher, 2026-10-26): both end with {emphasis}, the
# restorative quality this participant perceived most in the clip (phase1_tweak.EMPHASIS_KO),
# the same sentence under either mode. Captions are now written live and shown once they pass
# the gate (diff.phase1_usable), so the wording is approved here, once, rather than caption by
# caption, and participant sessions wait for it.
# Since 2026-10-26 (researcher) a caption the gate refuses is written again in another wording, since
# a greedy call repeats itself (Qwen3.5-9B, the writer that day). Claude, the writer since 2026-10-27
# (diff.claude_writer), keeps the wordings; its output varies, so the first wording is tried twice
# (researcher, 2026-10-27; on its bench the later wordings overran 80 characters in 8 of 10 calls). So the
# prompt has three wordings, tried in TWEAK_LADDER's order, each chosen on the 2026-10-26 delivery bench
# (artifacts/diff/bench/candidates-20261026-live, RULE.md and RESULTS.md):
# - every-sound (the bench's R1): the task first, the facts in sentences, the attended sound first,
#   every listed sound once and no other, no copying, one short sentence ending in 다 of about 40
#   characters;
# - checklist (R3): the same facts, the rules as a numbered list, one or two sentences of about 50;
# - prose (P2): every-sound's structure with one or two sentences of about 60, and only the listed
#   sounds.
# Every wording ends with {emphasis}'s sentence, the same under both modes, and keeps the 80-character
# limit's margin. The scene-aware 4 / acousmatic 5 "tweak-p1" texts they replace had the slots as
# "label: value" lines, which Trida answered with lists of names.
TWEAK_PROVISIONAL: Final = True
TWEAK_WORDINGS: Final[dict[str, dict[str, PromptTemplate]]] = {
    "every-sound": {
        "scene": PromptTemplate(
            template_id="tweak-p1-every-sound",
            version=1,
            text=(
                "이 시청자를 위해 영상의 소리 자막을 새로 쓰세요. "
                "소리가 화면 속 장면에서 나는 것으로 들리게 쓰는 것이 목표입니다.\n"
                "처음 자막은 「{caption}」입니다.\n"
                "이 영상에서 실제로 들리는 소리는 {sounds}입니다. "
                "이 시청자는 그중 {attended_sound}에 가장 주의를 기울였습니다.\n"
                "화면에서 두드러지는 것은 {scene_elements}이고, 장면의 배경은 {seen}입니다. "
                "화면에 음원이 보이는 소리는 {visible_sounds}입니다.\n"
                "이 시청자가 주의를 기울인 소리를 중심에 두고, 장면의 배경을 소리가 들리는 곳으로 쓰세요. "
                "음원이 보이는 소리는 그 소리를 내는 대상과만 이어 주세요.{offscreen} 화면에 없는 대상이나 "
                "장소는 지어내지 마세요. "
                "{emphasis}\n"
                "이 시청자가 주의를 기울인 소리부터 쓰세요. "
                "위에 적힌 소리를 빠짐없이 한 번씩 쓰고 그 밖의 소리는 쓰지 말고, 처음 자막에 있더라도 "
                "위에 없는 소리는 빼세요. "
                "처음 자막의 문장을 그대로 옮기지 마세요. "
                "소리 이름을 쉼표로 늘어놓지 말고, '다'로 끝나는 짧은 문장 하나로 40자 안팎으로 쓰세요. "
                "새로 쓴 자막만 출력하고, 앞에 '자막' 같은 말이나 따옴표, 설명을 붙이지 마세요."
            ),
        ),
        "acousmatic": PromptTemplate(
            template_id="tweak-p1-every-sound",
            version=1,
            text=(
                "이 시청자를 위해 영상의 소리 자막을 새로 쓰세요. "
                "화면을 보지 않고 귀로만 듣는 것처럼, 소리가 어떻게 들리는지 쓰는 것이 목표입니다.\n"
                "처음 자막은 「{caption}」입니다.\n"
                "이 영상에서 실제로 들리는 소리는 {sounds}입니다. "
                "이 시청자는 그중 {attended_sound}에 가장 주의를 기울였습니다.\n"
                "이 시청자가 주의를 기울인 소리를 중심에 두고, 각 소리가 무슨 소리인지는 밝히되 화면 속 "
                "대상이나 장소는 말하지 마세요. "
                "소리가 어떻게 들리는지(크기, 높낮이, 거리, 이어짐)를 살려 쓰세요. "
                "{emphasis}\n"
                "이 시청자가 주의를 기울인 소리부터 쓰세요. "
                "위에 적힌 소리를 빠짐없이 한 번씩 쓰고 그 밖의 소리는 쓰지 말고, 처음 자막에 있더라도 "
                "위에 없는 소리는 빼세요. "
                "처음 자막의 문장을 그대로 옮기지 마세요. "
                "소리 이름을 쉼표로 늘어놓지 말고, '다'로 끝나는 짧은 문장 하나로 40자 안팎으로 쓰세요. "
                "새로 쓴 자막만 출력하고, 앞에 '자막' 같은 말이나 따옴표, 설명을 붙이지 마세요."
            ),
        ),
    },
    "checklist": {
        "scene": PromptTemplate(
            template_id="tweak-p1-checklist",
            version=1,
            text=(
                "이 시청자를 위해 영상의 소리 자막을 새로 쓰세요. "
                "소리가 화면 속 장면에서 나는 것으로 들리게 쓰는 것이 목표입니다.\n"
                "처음 자막은 「{caption}」입니다.\n"
                "이 영상에서 실제로 들리는 소리는 {sounds}입니다. "
                "이 시청자는 그중 {attended_sound}에 가장 주의를 기울였습니다.\n"
                "화면에서 두드러지는 것은 {scene_elements}이고, 장면의 배경은 {seen}입니다. "
                "화면에 음원이 보이는 소리는 {visible_sounds}입니다.{offscreen}\n"
                "지킬 것:\n"
                "1. 이 시청자가 주의를 기울인 소리부터 쓰세요.\n"
                "2. 음원이 보이는 소리는 그 소리를 내는 대상과만 잇고, 화면에 없는 대상이나 장소는 "
                "지어내지 마세요.\n"
                "3. {emphasis}\n"
                "4. 위에 적힌 소리를 빠짐없이 한 번씩 쓰고, 그 밖의 소리는 쓰지 마세요. "
                "처음 자막에 있더라도 위에 없는 소리는 빼세요.\n"
                "5. 처음 자막의 문장을 그대로 옮기지 마세요.\n"
                "6. '다'로 끝나는 문장 한두 개로 50자 안팎으로 쓰세요.\n"
                "새로 쓴 자막만 출력하고, 번호나 따옴표, 설명을 붙이지 마세요."
            ),
        ),
        "acousmatic": PromptTemplate(
            template_id="tweak-p1-checklist",
            version=1,
            text=(
                "이 시청자를 위해 영상의 소리 자막을 새로 쓰세요. "
                "화면을 보지 않고 귀로만 듣는 것처럼, 소리가 어떻게 들리는지 쓰는 것이 목표입니다.\n"
                "처음 자막은 「{caption}」입니다.\n"
                "이 영상에서 실제로 들리는 소리는 {sounds}입니다. "
                "이 시청자는 그중 {attended_sound}에 가장 주의를 기울였습니다.\n"
                "지킬 것:\n"
                "1. 이 시청자가 주의를 기울인 소리부터 쓰세요.\n"
                "2. 각 소리가 무슨 소리인지는 밝히되 화면 속 대상이나 장소는 말하지 말고, 소리가 어떻게 "
                "들리는지(크기, 높낮이, 거리, 이어짐)를 살려 쓰세요.\n"
                "3. {emphasis}\n"
                "4. 위에 적힌 소리를 빠짐없이 한 번씩 쓰고, 그 밖의 소리는 쓰지 마세요. "
                "처음 자막에 있더라도 위에 없는 소리는 빼세요.\n"
                "5. 처음 자막의 문장을 그대로 옮기지 마세요.\n"
                "6. '다'로 끝나는 문장 한두 개로 50자 안팎으로 쓰세요.\n"
                "새로 쓴 자막만 출력하고, 번호나 따옴표, 설명을 붙이지 마세요."
            ),
        ),
    },
    "prose": {
        "scene": PromptTemplate(
            template_id="tweak-p1-prose",
            version=1,
            text=(
                "이 시청자를 위해 영상의 소리 자막을 새로 쓰세요. "
                "소리가 화면 속 장면에서 나는 것으로 들리게 쓰는 것이 목표입니다.\n"
                "처음 자막은 「{caption}」입니다.\n"
                "이 영상에서 실제로 들리는 소리는 {sounds}입니다. "
                "이 시청자는 그중 {attended_sound}에 가장 주의를 기울였습니다.\n"
                "화면에서 두드러지는 것은 {scene_elements}이고, 장면의 배경은 {seen}입니다. "
                "화면에 음원이 보이는 소리는 {visible_sounds}입니다.\n"
                "이 시청자가 주의를 기울인 소리를 중심에 두고, 장면의 배경을 소리가 들리는 곳으로 쓰세요. "
                "음원이 보이는 소리는 그 소리를 내는 대상과만 이어 주세요.{offscreen} 화면에 없는 대상이나 "
                "장소는 지어내지 마세요. "
                "{emphasis}\n"
                "이 시청자가 주의를 기울인 소리부터 쓰세요. "
                "위에 적힌 소리만 쓰고, 처음 자막에 있더라도 위에 없는 소리는 빼세요. "
                "처음 자막의 문장을 그대로 옮기지 마세요. "
                "소리 이름을 쉼표로 늘어놓지 말고, '다'로 끝나는 완결된 문장 한두 개로 60자 안팎으로 쓰세요. "
                "새로 쓴 자막만 출력하고, 앞에 '자막' 같은 말이나 따옴표, 설명을 붙이지 마세요."
            ),
        ),
        "acousmatic": PromptTemplate(
            template_id="tweak-p1-prose",
            version=1,
            text=(
                "이 시청자를 위해 영상의 소리 자막을 새로 쓰세요. "
                "화면을 보지 않고 귀로만 듣는 것처럼, 소리가 어떻게 들리는지 쓰는 것이 목표입니다.\n"
                "처음 자막은 「{caption}」입니다.\n"
                "이 영상에서 실제로 들리는 소리는 {sounds}입니다. "
                "이 시청자는 그중 {attended_sound}에 가장 주의를 기울였습니다.\n"
                "이 시청자가 주의를 기울인 소리를 중심에 두고, 각 소리가 무슨 소리인지는 밝히되 화면 속 "
                "대상이나 장소는 말하지 마세요. "
                "소리가 어떻게 들리는지(크기, 높낮이, 거리, 이어짐)를 살려 쓰세요. "
                "{emphasis}\n"
                "이 시청자가 주의를 기울인 소리부터 쓰세요. "
                "위에 적힌 소리만 쓰고, 처음 자막에 있더라도 위에 없는 소리는 빼세요. "
                "처음 자막의 문장을 그대로 옮기지 마세요. "
                "소리 이름을 쉼표로 늘어놓지 말고, '다'로 끝나는 완결된 문장 한두 개로 60자 안팎으로 쓰세요. "
                "새로 쓴 자막만 출력하고, 앞에 '자막' 같은 말이나 따옴표, 설명을 붙이지 마세요."
            ),
        ),
    },
}
TWEAK_LADDER: Final = ("every-sound", "every-sound", "checklist", "prose")
TWEAK_PROMPTS: Final = TWEAK_WORDINGS[TWEAK_LADDER[0]]  # the first rung's, by mode
_TWEAK_COMMON: Final = ("mode", "caption", "sounds", "attended_sound", "emphasis")
TWEAK_SLOTS: Final[dict[str, tuple[str, ...]]] = {
    "scene": (*_TWEAK_COMMON, "scene_elements", "seen", "visible_sounds", "offscreen"),
    "acousmatic": _TWEAK_COMMON,
}


def render_tweak(slots: Mapping[str, str], wording: str = TWEAK_LADDER[0]) -> RenderedPrompt:
    """Phase 1's second-caption prompt in ``wording`` for the slots' mode, every slot filled; the fixed
    system message."""
    mode = slots.get("mode")
    if mode not in TWEAK_SLOTS:
        raise ValueError(f"the tweak prompt's mode must be one of {sorted(TWEAK_SLOTS)}, not {mode!r}")
    if set(slots) != set(TWEAK_SLOTS[mode]):
        raise ValueError(f"the {mode} tweak prompt needs exactly the slots {list(TWEAK_SLOTS[mode])}")
    if wording not in TWEAK_WORDINGS:
        raise ValueError(f"the tweak prompt's wording must be one of {list(TWEAK_WORDINGS)}, not {wording!r}")
    template = TWEAK_WORDINGS[wording][mode]
    return RenderedPrompt(
        template.template_id,
        template.version,
        template.sha256,
        0,  # no evidence is rendered into a tweak prompt
        None,
        template.text.format(**slots),
    )


def render_tuner_system(slots: Mapping[str, str]) -> str:
    """The session's system message: the tuner template with every slot filled."""
    if set(slots) != set(TUNER_SLOTS):
        raise ValueError(f"the tuner needs exactly the slots {list(TUNER_SLOTS)}")
    return TUNER_SYSTEM.text.format(**slots)


def render_live(
    source: CaptionSource,
    texture: int,
    context: str,
    system_message: str,
    mode: str = "acousmatic",
    slots: Mapping[str, str] | None = None,
) -> RenderedPrompt:
    """The live phase-2 prompt: the slider cell's own prompt over a cue's starting caption, under the tuner.

    The prompts are ``LIVE_MODE_PROMPTS[mode]``, nine per mode; the scene-aware ones take the cue's
    scene slots (``phase2_scene.cue_steer``). The job's cell names which one a call used.
    """
    if context not in CONTEXT_STEPS:
        raise ValueError(f"Unknown context step: {context}")
    if type(texture) is not int or not 0 <= texture <= LIVE_TEXTURE_MAX:
        raise ValueError("texture must be an integer step from 0 to 2")
    if not system_message:
        raise ValueError("a live prompt needs the session's tuner system message")
    if mode not in LIVE_MODE_PROMPTS:
        raise ValueError(f"Unknown live mode: {mode}")
    needed = {"seen", "visible_sounds", "unseen_sounds"} if mode == "scene" else set()
    if set(slots or {}) != needed:
        raise ValueError(
            f"the {mode} live prompt takes the slots {sorted(needed)}, got {sorted(slots or {})}"
        )
    template = LIVE_MODE_PROMPTS[mode][(texture, context)]
    return RenderedPrompt(
        template.template_id,
        template.version,
        template.sha256,
        0,  # no evidence is rendered into a live prompt
        LIVE_RULES_VERSION,
        template.text.format(cue_text=source.text, **dict(slots or {})),
        system_message=system_message,
    )
