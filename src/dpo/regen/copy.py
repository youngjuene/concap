"""Every string the participant reads, once.

The strings live here and nowhere else. The app serves this dictionary to the
page, so the browser holds no second copy that could drift, and a test can
compare one object against the specification.

All copy is an English placeholder in the instrument's voice, pending the
study's own wording; the survey items are not here at all — §9.2 holds those as
data outside the code, in :mod:`dpo.regen.items`. What is here is the chrome:
instructions, button labels, and the waiting-screen message §6 requires.

Four strings carry more weight than their length suggests.

``headphones`` is §2's instruction line. The study is about sound captions, and
a participant on laptop speakers at 20% volume is not in the study; the line is
the only chance to fix that before the first clip plays.

``waiting.stay`` is §6's message discouraging abandonment. It has to say that
the wait is finite without promising a duration the ceiling may not keep, so it
names what is happening rather than how long it will take. It used to be the
whole of §6's answer to drop-off, and a sentence saying that leaving destroys
the session is a threat rather than a reason to stay — so ``ahead`` now says
what remains, and the elapsed count says the page is alive. Between them they
are the cheapest anti-abandonment devices available, and neither promises
anything the calibration does not already guarantee.

``steps`` was written on the first day and then discarded: the rail used the
six names only for their count. They are the headings now. A participant told
they are on *What you heard*, four of six, is a participant who knows the wait
is nearly over.

``auditory.confirm_none`` is the one string here that changes an interaction.
§5 could be finished in a single click, on a screen whose result conditions the
captions the participant then rates in §7 and §8 — an empty §5 does not thin
one measure, it moves the stimulus for two more. Submitting nothing is still a
valid answer, so it is still allowed; it now takes a second press and says so,
which costs a participant who means it one keystroke and stops the one who is
scanning for the way forward.

Section numbers cite ``docs/v3-regen/spec-behavior.md``; findings cite the
interface audit in ``docs/v3-regen/interface-audit.md``.
"""

from __future__ import annotations

from typing import Any

STRINGS: dict[str, Any] = {
    "app_title": "Sound captions",
    # The study's name is a constant, so it rides in the eyebrow with the step
    # number rather than being the h1 on five screens in a row while the thing
    # the screen wants sits under it in grey.
    "eyebrow": "Step {n} of {total} · {study}",
    # The language toggle. Each button is labelled in its own language, so a
    # participant who cannot read the other one can still find theirs.
    "languages": {
        "label": "Language",
        "names": {"en": "English", "ko": "한국어"},
        "locked": "Fixed for this session",
    },
    # "Watch again" said the second viewing was the first clip repeated. It is
    # the *other* segment (§1: `Assignment.regenerated_segment`), so the rail,
    # the heading it feeds and §6's "what happens next" all promised a
    # re-watch the participant never gets. A participant framed to expect the
    # same street can rate the second clip's ART items as a re-rating of the
    # first, which is the one contrast the study rests on.
    "steps": ["Watch", "Questions", "What you saw", "What you heard", "Second clip", "Survey"],
    "actions": {
        "start": "Start",
        "next": "Next",
        "submit": "Submit",
        # "mark" throughout §4: the instruction, the count and these buttons
        # say the same word for the same thing.
        "clear": "Clear all marks",
        "undo": "Undo last mark",
        "download": "Download session log",
        "retry": "Try that step again",
        "resume": "Resume the clip",
    },
    "view": {
        "headphones": "Put your headphones on and set the volume where you would normally listen. "
        "The clip plays once, in fullscreen, and cannot be paused or replayed.",
        "ready": "Press Start when you are ready.",
        # Shown on the second viewing only. Renaming the step to "Second clip"
        # fixed the label; this fixes the meaning. The screen was otherwise
        # word for word the first viewing's, so a participant who had read
        # "watch again" had nothing to correct them — and one who believes
        # they are re-watching the same street will attribute any change in
        # how it feels to the captions, which is the inference the study is
        # trying to make independently.
        "different": "This is a different scene from the first one.",
        # On the button while the clip is being fetched whole, before it can
        # be started: the wait is real off campus and a dead button is not an
        # explanation.
        "preparing": "Preparing the clip…",
        # Escape is the browser's own shortcut and the first thing a nervous
        # participant tries. The clip used to keep playing in the column with
        # the band pinned to the viewport, and the viewing recorded clean.
        "interrupted_heading": "The clip stopped filling the screen",
        "interrupted_body": "It has been paused. Press Resume to watch the rest full screen, "
        "and tell the researcher this happened.",
    },
    "visual": {
        # The moment is in the heading because the participant is choosing a
        # moment as much as a thing, and the heading is where they will look
        # for which one they are on.
        "heading": "What you saw at {seconds} seconds",
        # "the things that make this scene what it is" reads two ways — the
        # objects that identify the place, or the qualities that give it its
        # feel — and the two produce different marks and so different captions.
        # The matching settles it: points are matched against object masks
        # (Building, Person, Road), so the question is about things, and the
        # sentence now says so.
        "instruction": "Click the things in this picture that make this place what it is — the "
        "objects and features you would name if you were describing it to someone.",
        "keyboard": "Or use the arrow keys to move the crosshair and Enter to place a mark.",
        "drag": "A mark can be dragged. The ✕ beside it removes it.",
        # The floor is a configuration value (§9.4's sibling in the same
        # calibration), so the sentence is completed by the server rather than
        # stating a number this file would have to be kept in step with. One is
        # the default and reads badly as a numeral, so it has its own line.
        # It sits on its own reserved line rather than being appended to the
        # instruction, which used to lose a sentence — and a line of height —
        # the instant the first mark landed, shifting the picture up under the
        # cursor that had just placed it.
        "minimum": "Mark at least {minimum} things to continue.",
        "minimum_one": "Mark at least one thing to continue.",
        # Marks are per-frame and the count was reported across all of them, so
        # "5 marked" could mean five on one moment or one on each of five.
        "placed": "{count} marked · {moments} of {total} moments",
        "placed_none": "No marks yet",
        "moment": "{seconds}s",
        "tally": "{count}",
        "tally_none": "—",
        "moments_label": "Moments from the clip",
        "back": "Previous moment",
        "on": "Next moment",
        "plate": "The frame at {seconds} seconds. Click to place a mark.",
    },
    # §5 is a fixed judgment on five families now, not a selection among the
    # sources a clip happens to carry. The labels are AudioSet's own top-level
    # names because that is the vocabulary the study reports in; each carries
    # examples underneath, because "Sounds of things" is a taxonomy label and
    # a participant should not have to guess what counts.
    "auditory": {
        # "not whether you can hear it now" is the guard. It was written for
        # this screen in an earlier commit and then lost when the whole block
        # was replaced by the five-family redesign — the screen still carries
        # the clip's audio, so without the sentence the question reads as
        # being about what is audible on this page rather than about the
        # viewing. There is a test for it now, which is what was missing.
        # Answered from memory, and the page says so, because there is now
        # nothing on it to listen to. The reference mix was removed: the whole
        # reason all five families are asked of everyone is that a false alarm
        # becomes measurable, and a false alarm only exists against a memory.
        # With the clip replayable the screen was a listening test with a
        # ceiling, and it contradicted §2's own "cannot be paused or replayed".
        "instruction": "For each kind of sound, say whether you heard it while the clip was "
        "playing. Answer from memory — there is nothing to listen to on this page. Answer all "
        "five, including the ones you did not hear.",
        "legend": "The five kinds of sound",
        "families": {
            "human": "Human sounds",
            "animal": "Animal",
            "things": "Sounds of things",
            "music": "Music",
            "natural": "Natural sounds",
        },
        "examples": {
            "human": "talking, footsteps, laughter",
            "animal": "birds, dogs, insects",
            "things": "vehicles, machinery, doors",
            "music": "any music, played or live",
            "natural": "wind, water, rain",
        },
        "heard": "Heard",
        "not_heard": "Didn't hear",
        "answered": "{count} of {total} answered",
        "answered_none": "Nothing answered yet",
    },
    "waiting": {
        "eyebrow": "One moment — nothing to do",
        "heading": "Writing the captions for your next clip",
        "explains": "This uses what you just told us, so it is being written now rather than "
        "chosen from a list.",
        "stay": "Please keep this window open — leaving now ends the session.",
        # A number that moves is the proof of life a waiting participant is
        # actually asking for. The ceiling is named as a ceiling, which
        # promises nothing the calibration does not already guarantee.
        "elapsed": "{seconds}s elapsed",
        "ceiling": "at most {seconds}s",
        # The live region says what is happening; the elapsed counter beside it
        # carries the number. Putting the seconds in both made the announced
        # sentence and the visible count disagree — a region polite enough not
        # to interrupt every second is a region a second behind — and announcing
        # every tick is not a thing to do to a screen reader anyway.
        "status": "Still writing. This screen moves on by itself when the captions are ready.",
        # A cue that has been written is an event that happened, so saying how
        # many have arrived predicts nothing. Four announcements across the
        # wait is a live region worth having; a per-second one is not.
        "status_at": "{done} of {total} captions written.",
        "status_done": "The captions are ready. Going on to the next clip.",
        "ahead_heading": "Two steps left after this",
        "ahead_body": "You will watch a second clip with these new captions, then answer the "
        "last set of questions.",
    },
    "survey": {
        "instruction": "{count} questions about the scene you just watched. Answer every one.",
        "instruction_all": "{blocks} sets of questions, {count} in all. Answer every one.",
        "remaining": "{count} of {total} left",
        "complete": "All {total} answered",
        "block_done": "{done} of {total}",
        # The count used to sit at the bottom of a 22-row scroll and say how
        # many were left without saying which.
        "first": "Go to the first unanswered",
    },
    "done": {
        "heading": "That is everything",
        "body": "Thank you. Your answers are recorded — you can close this window.",
        # After a screen that warned them leaving would end the session, the
        # participant's last impression was an assurance that never came.
        "receipt": "Session {participant} · {at} · saved",
        "for_researcher": "For the researcher",
    },
    # A published instrument refuses new enrolments without the current access
    # code. Whoever lands here was not invited, or was invited to an earlier
    # opening; either way there is nothing for them to do but ask.
    "closed": {
        "heading": "This study is not open right now",
        "body": "The link you used is not active. If you were invited, ask the researcher "
        "for the current link.",
    },
    # fail() is called with a real message from six places and then fell back to
    # one generic sentence, so what the researcher standing behind the
    # participant got told was "something went wrong" whatever had happened.
    "error": {
        "heading": "The session has stopped here",
        "body": "Nothing you have done is lost. Please tell the researcher — reloading will not help.",
        "kept": "Everything up to this step is saved.",
        "unknown": "Something went wrong.",
        "labels": {"session": "SESSION", "step": "STEP", "at": "AT", "cause": "CAUSE"},
    },
}


# The same chrome in Korean. A parallel tree rather than a Korean form beside
# every English leaf: the comments above explain why each English string is
# worded as it is, and interleaving would bury them. Drift is prevented by a
# test instead — ``tests/regen/test_copy.py`` asserts the two trees have the
# same shape and that every string carries the same placeholders — which is a
# stronger guarantee than adjacency, because it fails on a missing key rather
# than relying on someone noticing one.
#
# Addressed to the participant in 합니다체 throughout: the instrument asks
# people to do things, and a research instrument that switches register between
# screens reads as two instruments.
#
# The survey items are NOT here. §9.2 holds those in :mod:`dpo.regen.items`,
# they are the ART and PRSS scales, and translating a measurement instrument is
# a validity question rather than a wording one. They stay in English until the
# study says where their Korean comes from.
KOREAN: dict[str, Any] = {
    "app_title": "소리 자막",
    "eyebrow": "{total}단계 중 {n}단계 · {study}",
    # Each button stays labelled in its own language, so a participant who
    # cannot read the other one can still find theirs. That is why "names" is
    # identical in both trees rather than translated.
    "languages": {
        "label": "언어",
        "names": {"en": "English", "ko": "한국어"},
        "locked": "이 세션에서는 변경할 수 없습니다",
    },
    # "다시 시청" was worse than its English original: 시청 and 다시 볼 수 없습니다
    # collide on the same screen, so the heading told the participant to watch
    # again four lines above the line saying they cannot.
    "steps": ["시청", "질문", "본 것", "들은 것", "두 번째 영상", "설문"],
    "actions": {
        "start": "시작",
        "next": "다음",
        "submit": "제출",
        "clear": "표시 모두 지우기",
        "undo": "마지막 표시 취소",
        "download": "세션 기록 내려받기",
        "retry": "이 단계 다시 시도",
        "resume": "영상 이어보기",
    },
    "view": {
        "headphones": "헤드폰을 착용하시고 평소 듣는 음량으로 맞춰 주십시오. "
        "영상은 전체 화면에서 한 번만 재생되며, 일시정지하거나 다시 볼 수 없습니다.",
        "ready": "준비되면 시작을 누르십시오.",
        "different": "이번에는 첫 번째와 다른 장면입니다.",
        "preparing": "영상을 준비하는 중…",
        "interrupted_heading": "영상이 전체 화면에서 벗어났습니다",
        "interrupted_body": "재생이 일시정지되었습니다. 이어보기를 눌러 나머지를 전체 화면으로 "
        "시청하시고, 이 사실을 연구자에게 알려 주십시오.",
    },
    "visual": {
        "heading": "{seconds}초에 보신 것",
        # "이 장면을 이 장면답게" was a word-for-word calque that repeats 장면
        # twice in six syllables, and "클릭하신 다음, 다음 순간으로" stuttered.
        # Both sat in the sentence that has to be understood on first read.
        # The English is one sentence and says nothing about moving on — the
        # picker is the navigation and speaks for itself. The Korean had kept a
        # third sentence telling the participant to advance, so the two arms
        # were being given different instructions on the screen that produces
        # half of §6's input. It also displaced the "또는" in `keyboard`, which
        # then attached to advancing rather than to clicking.
        "instruction": "이곳을 이곳답게 만드는 것들을 사진에서 클릭해 주십시오. 누군가에게 이 "
        "장소를 설명한다면 이름을 댈 만한 사물이나 특징을 고르면 됩니다.",
        "keyboard": "또는 방향키로 십자선을 옮기고 Enter로 표시하실 수 있습니다.",
        "drag": "표시는 끌어서 옮길 수 있습니다. 옆의 ✕를 누르면 지워집니다.",
        "minimum": "계속하시려면 적어도 {minimum}곳은 표시해 주십시오.",
        "minimum_one": "계속하시려면 적어도 한 곳은 표시해 주십시오.",
        "placed": "{count}개 표시 · {total}개 순간 중 {moments}개",
        "placed_none": "아직 표시가 없습니다",
        "moment": "{seconds}초",
        "tally": "{count}",
        "tally_none": "—",
        "moments_label": "영상 속 순간들",
        "back": "이전 순간",
        "on": "다음 순간",
        "plate": "{seconds}초의 화면입니다. 클릭하여 표시하십시오.",
    },
    "auditory": {
        "instruction": "아래 다섯 가지 소리 각각에 대해, 영상이 재생되는 동안 들었는지를 답해 "
        "주십시오. 이 화면에서는 다시 들어 볼 수 없으니 기억에 따라 답해 주십시오. 듣지 못한 "
        "것까지 모두 답해 주십시오.",
        "legend": "다섯 가지 소리",
        "families": {
            "human": "사람 소리",
            "animal": "동물 소리",
            "things": "사물 소리",
            "music": "음악",
            "natural": "자연 소리",
        },
        "examples": {
            "human": "말소리, 발소리, 웃음소리",
            "animal": "새, 개, 곤충",
            "things": "차량, 기계, 문",
            "music": "재생된 음악이든 라이브든",
            "natural": "바람, 물, 비",
        },
        "heard": "들었음",
        "not_heard": "못 들었음",
        "answered": "{total}가지 중 {count}가지 답변",
        "answered_none": "아직 답하지 않으셨습니다",
    },
    "waiting": {
        "eyebrow": "잠시만 기다려 주십시오 — 하실 일은 없습니다",
        "heading": "다음 영상의 자막을 작성하고 있습니다",
        "explains": "방금 알려 주신 내용을 사용하기 때문에, 목록에서 고르는 것이 아니라 "
        "지금 작성하고 있습니다.",
        "stay": "이 창을 닫지 말아 주십시오 — 지금 나가시면 세션이 종료됩니다.",
        "elapsed": "{seconds}초 경과",
        "ceiling": "최대 {seconds}초",
        "status": "작성 중입니다. 자막이 준비되면 이 화면은 자동으로 넘어갑니다.",
        "status_at": "자막 {total}개 중 {done}개를 작성했습니다.",
        "status_done": "자막이 준비되었습니다. 다음 영상으로 넘어갑니다.",
        "ahead_heading": "이후 두 단계가 남았습니다",
        "ahead_body": "새 자막이 붙은 두 번째 영상을 보신 뒤, 마지막 설문에 답해 주시면 됩니다.",
    },
    "survey": {
        "instruction": "방금 보신 장면에 관한 질문 {count}개입니다. 모두 답해 주십시오.",
        "instruction_all": "질문 묶음 {blocks}개, 모두 {count}개입니다. 빠짐없이 답해 주십시오.",
        "remaining": "{total}개 중 {count}개 남음",
        "complete": "{total}개 모두 답변 완료",
        "block_done": "{total}개 중 {done}개",
        "first": "답하지 않은 첫 질문으로",
    },
    "done": {
        "heading": "모두 끝났습니다",
        "body": "감사합니다. 답변이 기록되었습니다 — 이 창을 닫으셔도 됩니다.",
        "receipt": "세션 {participant} · {at} · 저장됨",
        "for_researcher": "연구자용",
    },
    "closed": {
        "heading": "지금은 연구가 열려 있지 않습니다",
        "body": "사용하신 링크가 활성화되어 있지 않습니다. 초대를 받으셨다면 연구자에게 "
        "현재 링크를 문의해 주십시오.",
    },
    "error": {
        "heading": "세션이 여기서 중단되었습니다",
        "body": "지금까지 하신 내용은 사라지지 않았습니다. 연구자에게 알려 주십시오 — "
        "새로고침은 도움이 되지 않습니다.",
        "kept": "이 단계까지의 내용은 모두 저장되었습니다.",
        "unknown": "문제가 발생했습니다.",
        "labels": {"session": "세션", "step": "단계", "at": "시각", "cause": "원인"},
    },
}

# Every language the chrome is written in. English is the fallback, and the
# only one guaranteed complete.
TRANSLATIONS: dict[str, dict[str, Any]] = {"en": STRINGS, "ko": KOREAN}


def strings_for(language: str) -> dict[str, Any]:
    """The chrome in ``language``, falling back to English string by string.

    Per string rather than per tree: a language that is missing one line should
    show that line in English and the rest in its own, which is what a
    participant can actually read, rather than reverting the whole interface
    over one gap.
    """
    translation = TRANSLATIONS.get(language)
    if translation is None or translation is STRINGS:
        return STRINGS
    # The top level is built here rather than by ``_merge`` so the return is a
    # mapping by construction; ``_merge`` recurses into whatever each value is.
    return {key: _merge(value, translation.get(key, value)) for key, value in STRINGS.items()}


def _merge(base: Any, over: Any) -> Any:
    if isinstance(base, dict) and isinstance(over, dict):
        return {key: _merge(value, over.get(key, value)) for key, value in base.items()}
    return over if over is not None else base
