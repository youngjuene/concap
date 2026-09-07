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
    "steps": ["Watch", "Questions", "What you saw", "What you heard", "Watch again", "Survey"],
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
        "instruction": "Click the things that make this scene what it is, then go on to the next moment.",
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
    "auditory": {
        "instruction": "These are the separated sounds of the clip you just watched. Play any of "
        "them, then tick the ones you noticed while watching.",
        "legend": "The separated sounds",
        # There was no way to hear the clip again on this screen, so every
        # judgment was a stem against a ten-second memory.
        "mix": "The clip as you heard it",
        "play": "Play",
        "stop": "Stop",
        # A lane whose audio will not play. §5 keeps the selection control
        # live — the participant may have heard the source in the clip even
        # though this screen cannot replay it — so the word is about the
        # button, not about the source.
        "unavailable": "No sound",
        # The selection control is a checkbox, so its label is what ticking it
        # would claim rather than what state it is in. "Selected" on a button
        # you press to deselect reads as already done rather than reversible.
        "noticed": "Noticed",
        "chosen": "{count} of {total} ticked",
        "chosen_none": "Nothing ticked yet",
        "confirm_none": "Nothing ticked. If you noticed none of these, press Submit again.",
        "seek": "Waveform for {label}. Click or use the arrow keys to hear from a point.",
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
        "ahead_body": "You will watch the clip once more with its new captions, then answer the "
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
