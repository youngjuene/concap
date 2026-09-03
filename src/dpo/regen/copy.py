"""Every string the participant reads, once.

The strings live here and nowhere else. The app serves this dictionary to the
page, so the browser holds no second copy that could drift, and a test can
compare one object against the specification.

All copy is an English placeholder in the instrument's voice, pending the
study's own wording; the survey items are not here at all — §9.2 holds those as
data outside the code, in :mod:`dpo.regen.items`. What is here is the chrome:
instructions, button labels, and the waiting-screen message §6 requires.

Two strings carry more weight than their length suggests.

``headphones`` is §2's instruction line. The study is about sound captions, and
a participant on laptop speakers at 20% volume is not in the study; the line is
the only chance to fix that before the first clip plays.

``waiting.stay`` is §6's message discouraging abandonment. It has to say that
the wait is finite without promising a duration the ceiling may not keep, so it
names what is happening rather than how long it will take.

Section numbers cite ``docs/v3-regen/spec-behavior.md``.
"""

from __future__ import annotations

from typing import Any

STRINGS: dict[str, Any] = {
    "app_title": "Sound captions",
    "steps": ["Watch", "Questions", "What you saw", "What you heard", "Watch again", "Survey"],
    "actions": {
        "start": "Start",
        "next": "Next",
        "submit": "Submit",
        "clear": "Clear all points",
        "download": "Download session log",
    },
    "view": {
        "headphones": "Put your headphones on and set the volume where you would normally listen. "
        "The clip plays once, in fullscreen, and cannot be paused or replayed.",
        "ready": "Press Start when you are ready.",
    },
    "visual": {
        "instruction": "Click on the things that make this scene what it is. "
        "Click a point again to remove it, or drag it to move it.",
        # The floor is a configuration value (§9.4's sibling in the same
        # calibration), so the sentence is completed by the server rather than
        # stating a number this file would have to be kept in step with.
        "minimum": "Select at least {minimum} before continuing.",
        "placed": "{count} selected",
    },
    "auditory": {
        "instruction": "These are the separated sounds of the clip you just watched. "
        "Play any of them, then select the ones you noticed while watching.",
        "play": "Play",
        "stop": "Stop",
        # A lane whose audio will not play. §5 keeps the selection control
        # live — the participant may have heard the source in the clip even
        # though this screen cannot replay it — so the word is about the
        # button, not about the source.
        "unavailable": "No sound",
        "select": "Select",
        "selected": "Selected",
        "none": "You can continue without selecting any.",
    },
    "waiting": {
        "heading": "Writing the captions for your next clip",
        "stay": "This uses what you just told us, so it is being written now rather than chosen "
        "from a list. Please keep this window open — leaving now ends the session.",
    },
    "survey": {
        "instruction": "Answer every item. The Next button enables when none are left blank.",
        "remaining": "{count} left",
    },
    "done": {
        "heading": "That is everything",
        "body": "Thank you. You can close this window.",
    },
    "error": "Something went wrong. Tell the researcher rather than reloading.",
}
