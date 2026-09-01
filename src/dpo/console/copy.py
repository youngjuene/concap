"""Table 3 of ``docs/v2-console/spec-uiux.md``, verbatim, once.

The strings live here and nowhere else. The app serves this dictionary to the
page, so the browser holds no second copy that could drift from the
specification, and a test can compare one object against the table.

All copy is an English placeholder pending the Korean construct check, and the
criterion sentence in particular is a construct decision, not a localization
task (§5, and §13's open item). Every container that renders these must hold
the longer Korean length.

Two strings the table names in Table 2 but omits from Table 3 are marked
below. They are needed to build the screens the table requires, so they are
placeholders in the same voice rather than inventions of a different one.
"""

from __future__ import annotations

from typing import Any

STRINGS: dict[str, Any] = {
    "app_title": "Caption session",
    "phase_rail": ["1 watch", "2 author", "3 watch again", "4 check"],
    "criterion": "Is this the caption you'd want for this moment?",
    "actions": {
        "begin": "Begin",
        # Table 2 gives the author intro a "Start"; Table 3's list of primary
        # actions does not repeat it. Kept as the table's own word.
        "start": "Start",
        "show": "Show caption",
        "keep": "Keep this caption",
        "revise": "Revise this caption",
        "finish": "Finish viewing",
        "submit": "Submit",
        "download": "Download session log",
    },
    "helpers": {
        "replay": "Replay shot",
        "pause": "Pause loop",
        "adjust": "Adjust, then show the caption",
    },
    "busy": {"writing": "Writing…", "preparing": "Preparing comparison…"},
    "check_heading": "Which caption would you rather have?",
    "probe": (
        "Across the clip, what, if anything, would you still change about the "
        "captions that the controls didn't let you change?"
    ),
    "eyebrows": {
        "first_viewing": "First viewing · automatic captions",
        "second_viewing": "Second viewing · your captions · revise any shot as it plays",
    },
    "error": "Caption request failed. Check the connection and try again.",
    # §4.2: poles set as small mono text, no numerals anywhere.
    "poles": {"eye": "EYE", "ear": "EAR"},
    # §4.3: four equal detents. The third reads "scene-level" in the prose of
    # §4.3 and §7; the id stays "scene".
    "grains": {
        "itemized": "Itemized",
        "grouped": "Grouped",
        "scene": "Scene-level",
        "atmospheric": "Atmospheric",
    },
    # Not in Table 3: the page needs a line when it is opened without an
    # identifier, since the session is "gated by a participant identifier" (§7)
    # and a blank screen states nothing.
    "participant_missing": "Open this page with a participant identifier.",
}

# Instruction-card copy. §3.2 requires the cards; the table does not write
# them, so these are placeholders in the interface's voice — statements of what
# happens next, no encouragement, no apology.
CARDS: dict[str, dict[str, Any]] = {
    "intro": {
        "heading": "Caption session",
        "body": [
            "You will watch a short street clip, shape the captions it carries, "
            "watch it again with your captions on it, and answer two short checks.",
            "There are no right answers. Work at your own pace.",
        ],
    },
    "author": {
        "heading": "Shaping the caption",
        "body": [
            "Each shot carries the sounds its caption could mention. Switch one off "
            "to leave it out, hold one to hear it alone, slide the balance to lean the "
            "list toward what is seen or what is heard, and set how much detail the "
            "caption goes into.",
            "Show the caption to read what those settings write, and keep it when it "
            "is the one you want for that moment.",
        ],
    },
    "check": {
        "heading": "Two captions",
        "body": [
            "For each shot, two captions. One is yours. Choose the one you would rather have.",
        ],
    },
}
