# The interface audit, and what is still open

A screen-by-screen audit of the participant's eight screens, made against
commit `10b6b9f` by recreating each of them at real width from `regen.html`,
`regen.css`, `identity.css` and `regen.js`. Thirty-seven findings, eight of
them blocking. All thirty-seven are applied in `1944762`, whose message says
what each one was and why the fix is shaped the way it is; the reasoning also
lives beside the code it argues for, which is where it is most use.

This file is the part that did not fit in either place: the decisions taken
where the audit deliberately did not take one, and the questions it left for
the study rather than for the interface.

## Decided

**§4 shows one moment at a time.** The audit offered a repaired strip or a
picker, and preferred the picker: the blur in the strip was being asked to say
"do not mark here" and "click me to navigate" at once. The picker is text and
the picture is a picture. This also puts §4 back inside the column every other
step uses.

**§5 keeps its lanes.** The prototype that came with the audit replaces them
with a picker of AudioSet families chosen in noticed-order. That is a different
question from the one §5 asks — free recall of separated sources, not
recognition against a taxonomy — and `spec-behavior.md` §5 requires the lanes
and their per-lane playback. It would also change `regeneration_inputs` in
`config.METHOD_CONSTANTS`, which is hashed, so no session run before it could
be compared with one run after. What shipped is the audit's other proposal:
the same question, asked with controls that say what they are.

**Reversed items are not marked on screen.** A methods call. `reverse` reaches
the page in the `/api/step` payload already, so marking them later is a
client-side change and nothing else.

**No middle scale anchor.** The prototype shows one between the two. `Scale`
holds a low and a high anchor and is hashed with the rest of the calibration
(§9.3), so a third is a calibration change rather than a design one.

## Open

**Does anyone actually submit §5 empty?** §5 could be finished in one press,
and what it records conditions the captions rated in §7 and §8 — so an empty
§5 does not thin one measure, it moves the stimulus for two more. An empty
submission now takes a second press, which costs a participant who means it one
keystroke. Whether that is enough is a question for the data: the lane counters
already record plays and `listened_ms`, so a pilot can be read for participants
who submitted with nothing selected and under two seconds on the screen. If
that number is large, §5 needs a per-lane answer rather than an optional
selection, and that is a different construct — say so in the analysis before
changing it, not after.

**Is a deaf or hard-of-hearing participant in scope?** The cues are served as a
real WebVTT track as well as drawn in the band, so the stimulus is inspectable
and the band announces. But if such a participant is in the sampling frame,
this stops being an accessibility question and becomes a design one: §5 asks
what they noticed *while watching*, and §2's manipulation is a caption over
audio they may not have. Decide it before recruitment, not during.

**Two font files.** `identity.css` names the stack that actually resolves and,
for a Korean session, names the Korean faces first — which is the half of this
that affects the measurement, since two lab machines were otherwise rendering
the Korean arm in two different typefaces at two different apparent sizes. To
fix the other half, ship one Latin and one Korean woff2 with the offline build:
that needs a binary read beside `_package_file` and an entry in `STATIC_FILES`,
as well as the files.

## What would sharpen the next pass

A screen recording of one real session — most of the audit's "major" findings
are estimates of how long something takes, and a recording settles all of them
at once. Two or three real frames and one real stem set, so §4 and §5 can be
judged against footage rather than the synthetic staging. And the event log
from any pilot run: `frame.selected`, `point.removed`, `lane.played` and
`auditory.empty_confirm_asked` already measure most of what this document
infers.
