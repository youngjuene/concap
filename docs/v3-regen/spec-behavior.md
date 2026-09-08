# The regeneration instrument — behaviour

A within-participant comparison of two sound-caption conditions over one street
scene. The participant watches one segment captioned from a **prepared** track,
reports what they saw and what they heard, and then watches the other segment
captioned by a track **regenerated** from that report. Both viewings are
measured with the same instrument, so the contrast is the caption's provenance
and nothing else.

Four pages, six steps. Sections are numbered here and cited as `§N` from the
code; every module in `dpo.regen` says so at the end of its docstring.

---

## §1 On entry (at Page 1 load)

- Issue a participant identifier.
- Fix the condition–segment assignment from the participant's sequence number:
  either (A = prepared, B = regenerated) or (A = regenerated, B = prepared).
- Store the assignment in the session. Every subsequent page refers to it
  rather than deciding again.

The two orders alternate by sequence number, so a run of participants is
balanced by construction rather than in expectation. The assignment is
derivable from the sequence number alone, which means the analysis can
reconstruct it from the log without a stored table.

## §2 Page 1 — viewing, prepared sound captions

**Display.** One start button and a headphone/volume instruction line. After
the click: fullscreen video, no playback controls.

**Assets.** One clip for the assigned segment, and that segment's prepared
caption track (cue slot count per §9.4).

**Behaviour.** Click enters fullscreen and begins playback; the caption overlay
renders against the track; playback end auto-advances to §3.

**Logging.** A viewing record — view index, condition (`prepared`), segment,
clip identifier — the full text of the displayed captions (per-cue text with
start and end times), and the playback start and end timestamps.

## §3 Page 2a — ART four sub-factors

Likert items. `Next` is disabled until every item is answered. Submit advances
to §4.

**Logging.** Per-item responses, the entry and submit timestamps, and the key
of the viewing record the responses are about.

## §4 Page 2b — visual environmental perception

**Display.** The screenshot of the five-second frame of the segment just
viewed, and the instruction to select the elements that characterise the scene.

**Behaviour.** A click on the image creates a point; several are allowed.
Points can be repositioned by dragging. A click on an existing point deletes
it, and a clear-all button removes them together. `Next` is disabled below the
minimum selection count. Mask matching runs **once**, on the submitted
coordinates.

**Logging.** Point coordinates normalised to the image in 0–1 with their
creation order, the matched object identifier and label, points outside every
mask tagged `unclassified` with their coordinates preserved, and the count and
proportion of unclassified points.

## §5 Page 2c — auditory environmental perception

**Display.** The five AudioSet top-level sound families this corpus uses —
human sounds, animal, sounds of things, music, natural sounds — in that fixed
order, each named with examples beneath it and offering two answers, *heard*
and *did not hear*. The same five for every participant and every clip,
whatever the clip contains. Above them, the clip's own audio with a single
playback control: the judgments are made against a ten-second memory, and
this is the only way to hear it again.

**Behaviour.** Every family must be answered before submit is enabled. The
two answers are separate controls rather than one checkbox, because a blank
is a participant who has not answered and is not the same as one who did not
hear. Submit goes to the regeneration waiting screen.

**Logging.** The families reported heard and the families reported not heard,
and — so a false alarm is readable without joining to the document — which
families the clip actually carries.

**Why fixed rather than per-clip.** Until 2026-09-07 this page showed one
waveform lane per separated source the clip carried, and asked which the
participant noticed. A participant could then only ever report a source that
was there: the page could not distinguish "I heard it" from "I was given the
chance to say so", and a family nobody had put in the clip could not be
claimed at all. Asking all five of everyone makes a false alarm a measure
rather than an impossibility. It also removes a dependency the study could
not meet — the lanes needed separated stem audio, which does not exist for
this corpus, so every lane read "no sound".

This changed what §5 measures and therefore what §6 is conditioned on, so it
changed `method_constants` and with them the configuration hash. **Sessions
run before this change are not comparable with sessions run after it**, and
the hash on every logged row is what says which side of the change a session
falls on.

## §6 Regeneration stage

Runs immediately on the §5 submit.

**Inputs.** The matched object labels from §4 with the unclassified points
excluded, the sound families reported heard in §5, and the video and audio of the
remaining segment.

**Behaviour.** Text is generated for the fixed cue slots only — the timings are
pre-set and the count is §9.4's config value. Output is validated per slot for
character and line limits, language, and empty output. If the latency ceiling
is exceeded the default fallback caption track is used instead. The waiting
screen shows a progress indicator and a message discouraging abandonment.

**Logging.** The full assembled prompt, the full raw model output before any
post-processing, the full final caption track, the referenced visual object
labels and auditory source labels, the generation duration, whether the
fallback triggered, and the model identifier and generation settings.

## §7 Page 3 — viewing, regenerated sound captions

**Display.** One start button; after the click, fullscreen video with no
playback controls.

**Assets.** The segment clip not used on §2, and the caption track just
generated — or the fallback.

**Behaviour.** As §2. Playback end auto-advances to §8.

**Logging.** A viewing record — view index, condition (`regenerated`), segment,
clip identifier — the full text of the displayed captions under the same schema
as §2, and the playback start and end timestamps.

## §8 Page 4 — survey

**Items.** The ART four sub-factors, the same items as §3; caption credibility
and its effect on restorativeness; and the PRSS.

**Behaviour.** Submit is disabled until every item is answered. Submit goes to
the completion screen.

**Logging.** Per-item responses, the entry and submit timestamps, and the key
of the viewing record the responses are about.

## §9 Applies throughout

1. Four pages, six steps counting §3, §4 and §5 separately, in linear
   progression. Back navigation and re-entry to a completed step are blocked.
2. Item definitions are held as data outside the code, and the ART block is
   reused verbatim across §3 and §8 — one definition referenced twice, never
   two copies that could drift.
3. Scale format, range and anchor wording are identical across the survey
   pages.
4. The caption cue slot count is a single config value, referenced by both the
   prepared track and the regeneration validation rules.
5. The storage unit is one row per viewing; survey responses join to the
   viewing record by key.
6. Captions for both conditions are stored under an identical schema, so no
   analysis can tell them apart by shape.

## §10 Assets prepared in advance

- Segment A and B clips, matched in encoding, resolution and loudness
  normalisation.
- The five-second frame screenshot for each segment.
- Object segmentation masks for each segment.
- Separated audio stems, their waveform data and their labels, for each
  segment.
- The prepared caption track.
- The default fallback caption track for regeneration.
