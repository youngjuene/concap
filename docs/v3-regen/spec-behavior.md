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

**Display.** Separated source waveform lanes over a ten-second timeline,
colour-coded and labelled, each with its own playback control.

**Behaviour.** Playing a lane plays that stem alone, one lane at a time —
starting another stops the previous. The playhead is shown on the waveform
during playback. Stem levels are normalised against the original mix.
Selection is a control separate from playback, and their hit areas must not
overlap. Several lanes may be selected, and selected lanes are highlighted.
Submit goes to the regeneration waiting screen.

**Logging.** Selected source identifiers with their selection order, per-lane
playback count and total listening time, and whether a selection was made
without the lane ever having been played.

## §6 Regeneration stage

Runs immediately on the §5 submit.

**Inputs.** The matched object labels from §4 with the unclassified points
excluded, the selected source labels from §5, and the video and audio of the
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
