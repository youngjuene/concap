# Spreadsheet questionnaire protocol

Source: [소리자막 설문문항표](https://docs.google.com/spreadsheets/d/1JP44YNkE2hgOTHUUJ69WrseeD8_fqXHDxYZ5_cMI-5Q/edit).
All four tabs (`안내`, `문항표`, `척도라벨`, `참고문헌`) were read on 2026-10-04.
Their complete bounded cell values, tab IDs, source rows, notes and references
are frozen in `items/sheet-workbook.json`. This is a source snapshot, not a
live Google Sheets dependency. Editing the workbook does not silently change an
active participant's instrument.

## New participant flow

New tabs use `dpo.sheet-questionnaire/v1` through `sheet_questionnaire.js` and
`/api/questionnaire/*`. Existing `regen.participant` sessions retain the historical
instrument. `?protocol=legacy` is an explicit compatibility/old-QA entry, not the
new study. The new questionnaire language is Korean because approved English
wording is missing for several source-authored questions; English scale originals
and reference text remain in provenance.

P0 presents the exact source introduction and configured consent. A separately
prepared response-control practice exercise is completed once; its answers are
stored under `sheet_practice`, marked `analysis_excluded`, and never included in
response exports or the frozen research observations. Practice images and sound
choice labels must differ from the formal materials; this exercise does not
claim to be a complete video trial.

For each short clip:

1. P1: first caption viewing, with server-validated playback coverage.
2. P2 V1: one explicit point on an annotated frame.
3. P3 A1s: one most-attended sound from the configured clip-specific list.
4. P4: F1–F3, BF1–BF3, E1–E3, P1–P2 (11 ratings).
5. P5: FA1 visit familiarity choice; FA2 familiarity and FA3 preference ratings.
6. P6: the same clip with its reviewed second caption track.
7. P7 V2 and P8 A2s: repeat visual/single-sound observations.
8. P9: the same 11 restoration questions with `_2` IDs; a session-stable
   shuffled order is saved with the actual submitted page.
9. P10: S1–S3 caption ratings.
10. P11: R1 reading amount and R2 multiple-choice caption recall.

There are 30 researcher-authored survey items plus four perception responses per
clip (34 response fields, across nine submission pages). The obsolete ART8,
caption6, chapter PRSS8 and five-family boolean forms are not shown to new
participants. Ratings use **0 through 6**, including a valid zero. A and C scales
show all seven source labels. B scales preserve the supplied labels at 0/3/6;
unprovided intermediate wording is not invented. R2 adds `기억나지 않음` as an
exclusive option. All other recall options are authored per stimulus.

Following the user's explicit decision on 2026-10-04, Phase 2 has **no per-video
or final questionnaires**. Existing playback, settings, generation and displayed
caption telemetry remains. After every configured long video, P12 displays the
oral D1/D2 interview prompts; D2 is conditional on reporting a difference. The
UI records completion acknowledgements, not fabricated oral responses. There is
no microphone capture or AI transcription integration. P13 shows the configured
debrief after the interview and then produces completion confirmation.

## Researcher setup

Pass `--questionnaire-config /path/to/reviewed-config.json` to `dpo regen serve`
(or `questionnaire_config=` to `build_app`). Required material includes:

- `consent_text`, `debrief_text`: researcher-approved full text. The source
  workbook describes these documents but does not supply their complete text.
- `practice`: `{reviewed: true, frames: [{still: "practice/frame.png", at_ms: 0}],
  sound_options: [{value: "practice-a", label: "..."}]}`. Paths are beneath the
  media root; formal clip frame paths cannot be reused as practice images.
- `clips.<segment>.options`: A1s options (A2s reuses them) and R2 sound/distractor
  options, each `{value, label}`. Mark `options_status: "reviewed"` only after
  checking options against the actual second-view caption. Correct-answer fields
  may be retained as researcher metadata but are removed from participant API
  options. The intended 3–4 mentioned sounds and 1–2 distractors still require
  authored content; the application does not infer or certify those facts.
- `clips.<segment>.condition`: assigned `audiovisual_congruence` (`congruent` or
  `incongruent`) and `description_depth` (`shallow` or `deep`). Assignment is
  researcher-configured; this change does not implement a randomization service.
- `stimulus_status: "reviewed"` and `second_caption_strategy: "maintain"` or
  `"opposite"`. Optional `first_track` supplies that condition's full first-view
  track; otherwise the document's prepared track is used. All timing/length/language
  checks remain active.
- `maintain` repeats the first track exactly. `opposite` requires reviewed
  `second_tracks`, keyed as `<matched object_id>|<A1s value>` (or
  `unclassified|<A1s value>`). Each value is a complete caption track with the
  first track's cue times. Before consent, every declared object ID plus
  `unclassified`, crossed with every A1s option, must have a valid track. Missing
  combinations block P0; the application never substitutes the old
  attention-reinforcing prompt.

P0 lists missing material and blocks enrollment into the actual experiment until
setup is complete. The same option/track validators are used by preparation and
the participant pages. Transition targets are validated inside the response
transaction, so a page that cannot render cannot consume an answer or advance
the session. Editing setup before consent, including A-to-B-to-A changes, is
supported; after consent the configuration is frozen. Source notes about translation review, IRB wording,
scale limitations and interview recording are preserved and are not marked resolved
by the software. The source's opposite/maintenance study is not qualified merely
because synthetic QA can complete the questionnaire.

## Storage and analysis

The new short protocol uses the existing `StudyStore` SQLite transaction for
answer submission, page advancement and idempotent receipts together. With a
configured long phase it shares the viewing store; without one it uses
`questionnaire.sqlite3` under the short output directory. Each formal response
keeps its page, clip/view identity, first/second condition, stimulus assignment,
caption strategy, exact item definitions, displayed order/options, source hash,
typed values and server submission time. Zero, choice strings, arrays and point
records are never coerced to the historical 1–7 integer-only schema.

Registration, its persisted `registration_sequence`, frozen clip order and
enrollment event are committed together. New sessions begin at revision zero;
clients use returned revisions rather than assuming an initial number. Mutation
events receive server-verified phase/page/clip/view/condition/protocol scope
before a page or clip advances. Practice events are marked `analysis_excluded`.

The authenticated local export is `/api/questionnaire/export`. The existing local
researcher dashboard reads both `questionnaire.sqlite3` and `study.sqlite3` in its
configured directories. It deduplicates identical files and copied session
snapshots, reads in-progress `sheet_responses`, excludes practice, deduplicates
the completed handoff mirror, and preserves scale/source/order fields and the
structured factorial assignment plus caption strategy in JSON and CSV.
Those factors remain separate in response groups and filters. Multiple-choice
charts count answers containing each option; totals can exceed 100% across
options. Historical missing scope remains unknown instead of being guessed.
P12/P13 acknowledgements are process events, not
survey answers. The old questionnaire's score heuristics are not applied to the
new item IDs. New-profile observations explicitly describe single-most-attended
choices rather than pretending to measure five heard/not-heard booleans.

Viewing exposure records with `display_interval_version: 2` separate source cue
identity (`cue`, `cue_start_ms`, `cue_end_ms`) from actual displayed playback
intervals (`start_ms`, `end_ms`). A caption can remain visible after its nominal
cue end until the browser replaces it. Do not clip these intervals back to cue
boundaries in analysis: the same observed replacement position closes one
exposure and opens the next. Pause/end use the observed stopping position;
seeking never credits the jumped interval. Historical v1 records retain their
original cue-bounded interpretation and are not backfilled.

Generated exposures bind the participant's actual successful job, cue, video,
text, settings revision and applied axes. Authored fallbacks retain null axes
and a delivery reason (`job_failed`, `not_ready_at_boundary`,
`initial_caption_timeout`, or `caption_fetch_failed`), plus the job/revision when
known. A job may finish after a boundary, so an authored exposure is not proof
that its job failed. Before initial playback, a bounded caption readiness check
selects an available result and only then enables native playback controls.

## Recovery and browser ownership

Both questionnaire and viewing credentials use persistent 30-day HttpOnly,
SameSite cookies. Valid questionnaire requests renew the same credential;
handoff preserves both cookie paths on one response. Browser-process restart
does not require copying a token or enrolling a new participant. An invalid
credential remains unauthorized and cannot be recovered from a participant label.

Questionnaire drafts are scoped to participant, instrument, clip and page, not
the changing server revision. Old revision-based drafts migrate in place;
server drafts are restored when this tab has no local snapshot. A 409 checkpoint
rejection fetches current state and resumes from the accepted position using a
new request. Ambiguous response loss retains its original idempotency key.
Initial playback and resume use the same heartbeat startup path.

Phase 2 uses a per-session Web Lock before rendering or writing records. A
second tab waits until the owner closes and then explicitly retries. Page hide
finalizes the pending exposure before releasing ownership; BFCache restores
ownership before further work. HTTPS/localhost Chrome provides the required
lock API; unsupported browsers show a blocked-session explanation. Historical
durable exposure keys remain readable. An interrupted open exposure already
present in the immutable upload queue is not enqueued a second time.

If browser storage rejects a write, the current value is retained in memory and
the regular server upload continues. The UI warns to keep the tab open while
unuploaded exposure records are volatile. Offline plus unavailable storage plus
closing the tab cannot provide durable recovery; no upload success is invented.

The original prototype/legacy files remain for already-started sessions. No live
service restart, participant-data migration, dependency installation or GPU run
is implied by these source changes.
