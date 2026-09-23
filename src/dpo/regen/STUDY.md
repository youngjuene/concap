# Two-stage caption study

Current software/QA readiness is recorded in [READINESS.md](READINESS.md).
The three supplied real videos are now connected; deployment and real-media test
evidence are in [REAL_VIDEO_QA.md](REAL_VIDEO_QA.md).

## Current protocol and configurable counts (2026-09-22)

New browser enrollments negotiate `client_protocol=3`. Their short-clip order,
study document digest and item digest are frozen; playback coverage is required
before each short-clip questionnaire. Existing saved sessions keep their assigned
protocol and profiles. API clients omitting the negotiation field retain the
historical v3-document behavior; v4 documents always create protocol-3 sessions.

For variable short-clip counts, use `schema: "dpo.caption-regen/v4"`, a nonempty
`segments` mapping with unique segment/clip IDs, and `clip_order` listing every
segment ID exactly once. The scaffold command supports this through
`--clips clip-one clip-two clip-three --segment-ids first second third`; media
subdirectories use the supplied segment IDs. Omitting `--segment-ids` retains
the historical two-clip A/B scaffold. Every segment retains the existing prepared media,
frames, masks, audio, and caption-track fields. For example, three segments named
`first`, `second`, `third` use `clip_order: ["first", "second", "third"]`.
Enrollment sequence rotates this order; original captions always precede updated
captions for the same clip. This balances clip positions, not every carryover
pair. Existing v3 documents continue to require A/B and preserve their parity order.

The viewing manifest accepts any positive count of calibration clips and long
videos. `--stage-viewing --sources` accepts one source per configured viewing
entry, and cue IDs must match those entries exactly. Preparation still creates
three pending long videos by default; edit that list before staging a different
count. The existing approximately five-minute long-video duration constraint
remains. Counts are frozen when viewing starts; changing the manifest cannot
change an active viewing session's assigned videos.

Both phases distinguish per-viewing/per-video answers from one overall survey
per chapter. Phase 1 ratings, including caption questions, use the configured
scale; Phase 2 caption ratings use five points, and its PRSS imports the configured
scale. Questionnaire wording and the personalization scoring policy are unchanged.

Protocol-3 Phase 1 interactions carry server-issued scope, stable `event_id`,
phase/stage/clip/viewing identity, condition and occurrence time. A persistent
browser outbox retries until acknowledged; server receipt is deduplicated across
restarts. The handoff waits for outstanding uploads and offers retry on failure.
If browser storage is unavailable, an explicit warning explains that pending
interactions require the tab to stay open. Historical unscoped events remain
unknown in exports. Playback coverage is browser-reported, not proof of attention.

Phase 2 persists video drafts as `{video_id, answers}` and renders frozen group
headings. Accepted settings events include `settings_revision` and control origin
(slider, pad, preset or reset). New sessions suppress unchanged settings revisions
and generation scheduling; queued superseded controls are coalesced. Exposures
still identify actual displayed captions, independently of selected values.

New participants in the integrated `dpo regen serve` interface complete two chapters:

1. For each configured short clip, watch its prepared captions, answer ART and
   visual/sound observations, watch the **same clip** with updated captions, and
   answer ART plus caption questions. Repeat for the remaining clips in the
   participant's assigned order. Ask PRSS once after all short clips.
2. Watch each configured five-minute video with acoustic-detail and
   source/scene-detail controls, then answer four questions about that video.
   After all videos, ask PRSS and the overall controls/effort questions once.

This separates responses about a particular viewing from responses about a
whole chapter. The longer videos play once each; caption adjustments remain
available during playback. A phase already in progress retains its assigned
flow. A saved continuation link that has not started the longer-video phase
receives the current instrument and matching transition rules when calibration
is handed over. The standalone v2 entry still uses its observation-only short calibration,
then joins the same new long-video hierarchy for new sessions.

## Continue after the short-clip chapter

Add these options to the existing `dpo regen serve` command; retain its writer,
backend, contract, public access-code gate, media and calibration output options:

```sh
--viewing-manifest /path/study.json --viewing-media /path/media --viewing-out /path/viewing-output
```

Supply all three together. Overall PRSS submission saves its answers and automatically
opens `/viewing/<session-id>/`. The participant starts the long-video phase there,
then watches three five-minute videos, uses the acoustic and source/scene detail
axes, answers the per-video questions, and completes the overall experience
survey. Pending media preserves calibration and
reports that the videos are not ready; it never reports the entire study finished.

The link imports the completed visual/sound observations for every short clip and
keeps all viewing records and questionnaire responses in `calibration_source`. It does
not repeat the standalone preference form or short calibration clips. The legacy
ART/PRSS responses are not direct detail-preference ratings, so both axes still
start at 50%. New handoffs now compile the Phase 1 ART, caption, and PRSS ratings
into a frozen questionnaire profile and include it in the Phase 2 system prompt.
Credibility and distraction responses guide source caution and conciseness;
paired ART and overall PRSS provide restorative-experience context. These are
versioned writing heuristics, not validated preference estimates. Live controls
override inferred style guidance. The prompt also retains visual/sound observations.
See [PERSONALIZATION.md](PERSONALIZATION.md) for scoring, custom-item behavior,
provenance, and verification. The
older protocol continues to supply its original single visual/sound observation.

EN/KO is retained throughout the handoff. The new viewing UI and final survey have
draft Korean translations. Declare `"languages": ["en", "ko"]` in a bilingual
manifest and provide cue fallbacks in both languages. Study questionnaire wording
and translations still need study review before recruitment.

New short-clip records carry `flow_version=clip-caption-prss-v2`, `clip_index`,
`clip_count`, and a scope (`clip_original`, `clip_updated`, or `overall`). Overall
PRSS is not attached to an individual viewing. New viewing sessions use
`flow_version=2` and a frozen `dpo.caption-study-instrument/v2` instrument;
`video_surveys` bind each answer set to its video, instrument, profile, and media.
The final instrument imports the configured PRSS items and response scale from
the short-clip chapter. Standalone sessions use the packaged pilot items.
Phase 2 caption questions use five points; Phase 1 caption questions and both
chapters’ PRSS retain the configured Phase 1 scale, normally seven points. Per-video drafts and submissions include `video_id`.
Short-clip generation carries the flow version into the model request and cache
key. The new prompt explicitly refers to the same clip; older sessions retain
their original different-segment prompt.

PRSS wording is preserved exactly, including authored wording and provenance.
The interface explicitly asks participants to consider the whole chapter. This
change in reference period is a research-protocol change, not validation of the
packaged placeholder scale or its translations.

When the legacy writer is Gemma, both phases use one shared, serialized, killable
inference process with the same backend/checkpoint and GPU assignment. The legacy
caption writer retains its existing prompt, cache identity and budget handling.
Viewing jobs and legacy calls have bounded queue/decode waits; a watchdog failure
restarts the shared worker. This avoids holding two model copies on one GPU.

New calibration tabs receive a random continuation capability in session storage.
Each viewing session uses an HttpOnly cookie scoped to its own participant URL.
Resuming by participant label alone cannot access the viewing profile. Tabs opened
before this upgrade need researcher-assisted recovery: from the study machine,
POST `{"participant":"<verified-participant>"}` to
`http://127.0.0.1:8779/api/continuation/recover` with JSON content type and
`x-study-request: 1`. After verifying the participant, privately give them the
returned relative URL on the study origin. Its fragment contains a session secret;
do not publish it or place it in shared logs. The browser validates it, saves it
in that tab, and removes the fragment. Recovery is rejected from remote clients.
Old visual points absent from server snapshots are marked unavailable; client
telemetry is never used as authoritative calibration evidence.

## Branch boundary

Development lives on `regen-two-stage-study`, based on `off-campus-sessions` at
`72db0cb5a617ace6c171b8479f97a7940d69b5fb`. Compare this branch against
`off-campus-sessions` when reviewing the new feature; comparing against `main`
also includes the inherited off-campus deployment work. The language-prompt and
legacy session-locking fixes are separate commits from the v2 study feature.
No deployment files or existing service configuration are changed by this feature.

## Prepare and run

Run from the repository with its existing virtual environment. To reuse staged
calibration assets (all paths must resolve under the same media root):

```sh
.venv/bin/python -m dpo.regen.study_prepare --legacy /path/regen.json --output /path/study.json
.venv/bin/python -m dpo.regen.study_prepare --validate /path/study.json --media /path/media
.venv/bin/python -m dpo.regen.study_api --manifest /path/study.json --media /path/media --out /path/study-output
```

The new app defaults to `http://127.0.0.1:8780`. It does not replace or restart an
existing server. Multiple legacy documents may be supplied to `--legacy`; repeated
clip IDs are deduplicated. The manifest chooses the standalone language;
the integrated flow preserves the calibration language. Draft questionnaire wording needs study
review before recruitment. No new packages are required; FFmpeg/ffprobe are used
for the media fixtures and validation, as elsewhere in the project.

The generated manifest intentionally leaves all three long videos `pending`.
Calibration works independently. Its saved profile remains resumable until all
three videos are ready. Update the manifest at that point; the server rereads it
when the participant starts viewing. Changes to calibration require a new study
output/session. Changes to long media after viewing starts are rejected.

## Long-video entries

Replace each pending entry with its actual staged media and complete cue list:

```json
{
  "id": "viewing-1",
  "title": "Video 1",
  "status": "ready",
  "planned_duration_ms": 300000,
  "duration_ms": 300000,
  "video": "viewing/one.mp4",
  "audio": "viewing/one.wav",
  "cues": [
    {
      "start_ms": 0,
      "end_ms": 5000,
      "evidence": "Authored, verified description of audible events and relevant visible sources in this window.",
      "fallback": {"en": "An authored sound caption for this exact window."}
    }
  ]
}
```

The example shows only the first cue: extend coverage to the full duration.
Cues must be contiguous, non-overlapping integer millisecond intervals, at most
30 seconds each. Supply a nonempty fallback in every declared caption language,
at most 160 characters. Stage uncompressed WAV audio aligned to video time zero.
Video and WAV durations are checked against the declared five minutes (±1 second).
Content hashes bind both media and cues to the viewing session and caption cache.
The model receives a cropped WAV whose local time begins at zero; logs and playback
retain absolute video timestamps. Visual context is authored evidence in this
version, not automatic visual inference.

## Generation and storage

Without a backend, the app explicitly records `model_not_configured` and uses
authored fallbacks. This mode tests the workflow, not personalized model quality.
To enable the existing pinned Gemma adapter:

```sh
.venv/bin/python -m dpo.regen.study_api --manifest /path/study.json --media /path/media --out /path/study-output --backend-config /path/backend.toml --contract /path/contract.toml --checkpoint /path/adapter
```

Use the same valid audio backend/contract as the existing regen writer. Model
weights, CUDA capacity and production latency are external prerequisites.
`--checkpoint` is optional. The supervised process reuses its loaded model,
queues bounded current/next-window jobs and is terminated on the configured
`--inference-timeout` (default 30 seconds). Missing or late results use each cue's
own fallback. Real-model latency and grounding must be measured on the actual
three videos before treating the study as ready for recruitment.

SQLite is the single authority for new sessions, jobs, cache, survey responses,
exposures and request receipts. Use a dedicated output directory. A lifetime file
lock permits one application process per output; GPU workers are spawned safely.
Session credentials are HttpOnly same-site cookies. Use `--access-code` for a
supervised shared link; a public deployment still needs the existing site's TLS
and traffic controls. The app defaults to loopback.

Cookie names are scoped to the output directory so two studies on the same host
do not overwrite one another's sessions, even when served on different ports.

Caption content cache keys exclude delivery revisions/epochs. Changing settings
or seeking invalidates delivery without discarding reusable identical content.
Controls have explicit 1% resolution. Caption text stays fixed until its cue ends.
The final survey cannot open until all three videos have sufficient played
coverage and their per-video answers are saved. Seeking to the end does not count
as watching.

The browser keeps draft forms and an exposure outbox keyed to its non-secret
session ID. Failed uploads retry with stable event IDs. An interrupted open cue
is marked incomplete on resume; no browser can prove unobserved time after a crash.
Exposure timestamps describe reported display, not verified gaze or attention.
New records use `timing: "display-v1"`: `start_ms` is the media position at
caption activation and `end_ms` is the sampled position when the DOM text is
replaced or playback is interrupted. `cue_end` preserves the nominal cue bound;
the displayed interval may extend beyond it until the next sampled replacement.
These records describe DOM assignment, not a measured rendered frame. An
`episode_id` separates each page load and seek, so replayed intervals must not be
treated as one monotonic viewing pass. Seeks close at the last pre-seek sample;
crash recovery retains only the last saved checkpoint and marks it incomplete.
Those unobserved tails remain unknown. Pause, visibility loss, page exit and
normal completion sample the current media position. The server validates
activation against its cue and the interval against video duration. Legacy
records without a timing version retain their original cue-bound validation.
Exposure records never grant played coverage or completion eligibility.
Researchers may read `StudyStore.export(session_cookie)` locally; session
credentials are omitted from its returned jobs. No public log-download endpoint
is exposed.

## Verification

The September 12 hierarchy is covered by `tests/test_survey_hierarchy.py` and
the updated study/continuation regressions. The combined `tests/regen` and
`src/dpo/regen/tests` suite passes **504 tests**. Ruff lint/format, strict mypy,
JavaScript syntax, and diff whitespace checks pass.

Run `node src/dpo/regen/tests/browser_survey_hierarchy.mjs` from the repository
root for the EN/KO browser journeys. It starts `tests/preview_survey_hierarchy.py`
with synthetic media, checks drafts, per-clip/overall grouping, configured scales,
responsive layouts, and submissions through the real survey APIs. Browser media
completion is simulated; a test-only loopback endpoint seeds long-video playback
coverage. Playback eligibility is tested separately by the API regressions. This
check does not establish live-media or model quality.

Implementation files: `app.py`, `progress.py`, `log.py`, and `continuation.py`
handle the short-clip order and handoff; `regen.js` and `copy.py` present its
localized hierarchy. `study_api.py`, `study_store.py`, `study_schema.py`, and
`study_instrument.py` handle scoped long-video surveys and frozen PRSS;
`study.js` and `study-ko.json` render them. Related tests, `DESIGN.md`, and
`IMPLEMENTATION.md` document the updated contract.

New v2 sessions freeze the rendered questionnaire wording, translated anchors,
choice labels and pilot provenance. The final response contains that snapshot and
its hash. Older sessions without a snapshot retain their original behavior and
are identified as `unversioned-legacy`; they are not relabelled as versioned data.

Both calibration and viewing pause after a hidden-tab or connection interruption
and replay from the last acknowledged checkpoint with a short overlap. Completion
controls wait for the final playback report (and viewing exposures) to be saved.
The legacy survey, visual marks and sound answers also have per-tab drafts scoped
to the participant, study/configuration, language and active step.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest src/dpo/regen/tests tests/regen -p no:cacheprovider
.venv/bin/ruff check --no-cache src/dpo/regen
.venv/bin/mypy --cache-dir /tmp/regen-study-mypy src/dpo/regen
node --check src/dpo/regen/study.js
```

`tests/preview_study.py` serves a localhost-only synthetic browser fixture on
18780. Its uniform video and silence are test assets, never research stimuli.
`tests/browser_study.mjs` uses an already installed Playwright module and Chromium
via `PLAYWRIGHT_MODULE` and `CHROMIUM_EXECUTABLE`; it does not install packages.
The browser test covers calibration, mouse/keyboard settings, seek/reload,
mobile overflow and the final survey's mixed inputs. Final-survey transport is
mocked in that browser check; the Python integration test covers all three real
API completion gates with a controlled clock.
`DESIGN.md` defines the shared visual theme. The actual long videos are not
supplied by this implementation.

Additional readiness checks run from the repository root:

```sh
PLAYWRIGHT_MODULE=/tmp/atlas-qa/node_modules/playwright-core/index.mjs CHROMIUM_EXECUTABLE=/home/june/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome node src/dpo/regen/tests/browser_recovery.mjs
PLAYWRIGHT_MODULE=/tmp/atlas-qa/node_modules/playwright-core/index.mjs CHROMIUM_EXECUTABLE=/home/june/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome node src/dpo/regen/tests/browser_accessibility.mjs
```

These use installed tooling and isolated synthetic fixtures. Recovery checks make
real API writes and play real short clips, inject transport/visibility failures,
and deliberately seek only to isolate the end-control failure path. Accessibility
checks use mocked stage access for some forms; neither replaces the full timed
journey. Browser binaries are supplied by the local environment, not dependencies
installed by these scripts.

## Versioned bilingual pilot configuration

Prepare an explicit new calibration version with Korean scale anchors:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m dpo.regen.study_instrument --legacy /path/regen.json --output /path/pilot/regen.json
```

Use the resulting document with `dpo regen serve --session ...` and fresh
calibration/viewing output directories. The command preserves the source document,
media references and question wording, adds a new study/session identity, preserves
the rating range, and refuses overwrites. Custom anchor wording requires its own
reviewed translation. Old studies keep their original anchors and configuration
hash; changing the running study's measurement wording in place is unsupported.

The prepared local artifact for this readiness pass is
`/tmp/regen-readiness-pilot-20260909/regen.json`. The questions remain pilot wording;
this command does not convert them into a validated ART/PRS/PRSS instrument.

## Caption qualification

The reusable evaluator records four control-corner prompts/outputs, cold and warm
inference times, fallbacks, cache behavior, source hashes and identical-output
diagnostics. An identical pair can flag an ineffective control; different strings
alone do not establish semantic independence or grounding.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m dpo.regen.tests.qualify_captions --manifest /path/study.json --media /path/media --output /tmp/caption-readiness.json --reviewer-sheet /tmp/caption-review.csv
```

For model sampling, supply `--audio`, `--backend-config`, `--contract`,
`--four-corners`, `--start-ms`, `--end-ms`, and reviewed `--evidence`/`--fallback`.
Run from the repository root so the local `regen/copy.py` cannot shadow Python's
standard-library `copy`. Choose an available GPU; do not launch a competing model
on the active study worker's GPU. Without a model backend, the report explicitly
records fallback rehearsal. Absolute audio windows are preserved; model prompts
describe time relative to the cropped excerpt.
