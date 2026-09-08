# Two-stage caption study

The standalone v2 application collects calibration across multiple short clips,
freezes a participant-specific prompt template, then presents three five-minute
videos with acoustic-detail and source/scene-detail controls. One final survey
follows all three videos. `dpo regen serve` can also continue directly from its
existing six-page calibration into this viewing phase using the flags below.

## Continue directly after page 6

Add these options to the existing `dpo regen serve` command; retain its writer,
backend, contract, public access-code gate, media and calibration output options:

```sh
--viewing-manifest /path/study.json --viewing-media /path/media --viewing-out /path/viewing-output
```

Supply all three together. Page 6 submission saves its answers and automatically
opens `/viewing/<session-id>/`. The participant starts the long-video phase there,
then watches three five-minute videos, uses the acoustic and source/scene detail
axes, and answers one experience survey. Pending media preserves calibration and
reports that the videos are not ready; it never reports the entire study finished.

The link imports the completed legacy visual/sound calibration and keeps both
viewing records and all questionnaire responses in `calibration_source`. It does
not repeat the standalone preference form or short calibration clips. The legacy
ART/PRSS responses are not detail-preference ratings: both axes start at 50%, and
`preference_source=not_collected` explicitly records this. The prompt uses the
observed visual labels and sound families; raw survey scores remain available for
analysis rather than being converted into an invented preference scale. The
legacy protocol supplies one visual/sound observation, not multiple observations.

EN/KO is retained throughout the handoff. The new viewing UI and final survey have
draft Korean translations. Declare `"languages": ["en", "ko"]` in a bilingual
manifest and provide cue fallbacks in both languages. Study questionnaire wording
and translations still need study review before recruitment.

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
coverage. Seeking to the end does not count as watching.

The browser keeps draft forms and an exposure outbox keyed to its non-secret
session ID. Failed uploads retry with stable event IDs. An interrupted open cue
is marked incomplete on resume; no browser can prove unobserved time after a crash.
Exposure timestamps describe reported display, not verified gaze or attention.
Researchers may read `StudyStore.export(session_cookie)` locally; session
credentials are omitted from its returned jobs. No public log-download endpoint
is exposed.

## Verification

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
