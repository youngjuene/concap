# Real-video survey and interaction wiring

Scope: connect the researcher's three `data/live/regen-media/viewing/viewing-*.mp4`
files to the existing calibration → interactive viewing → final survey protocol.
The original video files and existing response records must remain intact.

Acceptance checks:

1. Validate the actual video streams/durations, stage aligned PCM audio, and cover
   each video with contiguous caption windows and EN/KO fallback text. Preserve
   preparation provenance; generated captions are not labelled human-validated.
2. Validate the complete manifest before replacing the pending live manifest.
3. Run the real media and configured Gemma worker through isolated browser QA,
   including axes, fullscreen, reload, all three playback gates, the real final
   survey and exported records. No synthetic substitution or fake completion.
4. Preserve existing calibration and access configuration, then restart the live
   localhost service with verified artifacts and the current code. Retain a
   rollback record and leave the service available for interaction testing.

Initial probes: all three files are 1920×1080 H.264 with AAC stereo audio at
44.1 kHz, starting at zero. Exact durations are 299,679 ms, 300,259 ms and
300,142 ms. These meet the existing five-minute tolerance; originals need no trim.

## Live result

The real videos are active on the existing localhost service at
`http://127.0.0.1:8779/`. The original access gate and calibration configuration
remain in place. After calibration, the participant watches the three real videos,
uses the detail controls, and receives the actual final questionnaire and receipt.

The live manifest is `data/live/viewing-study.json`. All three entries are `ready`:

| Video | Duration | Windows | Aligned audio |
| --- | --- | --- | --- |
| `viewing/viewing-1.mp4` | 299,679 ms | 60 | `viewing/viewing-1.wav` |
| `viewing/viewing-2.mp4` | 300,259 ms | 60 | `viewing/viewing-2.wav` |
| `viewing/viewing-3.mp4` | 300,142 ms | 60 | `viewing/viewing-3.wav` |

Audio is mono 16 kHz PCM, aligned at time zero with exact sample counts. Caption
windows are five seconds, with short terminal remainders merged into the previous
window. The 180 windows contain 360 bilingual fallback drafts generated from their
actual audio excerpts, plus conservative sampled-frame visual context. Provenance
is explicit: assistant visual review and Gemma audio-caption drafts, not human
validated annotations. Original MP4 hashes are unchanged.

Preparation records are retained under
`data/live/regen-media/viewing/preparation-20260909/`, including cue data, generation
records, visual context, source hashes, the control probe and record verification.

## Real-media QA

English and Korean ran concurrently against one shared Gemma worker, using the
exact live calibration document and the prepared real viewing manifest. Each
played all three videos in real time. Presets/reset, keyboard adjustment, pointer
dragging, native fullscreen/exit, playback reload near 121 seconds, and real final
survey submission were exercised. QA records were isolated from live responses.

| Language | Video | Played coverage | Generated exposures | Prepared fallback exposures |
| --- | --- | --- | --- | --- |
| EN | 1 | 299,679 ms | 59 | 2 |
| EN | 2 | 300,259 ms | 59 | 1 |
| EN | 3 | 300,142 ms | 59 | 1 |
| KO | 1 | 299,679 ms | 57 | 4 |
| KO | 2 | 300,259 ms | 59 | 1 |
| KO | 3 | 300,142 ms | 59 | 1 |

Both sessions reached `done`. All 180 cue windows per language are represented;
352 of 362 exposures (97.2%) displayed generated captions. Every generated
exposure was reconciled to its job text, applied axes and exact audio window.
Each session retained one final-survey event, the two calibration viewings,
8- and 22-item calibration responses, and valid frozen profile/instrument hashes.
Recorded median generation times by video/language ranged from 699 to 1,011 ms;
these are model-job timings, not a production load or end-to-end latency guarantee.

The first reporter used the wrong export key (`spec.video` instead of
`spec.video_hash`). It failed after KO had rendered a receipt and EN had finished
all three videos. The key was corrected and EN's final form was completed through
the actual browser against the same preserved QA database; no video completion or
survey eligibility was fabricated. An independent export audit passed afterward.
EN viewing-to-receipt elapsed time includes that recovery (1,177.044 seconds); KO
took 905.394 seconds.

Primary evidence:

- `/tmp/regen-real-video-20260909/record-verification.json`
- `/tmp/regen-real-video-20260909/captured-qa/{en,ko}-export.json`
- `/tmp/regen-real-study-qa-live3/source.json`
- `/tmp/regen-real-study-qa-live3/en-done-recovered.png`
- `/tmp/regen-real-video-20260909/control-probe.json`

The bounded control probe used each video's 150–155 second window and the actual
frozen EN/KO QA profiles. It recorded all four control corners per window: 23 of
24 outputs passed generation validation; one used its prepared fallback. Some
settings produced identical text. This is inspection evidence, not a semantic
independence or human caption-quality score.

## Verification and activation

- Full Regen regression suite: **447 passed**, one existing Starlette/httpx
  deprecation warning.
- Strict mypy: **33 source files passed**. Ruff lint/formatting, browser harness
  syntax and diff whitespace checks passed.
- Staging tests include exact duration/audio alignment, invalid inputs, source
  preservation, same-duration stale audio rejection, and equivalent PCM with
  different WAV metadata. Existing WAVs are compared to a fresh source extraction;
  stale files are not overwritten.
- Independent review approved activation after verifying explicit guard checks,
  exact-manifest audit binding and rollback behavior.
- The old server (PID 3207214) was stopped gracefully and replaced by PID 955303
  with the same launch arguments/environment and port 8779. The manifest was
  replaced atomically. Existing response records were preserved.
- Live checks: `/` and `/api/strings` returned 200; test-only `/qa/source` was
  absent (404); an unowned viewing session was refused (401) without enrollment.
  The running configuration validates all three entries as ready.

Activation and rollback evidence is under `/tmp/regen-real-video-20260909/`:
`activation.json`, `live-verification.json`, `previous-viewing-study.json`, and the
private original launch record. Temporary QA/resume/model-probe processes were
closed; the live service is left running.

## Reproduce or prepare a later version

From the repository root, prepare a separate validated manifest using explicit
cue text and provenance:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m dpo.regen.study_prepare \
  --stage-viewing data/live/viewing-study.json \
  --media data/live/regen-media \
  --sources data/live/regen-media/viewing/viewing-1.mp4 data/live/regen-media/viewing/viewing-2.mp4 data/live/regen-media/viewing/viewing-3.mp4 \
  --cues data/live/regen-media/viewing/preparation-20260909/viewing-cues.json \
  --provenance reviewed-version-description \
  --output /tmp/viewing-study.next.json
```

`tests/browser_real_study.mjs` launches `tests/preview_real_study.py` with isolated
temporary outputs. Set `REGEN_REAL_BACKEND_CONFIG`, `REGEN_REAL_CONTRACT`,
`REGEN_REAL_READY_MANIFEST`, `CUDA_VISIBLE_DEVICES`, and the installed
`PLAYWRIGHT_MODULE` / `CHROMIUM_EXECUTABLE`. Use an available GPU; do not load a
second model beside the live study worker. Preserved QA roots are never deleted by
the resume wrapper. Do not replace bound media or cues during an active viewing
session; create a new version/output when changing the stimuli.

Changed code: `study_prepare.py`, `tests/test_viewing_prepare.py`,
`tests/preview_real_study.py`, and `tests/browser_real_study.mjs`. Existing app state,
worker, survey, controls and style implementations were reused. No dependencies
were added. Questionnaire wording remains the existing versioned pilot material;
caption-content review, physical-device and assistive-technology testing are not
claimed by this automated run.
