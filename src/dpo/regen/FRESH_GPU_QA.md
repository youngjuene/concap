# Fresh GPU launch and real-media verification — 2026-09-12

The updated live study was restarted at **13:06:12 UTC**, on
`http://127.0.0.1:8779`, using **GPU 1** and the existing pinned Gemma E4B base
model, backend, contract, access gate, media paths, and response directories.
The new process is PID `2976885`; it replaced PID `1731471` gracefully.

## Fresh test result

English and Korean completed the entire revised flow against an isolated,
freshly launched GPU service using the real study media. Both reached `done` at
approximately **13:04:30 UTC**. Video completion, playback timing, API responses,
caption generation, and survey submissions were real; no completion flags or
playback coverage were seeded. Prepared video delivery copies were reused, while
the inference worker and QA caption storage started fresh. Final successful
journeys followed the initial cold-load checks and used the same worker.

Each language recorded:

- Four short-clip viewings: original and updated captions on the same clip,
  repeated for the second clip.
- Short-chapter response sizes `8 / 14 / 8 / 14 / 8`, including one overall PRSS.
- Three five-minute videos and three four-answer per-video surveys.
- One final overall questionnaire, including the configured eight PRSS items.
- Successful reload/resume, presets/reset, keyboard adjustment, drag controls,
  fullscreen/exit, final draft restoration, and completion receipt.

| Measure | English | Korean |
| --- | ---: | ---: |
| Successful long-video generation jobs | 184 | 181 |
| Jobs using a validation fallback | 0 | 5 |
| Generated caption exposures | 177 / 181 | 172 / 181 |
| Median successful generation time | 812 ms | 764 ms |
| Long-video coverage, milliseconds | 299679 / 300259 / 300136.589 | 299679 / 300259 / 300129.247 |

Together, **349 of 362 exposures (96.4%)** displayed generated captions. Prepared
fallbacks also cover moments when a generated caption is not ready at its cue
boundary. The five failed jobs were rejected by the existing language/length
validator; no GPU failure or missing-model fallback was recorded. All four
short-clip regenerations for the completed sessions used Gemma without fallback.
The initial corrected-prompt cold request succeeded in about 16.7 seconds.

An independent record audit passed: caption text, job identity, settings,
video hashes, audio windows, played coverage, survey counts, and frozen
instrument/profile hashes reconcile. This is an integration check, not a
caption-quality or participant-capacity benchmark.

## Changes and activation checks

- `regeneration.py`: the new flow carries its version in the actual model
  request and cache key, so the prompt correctly refers to the same clip.
  Versionless sessions retain the different-segment wording.
- `tests/test_legacy_language.py` and `tests/test_survey_hierarchy.py`: regression
  coverage for dispatched/logged prompt agreement and cache separation.
- `tests/browser_real_study.mjs`: real playback harness updated for repeated
  short clips, phase-end PRSS, per-video questionnaires, and current exports.
- `STUDY.md`: current prompt behavior and verification count.

The old three-entry generated-caption cache was retained in its original study
directory as `captions.prompt-940bafea4a12.20260912.json`, because the corrected
prompt has a new writer identity. Existing participant responses, snapshots,
viewing records, and database contents were verified unchanged by aggregate
fingerprints. QA records remain separate under `/tmp`.

The restarted service returns 200 for `/`, `/api/strings`, and `/regen.js`,
exposes the updated hierarchy in English and Korean, returns 404 for the QA-only
source route, and refuses unauthenticated viewing-state access. It retains lazy
model loading: the isolated GPU test worker was stopped before live activation,
and the live worker loads on the next generation request. GPU 0 was untouched.

Regression verification: **484 tests passed**; Ruff lint/format, strict mypy,
JavaScript syntax, and whitespace checks passed.

## Evidence

Final artifacts are under `/tmp/regen-fresh-gpu-20260912/`:

- `browser-final-pass/browser-real-result.json` and `{en,ko}-export.json`
- `browser-final-pass/{en,ko}-done.png`
- `audit-results.json` — status `pass`
- `activation.json` and `live-verification.json`
- `record-fingerprints-before.json` and `record-fingerprints-after.json`

The private launch record is retained with restrictive permissions for operating
the service; it is not part of the report. Earlier interrupted harness outputs
are not the final verification evidence.
