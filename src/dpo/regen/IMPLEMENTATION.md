# Implementation handoff

Implemented 2026-09-08 against the two-stage clarification and `DESIGN.md`.

## Delivered

- `study_schema.py`: typed survey validation, staged-media validation/hashes,
  multi-clip observation aggregation, immutable personalized prompt and axis policy.
- `study_store.py`: SQLite authority, conditional transitions, idempotent receipts,
  separate exposure records, content cache and job delivery identities.
- `study_worker.py`: bounded WAV excerpts with atomic cache writes and a supervised
  model process. Queued work is bounded, stale deliveries are rejected, and timeout
  or missing backend yields the current window's authored fallback.
- `study_api.py`: separate v2 application, calibration playback/observation gates,
  preparation handoff, three-video playback, settings, exposure and one final survey.
  A study-specific HttpOnly cookie avoids collisions between previews and studies.
- `study.html`, `study.css`, `study.js`: coherent existing theme, calibration forms,
  accessible marking, two-axis pad/sliders/presets, persistent drafts/outbox, playback
  checkpoints, responsive watching and mixed-type final survey.
- `study_prepare.py`: convert one or more staged legacy documents to calibration
  manifests with three explicitly pending viewing videos; validate actual media.
- `app.py`, `regeneration.py`: targeted legacy session-write serialization and
  participant-language propagation to the actual model prompt.
- `STUDY.md`: configuration, preparation, launch and verification instructions.

## Review gaps closed

Mixed survey types have individual validation. Playback reports position/sequence
and seek epochs independently of controls. Viewing start binds the frozen profile
to a validated three-video manifest transactionally. Model audio uses excerpt-relative
time while cues/exposures use absolute video time. No prepared media is invented.

The implementation shares identity tokens and caption/point/model primitives. It
does not introduce a frontend framework, queue service, new package dependency or
second state authority. Exposure records are not copied into every session snapshot.
Legacy A/B operation stays available; the optional page-six continuation is described below.

## Verification evidence

- Mainline integration: default repository suite passed with 1,154 tests and
  three opt-in live-model tests skipped. All 157 configured source files passed
  strict mypy; Ruff lint and formatting passed for 238 files; the lockfile check passed.
- Cold/warm offline canary produced the same artifact, the warm run used its cache
  with zero provider calls, and all 84 generated artifacts passed verification.
- Root pytest discovery now includes the new study tests. The root README identifies
  both application entry points and keeps the public legacy deployment distinct.
- Branch-separation verification: all 389 regen tests passed after moving the
  language regression into its independently cherry-pickable fix commit.
- Combined regression run: 388 passed (372 legacy + then-current 16 new tests).
- Final focused run: 17 passed, including the added same-host cookie-isolation test.
- Atomic audio-excerpt and worker-timeout tests rerun after the final worker change: passed.
- Strict mypy: all 22 Python files under regen, including new tests, passed.
- Ruff, JavaScript syntax and diff whitespace checks: passed.
- Browser: three calibration clips; mouse and keyboard axis changes; reload at
  121 seconds; no mobile horizontal overflow or JavaScript errors. Final survey
  mixed-input form tested with mocked transport; full three-video completion and
  final-survey eligibility tested against the actual API with a controlled clock.
- Direct visual comparison: 93/100 for shared-theme coherence, with screenshots
  and reproduction script referenced in `tests/visual-review.json` and `STUDY.md`.

One existing Starlette/httpx deprecation warning remains. No packages were installed.
Independent OMX review was unavailable (`unsupported_documented_leader_proof`);
this handoff does not claim independent architecture approval.

## Remaining external qualification

The three real five-minute videos and their aligned evidence/fallback cues still
need preparation. Gemma integration uses the existing adapter but real-model latency,
grounding, independent-axis behavior and caption quality were not measured here.
The no-backend mode is an explicitly recorded authored-fallback rehearsal.

Questionnaire wording and EN/KO translations are draft and require study review.
This is a single-process application with one supervised model worker; a file lock
enforces output-directory ownership. Existing deployment services were not replaced.


## Page-six continuation follow-up — 2026-09-08

`dpo regen serve --viewing-manifest … --viewing-media … --viewing-out …` now mounts
the long-video study and automatically hands over after page 6. The frozen profile
carries actual validated calibration observations and retains both legacy survey
records. Neutral axes are explicit because the legacy questionnaires do not ask
for detail preferences. No new calibration is collected at the handoff.

Integration adds participant-scoped capability/cookies, idempotent profile import,
EN/KO continuation UI, a local researcher recovery route for old tabs, authoritative
snapshot points, and one shared inference process for calibration and viewing.
Independent review identified the duplicate-GPU-model, old-tab recovery and
telemetry provenance issues; all were fixed and the fixes re-reviewed.

Follow-up verification: 412 regen tests pass; Ruff, strict mypy (26 source files),
JavaScript syntax and diff checks pass. Native Mac Chrome verified page-six
submission, automatic handoff, the two-axis player, pointer/keyboard controls,
fullscreen with controls, and persisted settings after reload. The UI run used
three explicitly synthetic five-minute fixtures and isolated output directories;
it does not qualify the actual 15-minute research experience. A real Gemma smoke
on the existing calibration audio served a calibration-style prompt and viewing
prompt from the same GPU worker, with non-fallback output for both. Actual long
videos, grounded cues, independent-axis quality and latency still require testing
on the final stimuli. See `STUDY.md` for deployment and old-session recovery.
