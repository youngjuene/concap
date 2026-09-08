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
Legacy A/B operation remains separate rather than changing old session meaning.

## Verification evidence

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

Questionnaire/UI text is English and draft; caption language can be English or
Korean. Questionnaire validation/translation remains a study preparation task.
This is a single-process application with one supervised model worker; a file lock
enforces output-directory ownership. Existing deployment services were not replaced.
