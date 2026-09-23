# Participant-test readiness

The subsequent real-video activation and QA are recorded in
[REAL_VIDEO_QA.md](REAL_VIDEO_QA.md). The synthetic preparation record below is
retained as historical evidence.

Work started 2026-09-09. Scope: finish software and reproducible QA while the
researcher prepares the three real five-minute videos. Synthetic fixtures must
remain separate from live study output and must never qualify research stimuli.

## Work and acceptance checks

1. Preserve unsubmitted legacy survey, visual and sound reports across reload and
   failed uploads, scoped to participant, configuration, language and step. Restore
   only valid draft input; acknowledged server state remains authoritative.
2. Version localized survey anchors and wording provenance without changing the
   meaning or identity of previously configured sessions. Keep draft measurement
   wording clearly distinguished from validated research instruments.
3. Exercise the integrated English and Korean journey through all three synthetic
   five-minute videos and the real final-survey API. Verify recovery, controls,
   caption/exposure integrity, keyboard access and responsive layouts. Retain an
   executable harness and isolated results; do not substitute mocked completion.
4. Qualify caption-control contracts and record reproducible model/cache timing on
   existing calibration audio. Provide a reusable evaluator for the final media;
   do not infer grounding or independent-axis quality from software assertions.
5. Run the Regen regression suite, lint, formatting, strict types and JavaScript
   checks. Review changes independently and resolve reproduced defects.
6. Record completed checks and the precise remaining final-media and human-review
   limits, including browser/device coverage and questionnaire provenance.

Baseline: clean worktree at `5c3eac3`; 412 Regen tests passed, Ruff and strict mypy
passed, both JavaScript files passed syntax checks. The existing live manifest
contains three pending long-video entries. Those entries and live participant
records are outside the write scope of this work.

## Results

Software preparation and available automated QA are complete. Both English and
Korean integrated browser journeys reached their final receipts after real-time
playback of all three synthetic five-minute videos.

### Implemented and verified changes

- Legacy drafts survive reload and failed uploads in the same tab, scoped to the
  study/configuration, participant, language and active step. Completed steps clear
  their drafts; restored input is validated before display.
- Page changes focus the heading. Mark removal returns focus to the image; close
  marks no longer cover each other's remove buttons. The existing theme and point
  coordinates are preserved.
- Both study players pause after hidden-tab or connection interruptions and replay
  the last unacknowledged interval with a 250 ms overlap. Play/pause/visibility/end
  events always enter the request queue; only periodic reports may be coalesced.
  Initial metadata records early playback, avoiding a missing first interval.
- Requests have a five-second timeout. Completion waits for acknowledged final
  playback and exposures; telemetry failure preserves the resume message and the
  disabled completion control. Ordinary failed submissions remain retryable.
- Fullscreen contains the existing completion actions and an explicit exit toggle.
  Final completion shows a stable, non-secret session receipt.
- Localized scale anchors have a versioned, hashed configuration representation.
  Old artifacts keep their original identity. New pilot preparation preserves
  rating range, refuses unreviewed custom-anchor translation and refuses overwrite.
- New v2 sessions freeze questionnaire text, anchors, choice labels and pilot
  provenance. Final answers retain that exact snapshot and its hash. Legacy
  unversioned sessions are identified honestly rather than relabelled.
- Caption instructions separate the acoustic budget from source/scene detail and
  explicitly obtain acoustic qualities from the attached audio. The evaluator
  preserves absolute audio-window bounds and records source hashes, output-equality
  diagnostics, cache behavior and a reviewer worksheet.

### Verification record

Commands run from `/mnt/hdd/research/2026/concap` with bytecode and pytest caches
disabled where applicable. Browser/model runs used existing installed tools and
isolated `/tmp` outputs; no new packages were installed.

| Check | Result | Evidence |
| --- | --- | --- |
| Complete Regen Python regression suite | 439 passed; one existing Starlette/httpx deprecation warning | `pytest tests/regen src/dpo/regen/tests -p no:cacheprovider -o addopts='' -q` |
| Strict types | Passed, 31 Python source files | `mypy --cache-dir /tmp/regen-readiness-mypy src/dpo/regen` |
| Ruff lint and formatting | Passed, 31 Python files formatted | `ruff check/format --check --no-cache src/dpo/regen` |
| JavaScript and whitespace | Syntax and `git diff --check` passed | Product JS and new browser harnesses |
| Interruption recovery | 11 checks passed, zero JavaScript errors; real short-clip/API recovery, manual and automatic end failures, continuous coverage and native fullscreen actions | `/tmp/regen-recovery-qa/result.json` |
| Accessibility/responsive smoke | Zero errors/findings; Chromium keyboard, names, focus, pointer marks, contrast samples, reduced motion, native fullscreen and 320/390/desktop layouts | `/tmp/regen-accessibility-qa/browser-accessibility-result.json` |
| Full integrated EN/KO browser journey | Both passed through all six calibration pages, three real-time five-minute fixtures and actual final-survey submission/receipt; zero reported product failures | `/tmp/regen-readiness-qa-final5/browser-readiness-result.json` |
| Independent export reconciliation | Passed: every caption cue represented, all three coverage gates met, one final-survey event, valid frozen profile/instrument hashes | `/tmp/regen-readiness-qa-final5/record-verification.json` |
| Real-model calibration-audio probe | Four control-corner outputs in EN/KO; no fallback in recorded samples | `/tmp/regen-caption-qualification/final-bangkok034-*-declared-after-quality-priority-windowfix-20260909.json` |

The final model samples used the existing `bangkok_034` calibration audio and its
staged caption as declared evidence. The two control-pair outputs differed at
each fixed setting in both languages after the prompt revision. This is a useful
diagnostic, not a validated semantic-independence or grounding score. Repeated
calls measured cold wall times of 16,779 ms (EN) and 16,611 ms (KO), with warm
medians of 1,191 ms and 1,095 ms respectively. These small probes are not a capacity
or production latency benchmark. The isolated model used an idle GPU and exited;
the pre-existing workload and live study process were not replaced.

The final timed browser run completed at 07:53 UTC on September 9. Each language
recorded 181 caption exposures spanning all 180 configured cue windows. Viewing
start to receipt took 903.144 seconds (EN) and 903.237 seconds (KO); every video
recorded at least 299,990 ms of played coverage. The exports retain two legacy
viewings, 8- and 22-answer calibration surveys, one frozen calibration observation,
and one final viewing questionnaire. Viewing captions in this synthetic rehearsal
are explicitly authored fallbacks; the separate Gemma probe supplies model evidence.

Independent code review approved the final recovery/fullscreen changes after the
reported defects were fixed and retested. The final visual comparison passed at
94/100 for theme/layout consistency. The temporary QA server was stopped after
the record audit; the original localhost 8779 study still returned HTTP 200.

### Reproduction and final-media handoff

The new pilot calibration document is prepared at
`/tmp/regen-readiness-pilot-20260909/regen.json`, with session
`street-regen-localized-scale-v1` and configuration hash `b28b2caaa070`.
Its source, `data/live/regen.json`, and the three pending entries in
`data/live/viewing-study.json` remain unchanged. See [STUDY.md](STUDY.md) for the
preparation command and the fresh output-directory requirement.

For the full timed rehearsal, start `tests/preview_readiness.py` from the repo
root using the existing Python environment, then run `tests/browser_readiness.mjs`
with the installed Playwright/Chromium paths shown in `STUDY.md`. The normal mode
plays each five-minute fixture in real time. `REGEN_READINESS_FAST=1` is a separate
diagnostic mode and must never be reported as full playback evidence.

The remaining media-dependent work is to validate the researcher's three completed
videos, aligned WAV audio, contiguous cue evidence and EN/KO fallbacks; run the
same full journey and the caption evaluator against those final assets. The
evaluator's reviewer sheet is
`/tmp/regen-caption-qualification/final-window-reviewer-sheet-20260909.csv`.

### Limits that software QA cannot establish

- Question wording remains pilot/placeholder material and Korean translations
  remain draft. No approved replacement instrument was supplied in this session.
  This work does not claim scientific validation of the questionnaire or turn the
  authored experience questions into a validated ART/PRS/PRSS measure.
- Automated browser coverage is Chromium. Firefox/WebKit executables were absent;
  physical-device and assistive-technology user testing was not performed.
- Synthetic fixtures and the existing short-audio probes do not qualify the final
  videos, long-session model quality, participant comprehension or deployment load.
- Original running studies retain their original configuration/anchors. Use the
  prepared version with fresh outputs for the localized pilot; do not mix it into
  existing research records.

`QA_REPORT.md` remains the historical September 8 record. Earlier failed and
superseded `/tmp` artifacts are retained for diagnosis; use the final artifacts
named here when assessing the completed checks.

### Changed files

- Interface/recovery: `regen.js`, `regen.css`, `study.js`, `study.css`, `study-ko.json`.
- Instrument/personalization: `config.py`, `app.py`, `study_instrument.py`,
  `continuation.py`, `study_api.py`, `study_store.py`, `study_schema.py`.
- Regression and QA: `tests/test_instrument_localization.py`,
  `tests/test_study_instrument_records.py`, `tests/test_caption_qualification.py`,
  `tests/qualify_captions.py`, `tests/preview_readiness.py`,
  `tests/browser_readiness.mjs`, `tests/browser_recovery.mjs`,
  `tests/browser_accessibility.mjs`.
- Documentation: `READINESS.md`, `STUDY.md`, `IMPLEMENTATION.md`, `QA_REPORT.md`.

The implementation reuses the existing state authority, server gates, request
queue, controls and identity styles. It adds no application dependency or parallel
storage system. No commit or deployment switch was made by this readiness pass.
