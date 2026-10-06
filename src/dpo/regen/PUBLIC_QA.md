# Public viewing deployment verification

Scope: the public Tailscale URL must support all six calibration pages, all three
real five-minute videos, generated captions responding to both detail controls,
and one final questionnaire with a saved receipt.

## Delivery changes

- Long-video playback starts only after the whole video has downloaded. EN/KO
  progress and retry messages explain preparation. The browser retains a private
  cache for reloads; cleanup cancels pending downloads and releases blob URLs.
- Caption polling is independent of playback acknowledgements. A delayed caption
  request uses the prepared cue without rewinding video. Actual playback failures
  still pause/rewind to acknowledged progress and enforce the completion gate.
- Recovery seeks remain pending until acknowledged. Queued samples from obsolete
  seek generations are discarded, and the seek flag is resolved when a queued
  request starts so a delayed acknowledgment cannot create duplicate seeks.
- Viewing credit uses an uninterrupted wall-clock budget with bounded resume
  allowance, accommodating bunched HTTP arrivals without crediting seeks or
  fast-forward jumps. Control position hints schedule the visible caption window
  but never change credited playback.
- Public requests have a 15-second transport deadline. Long media downloads use
  their own cancellable stream rather than that API deadline.
- Web MP4 copies are around 65 MB each, compared with 217–228 MB originals. Source
  files are immutable. Audio packets/timestamps match exactly; resolution and
  nominal frame rate match; duration differs by 34–41 ms. Video is recompressed
  and variable timestamps are normalized, so the original remains the reference
  for frame-exact analysis. Delivery hashes and checks are recorded in
  `data/live/regen-viewing-responses/streamable-media/delivery-audit.json`.
- `make open` retains the viewing continuation when the live manifest exists,
  including after a restart. The live output paths and access gate are preserved.

## Reproduction

Use the installed browser/environment; no new dependencies are required:

```sh
PLAYWRIGHT_MODULE=/tmp/atlas-qa/node_modules/playwright-core/index.mjs \
CHROMIUM_EXECUTABLE=/home/june/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome \
REGEN_REAL_PUBLIC=1 REGEN_REAL_URL=https://concap.taild3b716.ts.net/ \
REGEN_REAL_CODE_FILE=data/live/access-code \
REGEN_REAL_DATABASE=/mnt/hdd/research/2026/concap/data/live/regen-viewing-responses/study.sqlite3 \
REGEN_REAL_QA_PREFIX=qa-public-UNIQUE-RUN \
REGEN_REAL_OUTPUT=/tmp/concap-public-UNIQUE-RUN REGEN_REAL_LOCALES=en \
node src/dpo/regen/tests/browser_real_study.mjs

.venv/bin/python -m dpo.regen.tests.verify_public_study \
  /tmp/concap-public-UNIQUE-RUN data/live/viewing-study.json data/live/regen-media
```

The public harness creates labelled QA records. Exclude `qa-public-*` participants
from research analysis. It never changes playback time, fabricates completion,
or substitutes generation results. Its local SQLite export is read-only and uses
only the credential issued to its own browser, never sending a database-read
credential to an endpoint.

## Verified result — 2026-09-10

- Media-delivery tests cover immutable originals, preserved audio, resolution and
  nominal cadence, concurrent cache creation, byte ranges, and responsive controls.
- Browser interruption regression passed, including a 6.5-second caption delay,
  hidden/offline recovery, continuous credited coverage, fullscreen and end gates.
- All 456 regeneration/viewing regression tests passed. Repository-wide Ruff
  checks and formatting, mypy (171 files), JavaScript syntax and shell syntax pass.
- The actual public HTTPS journey completed at 02:56 UTC in English/Chromium.
  It traversed the six legacy pages, restored the viewing session after recovery
  fixes, completed every long video, and submitted the final questionnaire once.
  Its browser profile and legitimate progress were preserved across test attempts;
  this was not an uninterrupted fresh-session performance benchmark.

| Video | Verified played coverage | Cue windows | Generated exposures | Fallback exposures |
| --- | ---: | ---: | ---: | ---: |
| 1 | 299,679 ms | 60 | 51 | 12 |
| 2 | 300,259 ms | 60 | 57 | 3 |
| 3 | 300,142 ms | 60 | 57 | 3 |

Every video has saved generated-caption exposures at both 0% and 100% detail.
Keyboard adjustment, pointer-pad steering, reset and fullscreen also passed.
Warm preset changes applied in 5.0–7.9 seconds; the first change after the model
restart took 26.4 seconds. Captions apply at cue boundaries, not instantaneously.
Reload restored the acknowledged playback position from the prepared video cache.
The final questionnaire's draft survived reload, the mobile-width layout fit,
and its submission produced the visible receipt and exactly one stored final event.

The separate export audit reconciled all 180 cue windows, generated text, axes,
job revisions, audio windows, the frozen calibration profile and final-survey
instrument hashes. Original video hashes remain unchanged.

Evidence on this machine:

- [Public browser result](/tmp/concap-public-proof-20260910/browser-real-result.json)
- [Record reconciliation](/tmp/concap-public-proof-20260910/record-verification.json)
- [Final receipt screenshot](/tmp/concap-public-proof-20260910/en-done.png)
- [Delayed/lost-seek recovery checks](/tmp/concap-recovery-complete-20260910/result.json)
- [Media audit](/mnt/hdd/research/2026/concap/data/live/regen-viewing-responses/streamable-media/delivery-audit.json)

The public probe originated on this server and traversed the public Funnel URL;
it was not a physical off-campus laptop test. This run covers English/Chromium.
Preparation took minutes and varied with the public connection. Multi-participant
capacity, other browsers, physical devices and scientific caption/questionnaire
validation are not established by this functional test.

## Files changed for this work

`deploy/edge`, `deploy/README.md`, `study.js`, `study.css`, `study-ko.json`,
`study_api.py`, `study_media.py`, and this report. Verification changes are in
`tests/browser_real_study.mjs`, `tests/browser_recovery.mjs`, `tests/test_study.py`,
`tests/test_study_media.py`, `tests/test_playback_timing.py`, and
`tests/verify_public_study.py` (paths relative to this directory except `deploy/`).

Exclude all `qa-public-*` records from participant analysis. Test browser profiles
under `/tmp` contain their own session cookies and cached media; do not publish them.
