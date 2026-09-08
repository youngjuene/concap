# Regen bilingual QA — 2026-09-08

**Result: both original A/B interface journeys pass after fixes. One study-content localization gap remains: Korean survey scale anchors are still English.**

The requested localhost was freshly restarted at **http://127.0.0.1:8779/**. Final server PID: **3148767** (original PID 2700137). Existing Gemma settings, CUDA device selection, access-code gate and study output were preserved. Test participants were created only in `/tmp/regen-qa-20260908/responses`, through a separate loopback instance on port 18779, which was stopped after QA.

## Scope and approach

Tested the original bilingual `dpo regen serve` interface using actual staged media and the configured local Gemma writer, real Chrome playback, Playwright browser interaction, screenshots, API edge-case probes, native QA/implementation/review agents, the UltraQA scenario checklist and visual-verdict review. No packages were installed. OMX tmux orchestration was unavailable on this App surface; no OMX runtime completion is claimed.

Success criteria: fresh server responds, each language reaches a receipt through every participant step, exported records agree with browser behavior, reported defects are reproduced and retested, and unsupported/content-dependent claims remain explicit. No study-data test writes, access-code disclosure, broad process termination or external publication were performed.

The separate two-stage `study_api` interface was included in Python regression coverage, but not treated as a completed bilingual browser product: its documentation identifies its survey UI as English-only.

## Verification

| Check | Result | Evidence |
|---|---|---|
| Full Regen Python suite | **396 passed**, one existing Starlette/httpx deprecation warning; 25.22 seconds | `/tmp/regen-qa-20260908/pytest.log` |
| Lint and formatting | Ruff passed; 24 Python files already formatted | Commands below |
| Strict types | mypy passed for 24 source files | Commands below |
| JavaScript / whitespace | Both JS syntax checks and `git diff --check` passed | Commands below |
| Complete English and Korean journeys | **70 assertions passed**; both reach `done`; no normal-flow page errors or HTTP 5xx | `browser-results.json`, `browser-final.log` |
| Repeated network failures | **46 assertions passed**; each form upload aborted twice per language, input retained, localized message shown, retry succeeds | `network-retest-results.json` |
| Malformed inputs / sequencing | **32 probes passed** | `api-results.json`, `api-final.log` |
| Gate, caching, ranges, completed-state integrity | **16 probes passed** | `final-probes.json` |
| Mark/image geometry | **6 checks passed**, matching center coordinates at 320, 390, 1280px in both languages | `geometry-results.json` |
| HTTP failure classification and sequential recovery | **8 checks passed** for English/Korean 400, 500 and 503 diagnostics plus recovery | `http-errors-results.json` |
| Visual comparison | Existing theme retained; 94/100 qualitative layout verdict | `visual-verdict.json`, final screenshots |

The final HTTP-error classification checks used a short-lived template-writer QA instance with separate `error-mode-responses`; the complete-journey Gemma evidence above was retained. Transport retries were also rerun successfully after the HTTP-error change.

All evidence filenames above are under `/tmp/regen-qa-20260908/` unless otherwise stated. Browser layout checks use 320px and 390px viewports plus desktop. These are Chrome viewport simulations, not physical-device Safari tests.

## Scenario matrix

| ID | Intent / user model | Setup and harness | Expected / actual result | Fix | Evidence / cleanup |
|---|---|---|---|---|---|
| HOST | Operator requests a fresh local server | `restart.py`, `restart-final.py` | New PID and HTTP 200; final PID 3148767 | Restart with original settings | `restart-final.json`; main server retained |
| EN | English participant | `browser.mjs`, actual media/model | Both viewings, 8-item first survey, marking, five sound families, regeneration, 22-item final survey, download and reload completed | Shared UI fixes below | `en-export.json`; QA data retained separately |
| KO | Korean participant | Same journey after selecting Korean | Korean UI/captions and locked language survive reload; completion reached | Localized title and progress landmark; English scale-anchor gap remains | `ko-export.json`; screenshots retained |
| MOBILE | Participant with a small viewport | Every screen at 320px/390px | No horizontal overflow after fixes | Header and moment-picker wrap | `browser-results.json`; screenshots retained |
| GEOMETRY | Keyboard/pointer marking on responsive image | `geometry.mjs` | Points use actual image geometry at all three sizes | Remove loaded-image plate minimum height | `geometry-results.json`; screenshots retained |
| INTERRUPT | Participant leaves fullscreen | Exit fullscreen then click Resume | Video pauses, resumes, interruption appears in export | None | `*-export.json`, `*-interrupted.png` |
| RESUME | Reloaded/completed or stale client | Start/completion reload; duplicate/out-of-order API calls | Same locale/state, invalid transitions rejected, exactly two viewings and two survey records | None | `final-probes.json`, `export-verification.json` |
| NETWORK | Unreliable connection | `network-retest.mjs`; twice-aborted survey/visual/auditory uploads | Inputs remain selected, retry enabled, localized failure and successful continuation | Catch failed submissions in-place | `network-retest-results.json`; routes/browser contexts closed |
| INPUT | Malformed or hostile client | `api-probes.py`; broken JSON, arrays/objects, long strings, Unicode, traversal and instruction-like text | Controlled 4xx, no execution or state advancement | Reject non-string page/step before dictionary lookup | `api-first-results.json` → `api-results.json` |
| MEDIA | Browser/cache client | Real video playback and captions, Range and ETag requests | H.264 decode, localized captions, 206 range response, 304 revalidation | No product codec change | Browser exports, `final-probes.json` |
| GATE | Unauthorized/forwarded client | Main server without code and forwarded log request | Both denied with 403, no new study participant | None | `final-probes.json` |
| HYGIENE | Dirty state / misleading output / hung harness | Before/after worktree, exit checks, bounded probe timeouts | No unrelated work overwritten; setup failures separated from app failures | Use compatible Chrome and host TestClient execution | Reported below; temporary server stopped |

Prompt-agent cancellation and OMX state manipulation are not participant actions in this app; application reload, stale-step, literal-input and retry checks were used instead. No load-test capacity or arbitrary concurrent model inference guarantee is claimed.

## Failures found and fixes

- `app.py`: list/object `page` and `step` values caused `TypeError` and HTTP 500. Two type guards now return 400; four regression cases were added in `tests/test_qa_validation.py`.
- `regen.js`, `copy.py`: failed survey/visual/auditory uploads could leave submission controls disabled and raise unhandled promises. Transport failures now keep the current form and provide a localized retry message using existing status elements. HTTP/API failures retain the actual diagnostic instead of being mislabeled as connection failures. Sequential diagnostic errors also reset the Retry button, fixing a second-error dead end found in the final review retest.
- `regen.js`, `copy.py`: Korean selection/reload left the browser title and progress landmark in English. Both now synchronize with the selected locale.
- `regen.css`: the progress/language header and visual moment selector overflowed at 320px. Existing flex containers now wrap at narrow widths.
- `regen.css`: the 240px marking-plate minimum height exceeded the displayed mobile image, shifting percentage-positioned marks. The plate now follows the image height; all six measured image/point center checks match.
- `tests/test_legacy_language.py`: added locale-copy coverage for progress labels and retry messages.

The implementation reuses existing controls, state and status regions. No new dependency or application layer was added. Independent code review flagged the distinction between transport and HTTP errors; the follow-up browser test verified the correction and exposed the sequential Retry-button issue, which was also corrected. This report and the local validation test are the new repository files; no commit was created.

## Model and record evidence

On the final complete journeys, actual Gemma regeneration took **16,025ms in English** and **1,113ms in Korean**, with `writers=["gemma"]` and `fallback=false`. These are two observations, not a latency benchmark. The first complete pass also produced real Gemma output without fallback in both languages (16,801ms / 1,222ms).

Each final export contains exactly two viewings of different segments, two survey records with 8 and 22 answers, one regeneration event, the selected language, and completion state `done`. Export checks also verified the fullscreen interruption record. See `export-check.py` and `export-verification.json`.

## Commands and apparatus issues

Successful commands (exit 0):

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/regen src/dpo/regen/tests -p no:cacheprovider -o faulthandler_timeout=45
.venv/bin/ruff check --no-cache src/dpo/regen
.venv/bin/ruff format --check --no-cache src/dpo/regen
.venv/bin/mypy --cache-dir /tmp/regen-qa-20260908/mypy-final src/dpo/regen
node --check src/dpo/regen/regen.js
node --check src/dpo/regen/study.js
git diff --check
node /tmp/regen-qa-20260908/browser.mjs
node /tmp/regen-qa-20260908/network-retest.mjs
node /tmp/regen-qa-20260908/geometry.mjs
node /tmp/regen-qa-20260908/http-errors.mjs
.venv/bin/python /tmp/regen-qa-20260908/api-probes.py
.venv/bin/python /tmp/regen-qa-20260908/final-probes.py
.venv/bin/python /tmp/regen-qa-20260908/export-check.py
```

Repository commands ran from `/mnt/hdd/research/2026/concap`. Browser harnesses imported the already installed `/tmp/atlas-qa/node_modules/playwright-core/index.mjs` and launched `/home/june/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome`.

The older Chromium 1140 build could not decode H.264; its failure was a tooling limitation, resolved using the newer installed Chrome. Sandboxed TestClient tests stalled and were interrupted; the same complete suite passed outside that sandbox. A first detached QA server exited without a diagnostic, and was replaced with a managed execution session. An initial Ruff format command tried to write the read-only repository cache; rerunning with `--no-cache` passed. These failures are not counted as successful product tests.

## Cleanup and remaining limits

- Main localhost 8779 remains running with the final fixes and its original configuration. QA localhost 18779 was stopped; browser processes/contexts were closed.
- Test sessions (including separate `error-mode-responses`), screenshots, harnesses, before/after failure results and logs under `/tmp/regen-qa-20260908` are intentionally retained as QA evidence. The final QA server log is `/tmp/regen-qa-20260909-qa-final.log` (filename date is a naming typo). Other pre-existing `/tmp` artifacts were not removed.
- The initial worktree was clean. Intentional changes are `app.py`, `copy.py`, `regen.css`, `regen.js`, `tests/test_legacy_language.py`, new `tests/test_qa_validation.py`, and this report.
- **Open localization finding:** Korean questionnaires still show configured English scale anchors “Not at all” / “Very much”. The anchors are part of the hashed study calibration (`config.py`, `app.py` survey scale response), so translating the chrome alone would silently change a measurement without recording its configured wording. A versioned Korean scale-content decision remains necessary for a fully localized study.
- The server reports questionnaire provenance as `placeholder`; this QA does not validate the measurement instrument, translations, or caption grounding/quality for research use.
- Unsubmitted drafts are not persisted through a full page reload in the legacy interface. The new retry fix preserves current in-memory input across failed uploads; it does not add persistent draft storage.
- No physical mobile device, Safari/Firefox, screen reader, public tunnel, sustained load, or broad model-quality qualification was performed. The newer two-stage study UI remains outside the bilingual browser result.
