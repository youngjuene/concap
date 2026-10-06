# Configurable clips and reliable collection

Implemented from `.omx/plans/regen-consistency-and-variable-clips.md` on 2026-09-22.
Existing working-tree changes were preserved. No participant files were migrated,
no service was restarted, and no dependencies were added.

## Delivered

- New integrated documents use `dpo.caption-regen/v4`, arbitrary safe segment IDs,
  and an explicit nonempty `clip_order`. Enrollment freezes the rotated order.
  V3 A/B documents and historical sessions retain their existing behavior.
- `regen scaffold --clips ... --segment-ids ...` creates a variable-count v4
  scaffold. Omitting segment IDs retains the historical two-clip A/B scaffold.
  Each output still requires authored caption text and media annotations.
- Viewing manifests and staging accept any positive video count. Participant
  progress, video questions, final-survey transitions and dashboard totals use
  the configured/frozen count. Long-video duration requirements are unchanged.
- New browser sessions negotiate protocol 3. Short playback checkpoints survive
  reload and enforce coverage before questionnaires. The algorithm is shared with
  Phase 2; short clips have a smaller duration-relative completion tolerance.
  Server start/end timestamps and submitted client timestamps remain distinct.
- Phase 1 interaction records retain their issued phase/stage/clip/view/condition
  context and stable event IDs. A persistent outbox retries failed uploads;
  acknowledgment and restart-safe server deduplication prevent duplicate rows.
  The signing secret is excluded from exports. Handoff drains pending uploads.
- Partial event-write failures invalidate the cached receipt index; retry repairs
  a torn tail and preserves already-written events without duplicate appends.
- Phase 2 drafts retain `{video_id, answers}`, and questionnaire headings render
  from the frozen instrument. Historical draft shapes remain readable.
- Accepted settings events record their assigned settings revision and origin.
  Unchanged settings avoid extra revisions/generation work for new sessions.
  Frontend coalescing retains the latest requested value, including an in-flight
  A-to-B-to-A reversal, and failed settings remain retryable.
- Dashboard events preserve recorded identity without inventing historical
  missing metadata; participant rows show completed/assigned video counts.

## Changed implementation areas

- `document.py`, `app.py`, `continuation.py`, `regen.js`: clip configuration,
  session compatibility, scoped collection, checkpoints and phase handoff.
- `log.py`: durable idempotent telemetry and export redaction.
- `playback.py`: shared coverage/sequence validation extracted from `study_api.py`.
- `study_api.py`, `study_store.py`, `study_schema.py`, `study_prepare.py`: variable
  counts, survey drafts and accepted settings provenance.
- `study.js`, `study-ko.json`: dynamic counts, frozen headings and control recovery.
- `dashboard_data.py`, `dashboard.js`: retained identities and assigned totals.
- `../../cli/regen.py`: optional explicit segment IDs in the scaffold command.
  The repository-relative path is `src/dpo/cli/regen.py`.
- `STUDY.md`, `DESIGN.md`, `DASHBOARD_API.md`: current protocol/configuration contracts.
- Focused Python tests plus `tests/browser_protocol.mjs`,
  `tests/preview_protocol.py`, `tests/settings_coalescing_vm.mjs` and
  `tests/benchmark_protocol.py` provide reproducible verification.

## Verification evidence

- The full API count matrix `(1 short, 1 long)`, `(2, 3)`, `(4, 5)` reaches final
  completion through actual routes, including mounted handoff, with exact
  viewing/observation/survey cardinality and frozen-source hashes.
- The final full regression run passed 568 tests. Follow-up focused checks cover
  subsequent scope/recovery adjustments; browser results are recorded separately.
- Ruff lint/format checks, strict mypy for the package and scaffold CLI,
  JavaScript syntax checks, and diff whitespace checks passed.
- `settings_coalescing_vm.mjs` executes the live settings queue implementation
  and proves that A-to-B-in-flight-to-A ends at A after recording both acknowledgments.
- The browser protocol harness separately exercises real short-clip playback,
  skip rejection, checkpoint reload, failed telemetry uploads, deduplication and
  variable-count surveys. Its result records which later coverage was supplied
  by controlled test helpers; those paths do not qualify actual long-video playback.
- Fresh browser journeys passed EN 1 short / 2 long, KO 1 / 1 and EN 2 / 1.
  The EN 1 / 2 rerun also hides the page while a checkpoint acknowledgment is
  pending and verifies pause/resume; its final outbox is empty. The prior nonempty
  outbox was traced to a test reset clearing storage while old-page callbacks
  remained active; clearing at the next document start removed that test race.
  Evidence: `/tmp/regen-protocol-qa-en-1x2-hide/browser-protocol-result.json`,
  `/tmp/regen-protocol-qa-ko-1x1/browser-protocol-result.json` and
  `/tmp/regen-protocol-qa-2x1/browser-protocol-result.json`.
- An independent verifier passed 13 protocol/count/instrument tests outside the
  restricted sandbox. A follow-up 19-test pass covers the final chapter scope,
  playback and durable event changes.
- The existing EN/KO survey-hierarchy browser harness also passes through all
  three per-video surveys and final completion, including the historical rail.
  Test-only loopback helpers seed simulated playback coverage; survey submission
  and draft APIs remain real. Evidence: `/tmp/regen-survey-hierarchy/browser-survey-hierarchy-result.json`.
- Independent code/spec/security review approved the main changes and the
  scaffold follow-up. Independent architecture approval is unavailable: the
  installed architect role's configured model is unsupported on this account.

## Bounded performance evidence

`tests/benchmark_protocol.py` compares identical no-model sessions using 20
unchanged settings requests and five playback acknowledgments. In the measured
run, protocol 3 produced 2 job rows and no settings revisions; the historical
path produced 44 job rows and 20 settings revisions. Playback acknowledgment
p95 was 21.77 ms versus 34.335 ms. These are local orchestration measurements,
not GPU inference speed, caption quality or deployment-capacity evidence.
The raw run is `/tmp/regen-protocol-performance.json`.

## Cleanup and remaining qualification

The cleanup pass was bounded to implementation changes and protected by regression
tests. Shared playback logic was extracted once, duplicated settings scheduling
state was removed, explicit count/scope boundaries replaced fixed assumptions,
and historical compatibility remains confined to documented version boundaries.
Prepared-caption fallback remains intentional and recorded. Failed telemetry is
retained for retry instead of being swallowed and discarded.

Fresh GPU qualification could not run: `nvidia-smi` cannot communicate with the
driver here. The architecture-review workflow cannot be marked approved without
its independent architect lane. OMX mode-state writes were also unavailable
because its existing session scope could not be resolved; no runtime completion
or native Codex goal completion is claimed. Questionnaire validation, actual participant
burden, higher-concurrency capacity and deployment remain separate work.

Follow-up clarification (2026-09-22): the GPU limitation above was sandbox-specific.
An approved host-level check found both RTX 3090s accessible. A separate current-source
QA instance on port 18789 then exercised real browser playback and GPU-backed short
regeneration: the cold request took 21,904 ms and correctly fell back after exceeding
the unchanged 20,000 ms ceiling; a warm request on the other clip took 906 ms without
fallback. This is bounded remote-host smoke evidence, not full local-computer E2E or
complete GPU qualification. See `LOCAL_COMPUTER_QA_HANDOFF.md` for the ready target,
artifact paths and remaining local-session work.
