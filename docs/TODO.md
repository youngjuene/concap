# Wiring TODO — after Phase 2

Deferred deliberately, not forgotten. What is built and has no production
caller is listed here with the reason; what was wired since is marked so the
next session does not re-derive it. Revisit once Phase 2 lands (see
`docs/study-runbook.md` for the sequence).

## 1. `dpo report analyze` — DONE, publishes

`dpo report analyze` takes the validation and selection report ids and
publishes `dpo.analysis-report/v1` (stage `analyze`, contract slice
`validation`): clip-clustered bootstrap CIs per experiment at
`validation.bootstrap_samples` (10000 on `street-audio`, 200 on the canary
because it is a speed fixture), exact paired sign tests vs SEED with BH
correction, a Bradley-Terry fit over per-pair contests between the selected
variants, the natural-noise slices for the ranked winner, and the flip-rate
curves (entry 2). It re-scores nothing, so a rerun republishes to the same id.
Verified against the canary and the CLI pipeline end to end.

## 2. Flip-manifest consumers — DONE, closes the robustness loop

`views derive` publishes `dpo.flip-manifest/v1` per `[robustness].flip_rates`
entry; `train run` now takes those manifests and retrains every preference
cell on `pair_strict` once per positive rate (`training_stage.matrix_cells`
is the one enumeration of the matrix); `select run` scores each retraining on
the unflipped validation pairs and persists the scores under
`robustness_scores`; `report analyze` assembles `flip_curves` from them.

**What that costs on the live contract:** three positive rates × seven
pair_strict arms = 21 extra E4B cells on top of the 9 base cells. The
contract's own comment says how to opt out honestly (`flip_rates = [0.0]`);
what is no longer possible is publishing manifests no report reads.

## 3. Data-level leakage gates — RESOLVED (this entry was wrong)

The auditors were never unwired: `run_leakage_audit(enforce=True)` runs inside
`publish_track_views` (`pipeline/view_stage.py`), so views derive was always a
hard gate. The original entry here was a dead-symbol-sweep false positive
(the individual auditors' only caller is same-file).

What that gate revealed when run against the live pools ahead of time: 8
cross-split near-duplicate violations that would have refused views derive
*after* annotation. Resolved with `dpo candidates dedup` — a code-owned
transform sharing the audit's threshold constant — which cut both sides of
every collision, republished both pools with `dedup-source` lineage, and
proved in-band that the audit now passes. Within-split duplicates remain by
design: the gate does not police them, and cutting them is not representable
on this corpus (34 of 48 clips would fall below `per_clip_min`).

**Orphaned on the other track:** only the two audio pools carry a
`dedup-source` edge. The visual pools in `artifacts/street/` (`238ad3d4…`
train, `8e7981fa…` validation) predate the decision to declare
`[tracks.audio]` alone, so nothing on the live contract can read, dedup, or
export them — `candidates generate --track visual` is refused outright. One
decision, not two: either restore `[tracks.visual]` and `[backends.visual]`
from git history and put the visual pools through the same dedup pass before
exporting a session, or drop them at the next `artifact gc`. Leaving them is
the option that later reads as a second track having been collected.

## 4. The one remaining dead knob — `transcribe_speech`

`tracks.audio.transcribe_speech` is parsed into `CaptionContract` and read by
nothing. (`transcribe_generation` in `candidates/generation.py` is not the
knob's consumer and is not dead: it is the tiny backend's hex transcription of
raw byte output, on the live offline path — an earlier version of this entry
had that wrong.) Deleting the knob is a `[tracks]` schema change, and the
`candidates` stage keys its artifacts on the `tracks` section: dropping the
line from `street-audio.toml` re-keys both live audio pools and orphans the
212 exported annotation tasks. Do it alongside the next corpus regeneration,
never mid-study.

## Not wiring, but blocks reporting

- **wDPO scalar maps** (`objectives/wdpo.py`): explicit, documented
  placeholders where `arxiv_2603.07211v1` leaves the maps unspecified, and
  `code_commit` is unpinned. Re-verify against the pinned official code before
  reporting any wDPO number — or report wDPO as excluded, with this as the
  stated reason.

## Beyond wiring (separate builds, separately planned)

- **User-study instrument — BUILT** (2026-07-31/08-01), as a congruency-ladder
  study rather than the A/A′ presentation study that was planned here:
  `dpo study export` measures the axis and publishes ladders
  (`evaluation/congruency.py`, `evaluation/ablation.py`,
  `pipeline/study_stage.py`), `dpo study serve` runs the participant-facing
  app (`src/dpo/userstudy/`, schema `dpo.userstudy-responses/v2`). A′ staging
  was not built and is not needed by this design; it belongs to the
  three-condition follow-on study (`docs/proposal.md` §4.6), which stays
  unbuilt.
- **Responses are ingested — DONE.** `dpo study ingest` validates every
  `responses-<participant>.json` against the export (`userstudy/responses.py`)
  and publishes `dpo.study-responses/v1` (the record) and
  `dpo.study-results/v1` (the analysis, `analysis/study.py`; stage
  `study-ingest`). The analysis is descriptive with clip- and
  participant-clustered bootstrap intervals; the mixed-effects fit the
  proposal names is downstream work over the persisted rows, not in this
  repository.
- **Rung count is a contract key — DONE.** `[study].rungs` replaces the
  `--rungs` flag; the `study-export` and `study-ingest` stages declare the
  section, so an export made under a different rung count has a different
  identity and `study ingest` refuses it.

## Removed as dead (2026-08-25), recoverable from git history

- `dpo evaluate run`, the exit-3 placeholder (`DEFERRED_GATES`), and the
  `live-boundary-smoke` make target; the real live gates in `_resolve_backend`
  are unchanged.
- `load_config_text` (`models/gemma4/backend_config.py`): no caller.
- The reserved `dpo.test-metrics/v1`, `dpo.test-reservation/v1`,
  `dpo.test-resume/v1`, `dpo.test-finalization/v1` types, and the
  `confirmatory-test` capability scope whose reservation table nothing
  created. The test role stays sealed by construction: it is in
  `CAPABILITY_READ_ROLES` with no scope in `ROLE_SCOPES`, so no capability can
  open it until the confirmatory apparatus is restored.
- `[terminal_states]` and `training.world_size`: optional contract keys no
  stage read.
