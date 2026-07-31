# Wiring TODO — after Phase 2

Deferred deliberately, not forgotten. Each subsystem below is already built and
unit-tested but has no production caller; wiring it before `dpo select run` has
published a validation report, a selection report, and a lock would mean
designing against no data — which is exactly how the dead knobs this repo just
shed came to exist. Revisit this file once Phase 2 lands (see
`docs/study-runbook.md` for the sequence).

## 1. `dpo report analyze` — the results-comparison command

**What exists:** `src/dpo/analysis/` — `fit_bradley_terry` (per-experiment
strength from pairwise outcomes), `clip_cluster_bootstrap` (CIs that respect
clip clustering), `benjamini_hochberg` (multiplicity across the nine
experiments), `flip_curve` and `sliced_preference_reports` (robustness). All
tested in `tests/analysis/`, none reachable from the CLI.

**The wiring:** a `report analyze` action that consumes the published
validation report plus the per-pair scored data, and emits the comparison the
study actually reports: experiment ranking with bootstrap CIs, BH-corrected
significance across experiments, and accuracy sliced by difficulty/agreement.
`dpo report show` already prints raw per-variant accuracy; `analyze` is the
inferential layer on top.

**Also restore then:** `validation.bootstrap_samples` — made optional and
dropped from the live contract while nothing read it; when `analyze` lands,
its resample count should come from the contract again, not a CLI flag.

## 2. Flip-manifest consumers — close the robustness loop

**What exists:** `views derive` publishes `dpo.flip-manifest/v1` artifacts for
every `[robustness].flip_rates` entry, and `dpo/data/noise.py` has
`parse_flip_manifest` / `apply_flips` to read them back. Nothing downstream
consumes a flip manifest today: the robustness data is produced and then
orphaned.

**The wiring:** score selected variants on flipped views (via `apply_flips`)
and feed `analysis.robustness.flip_curve` — naturally part of, or a sibling
to, `report analyze`. If the study ends up not reporting robustness curves,
the honest alternative is to drop `[robustness]` from the live contract and
stop publishing manifests no one reads; producing-and-ignoring is the worst of
the three options.

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

## 4. Small orphans — keep-or-delete, one decision each

| symbol | intended for | note |
| --- | --- | --- |
| `load_config_text` (`models/gemma4/backend_config.py`) | the `evaluate` live-scoring boundary: parsing an artifact-resolved backend config without a file path | `dpo evaluate` is a pure exit-3 gate today. Wire only if external/live scoring is ever actually needed — for this study, `select run` + `study export` may make it permanently unnecessary. |
| `compute_report_fields` (`pipeline/experiments.py`) | report enrichment | fold into `report analyze` or delete. |
| `transcribe_generation` (`candidates/generation.py`) + the `transcribe_speech` track knob | speech transcription during candidate generation | dead pair: the knob is `false` in every contract and the function has no caller. Street audio is non-speech, so likely delete both — but that changes the `[tracks]` schema surface, so do it alongside the next corpus regeneration, never mid-study. |

## Not wiring, but blocks reporting

- **wDPO scalar maps** (`objectives/wdpo.py`): explicit, documented
  placeholders where `arxiv_2603.07211v1` leaves the maps unspecified, and
  `code_commit` is unpinned. Re-verify against the pinned official code before
  reporting any wDPO number — or report wDPO as excluded, with this as the
  stated reason.

## Beyond wiring (separate builds, separately planned)

- Phase 4 user-study instrument (`src/dpo/userstudy/`, A′ ambience staging,
  ratings + comprehension schema) — planned in the session plan file; A′
  sound character still needs author sign-off.
- `dpo.study-results/v1` / `dpo.analysis-report/v1`: reserved in
  `core/artifacts.py` (public-derived, GC roots), producers to be built once
  the user-study analysis shape is settled.
