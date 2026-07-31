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

## 3. Data-level leakage gates — decision required, not just wiring

**What exists:** `dpo/data/leakage_audit.py` — `audit_corpus`, `audit_pool`,
`audit_sft_rows`, `audit_text_leakage`. The canary's "leakage oracle" tests
the *store's* protected-exposure rejection; these functions audit the *data*
(text overlap across splits, role bleed in derived views) and are never
called.

**The wiring:** run them inside `views derive` before anything publishes.

**The decision (authors', not an implementation detail):** these are refusal
gates — a false positive blocks a legitimate run. Wire them as hard failures,
as warnings recorded in the view artifacts, or not at all; pick before Phase 2
re-runs views on real annotations, because retrofitting a gate after views are
published means regenerating them.

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
