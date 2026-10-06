# Researcher dashboard contract

Separate read-only app, localhost port 8781. Backend normalizer module
`dashboard_data.py`; HTTP/CLI module `dashboard.py`; static assets
`dashboard.html`, `dashboard.css`, `dashboard.js`. No dependencies added.

`DashboardData(legacy_dir: Path, viewing_dir: Path).overview(include_qa=False)`
and `.participant(id: str, include_qa=False)` return JSON-safe dictionaries.
Participant IDs returned by overview are the only accepted detail selectors.
Reader must never initialize StudyStore or write source files.

GET `/api/overview?include_qa=false`:

```json
{
  "updated_at": "ISO UTC",
  "warnings": [],
  "metrics": {"participants":0,"phase1_complete":0,"phase2_started":0,"phase2_complete":0,"survey_submissions":0,"response_values":0,"control_changes":0,"generated_exposures":0,"fallback_exposures":0,"generation_median_ms":null},
  "participants": [{"id":"safe selector","label":"p123","language":"en","qa":false,"phase1_status":"done|in_progress|missing","phase2_status":"done|watch|missing|...","phase1_surveys":0,"phase2_surveys":0,"videos_completed":0,"control_changes":0,"last_activity":"ISO or null"}],
  "responses": [{"participant":"selector","phase":1,"scope":"art|survey|overall|video|preferences","video_id":null,"instrument":"stored hash or unknown","item_id":"id","item_text":"stored wording or item_id","wording_available":false,"type":"rating|choice|text|unknown","value":4,"points":null,"options":[],"language":"en","submitted_at":null}],
  "activity": [{"date":"YYYY-MM-DD","phase1":0,"phase2":0}]
}
```

GET `/api/participants/{id}?include_qa=false`:

```json
{
  "participant": {"same":"summary row"},
  "responses": [],
  "interactions": [{"phase":2,"video_id":"viewing-1","at":"ISO UTC","position_ms":5000,"texture":0.5,"context":0.8,"revision":1,"kind":"settings"}],
  "captions": [{"phase":2,"video_id":"viewing-1","text":"caption","start_ms":0,"end_ms":5000,"kind":"generated|prepared|fallback|job","displayed":true,"job_id":null,"revision":null,"texture":null,"context":null,"generation_ms":null,"created_at":null,"status":"succeeded|..."}],
  "events": [{"phase":1,"at":"ISO or null","kind":"step.entered","video_id":null,"position_ms":null,"detail":{}}],
  "warnings": []
}
```

Responses contain only submitted answers (no draft duplicates). Phase 1 embedded
calibration responses supplement missing JSONL only; deduplicate by original
submission identity. Separate per-video/final/preference Phase 2 surveys. Do not
mix hashes/wording/languages/scopes when aggregating distributions. Generation
median uses completed job result.duration_ms, never exposure duration or deadline.
Interactions use settings events plus recorded playback position or position hint,
label missing position null. Caption rows include displayed exposures and jobs
never displayed, without double-counting displayed exposures in metrics.

HTTP layer also provides `/api/export/responses.csv?include_qa=false` and
`/api/export/participants/{id}.json?include_qa=false`. Reject non-local/proxied
requests on all routes. All API strings are untrusted text in the UI. CSV neutralizes
spreadsheet formulas. API omits bearer tokens, cookies, prompts and media file paths.

Response rows also carry `flow_version`, `condition`, and `view_id`; these must
be included in cohort labeling and `flow_version`/`condition` in distribution
group keys. Activity is deduplicated survey submissions per UTC day, not answered
item counts or raw interaction events.

New event rows retain explicit stage, clip ID/index, viewing ID, condition, flow
version, session ID and configuration hash when recorded. Historical missing
identity remains null. Issued context IDs are omitted. Participant summaries add
`videos_total` from the frozen viewing list (null before assignment is available),
and the UI shows completed/assigned videos. Settings interactions use the server's
recorded `settings_revision`, enabling a direct join to generated exposures.

Spreadsheet-protocol responses additionally preserve `stimulus_assignment` as
`{audiovisual_congruence, description_depth}` and `caption_strategy` as a separate
field. Historical scalar assignments remain readable. Response groups and
filters keep both factorial axes and the strategy separate from first/second
viewing `condition`; CSV writes the assignment object as JSON. Multiple-choice
distributions count answers selecting each option, independent of click order,
and explicitly allow percentages across options to total more than 100%.

The read-only loader searches both `questionnaire.sqlite3` and `study.sqlite3`
under both configured directories. Identical files are read once, and copied
snapshots of the same session use its highest committed revision rather than
double-counting answers. New short-phase mutation events retain server-stamped
phase/page/stage, clip and viewing identity, protocol, condition, assignment and
strategy. Practice is `phase=0` with `analysis_excluded=true`; it is excluded from
formal collection views. Historical missing scope remains null.

## Phase 1 observational analysis

The read-only `DashboardData.phase1_analysis(include_qa=False, *, participant="",
language="", status="", clip_id="")` supplies the analysis view and both new
CSV exports from the same filter contract:

| GET endpoint | Result |
| --- | --- |
| `/api/phase1-analysis` | Grouped analysis JSON |
| `/api/export/phase1-observations.csv` | One submitted round per observation row |
| `/api/export/phase1-pairs.csv` | One first/second pair per session and clip index |

All three accept these query parameters:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `include_qa` | `false` | Include QA-labelled participants |
| `participant` | empty | Case-insensitive substring search of participant label or selector |
| `language` | empty | Exact participant-language filter |
| `status` | empty | `complete` (Phase 2 complete), `active` (not complete), `phase1` (Phase 1 complete), or `no_phase2` |
| `clip_id` | empty | Exact short-clip filter applied before analysis aggregates |

An empty string means no filter; an unsupported nonempty status selects no
participants. `available_clips` reflects the selected participant cohort before
the clip filter, so users can change clips without clearing their cohort.

```json
{
  "schema": "dpo.phase1-analysis/v1",
  "observations": [],
  "pairs": [],
  "summaries": [],
  "selection_rates": [],
  "transitions": [],
  "warnings": [],
  "available_clips": [],
  "filters": {"include_qa": false, "participant": "", "language": "", "status": "", "clip_id": ""},
  "updated_at": "ISO UTC"
}
```

`observations` retain source sound/menu codes and visual points, saved mask
measurements and provenance, scope, status and issues. `pairs` retain both rounds
with `_1`/`_2` suffixes and `area_delta_pp`, `type_changed`, `family_changed`,
`frame_changed`, `menu_changed` and `order_changed`. Type/family changes are
integer `0`/`1` or null; menu/frame/order flags are boolean or null.

Summaries group by clip, instrument, codebook version/hash, assigned AV condition,
description depth, caption strategy and menu-change flag. Each metric has its own
total/valid/NA denominator. `selection_rates` carries explicit numerators and
denominators; zero eligible observations yields null, not zero. Transition kinds
`type` and `family` describe valid first/second pairs. Kind `visual_sound` instead
contains per-round selected visual-code × sound-type cross-counts.

`av_assigned_condition` is copied assignment metadata. `av_selection_relation`
is available only from an explicit frozen relationship codebook; it is not an
accuracy score. No relationship is inferred from labels. Historical saved match
codes can remain available while missing mask areas stay null; there is no
filesystem backfill or source mutation.

CSV columns are the explicit `OBSERVATION_FIELDS` and `PAIR_FIELDS` lists in
`phase1_analysis.py`. Null becomes an empty CSV cell, nested values are JSON,
and existing UTF-8 BOM/formula-neutralization and local-access protections apply.
These exports honor analysis filters; the general Responses CSV retains its
existing whole-QA-cohort scope. Credentials and internal prompts are excluded.
See [PHASE1_ANALYSIS.md](PHASE1_ANALYSIS.md) for functions, field semantics,
codebook examples, exact equations, historical-data policy and interpretation
limits. No current full-QA completion is implied by this API specification.
