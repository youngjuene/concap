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
