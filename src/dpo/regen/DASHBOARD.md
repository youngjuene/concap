# Research dashboard

The researcher console reads the existing Phase 1 JSONL/snapshot files and Phase 2
SQLite database. It runs separately from the participant interface, without
changing responses, triggering inference or loading a GPU model.

From the repository:

```sh
make dashboard
```

Open **http://127.0.0.1:8781/** on the study machine. Custom sources:

```sh
.venv/bin/python -m dpo.regen.dashboard \
  --legacy-dir data/live/regen-responses \
  --viewing-dir data/live/regen-viewing-responses --port 8781
```

The dashboard accepts local requests only, checks the host/origin, rejects
forwarding headers, and is not published by Tailscale Funnel. Do not add this
port to the participant Funnel. Researcher access from a different machine would
need a separately authorized private access arrangement.

The dashboard launched in this session writes its process record and server log
to `/tmp/concap-dashboard-service/`. `make dashboard` is the foreground command
to use when it is not already running. The participant service remains on 8779.

## What to review

- **Overview:** collection counts, completion by phase, submissions by UTC day,
  generated/fallback exposure counts and participant status.
- **Responses:** exact recorded answers and distributions separated by phase,
  questionnaire/version, condition, video and language. Historical item wording
  or scale limits are marked unavailable when they were not recorded.
- **Session inspector:** a participant's submitted surveys, control changes over
  video playback time, generated sentences and prepared fallbacks, model job
  duration, and a chronological interaction/event log.
- **Exports:** normalized response CSV and a sanitized participant JSON record.
  QA sessions are excluded unless explicitly included. Spreadsheet formulas in
  text fields are neutralized on CSV export; the source values remain unchanged.

The CSV exports the whole selected QA cohort, regardless of the UI search/item
filters; the button and response view label that scope. Inspector JSON contains
the selected participant's records. Optional 15-second refresh is off initially.
When historical control logs lack playback positions, the timeline can use
recorded seconds since the first control change; the axis labels identify which
clock is being used. Solid/dashed strokes distinguish coincident control levels.

## Interpretation

### Linked inspection and metadata

The response view keeps question wording, phase/video/condition and sample counts
visible. **Question details** contains the complete item/instrument/flow identity
and denominator definition. **Record details** in tables retains the original
IDs and provenance; reducing visual repetition does not change grouping or exports.

In the session inspector, click a timeline point, caption window or event moment.
The **Selected moment** panel describes the focus; captions and events sit side
by side and update together for the same phase/video. A control/event with a
playback position shows a ±5-second neighborhood. Caption selection uses its
exact half-open display window, so an adjacent caption beginning at its end is
not included. A matching control event is highlighted in the event list.

If playback position is missing, captions stay unfiltered and event matches use
only recorded time ±5 seconds, with this limitation shown in the moment panel.
Job-created timestamps are never used as visible-caption timestamps. This is
inspection of nearby records, not a claim of causal or measured response latency.
**Show all moments** clears the temporal focus while retaining the chosen video.
Changing participant/video clears it; refreshing the same session preserves it.
The same selections work with keyboard Enter/Space.

Model timing and submitted-response panels remain available in expandable
sections below the linked records. The established colors, sidebar and overall
desktop sizing are retained.

Counts describe records, not scientific validation. N/A, unanswered/absent values,
and numeric zero are distinct. Incomplete sessions remain visible. Instrument
versions and viewing conditions must not be silently pooled. Model generation
duration is not network latency or exact time from moving a control to seeing a
caption. A generated job may finish without being displayed; exposure records
are the evidence of what appeared on screen. Missing timestamps stay missing.

All participant answers, free text and caption strings are rendered as text. API
responses omit session credentials, internal prompts and filesystem paths. The
reader opens SQLite with `mode=ro` and never instantiates the write-capable study
store. Truncated/invalid source records produce visible warnings.

## Scope and limitations

Designed for the study's desktop/27-inch-monitor workflow. No external chart
service, asset CDN, additional package or telemetry is used. Refresh reads the
current source files; this console is not a replacement for backups or raw-data
analysis. JSONL files are append-only live sources, so records arriving during a
refresh can appear on the next refresh. SQLite queries use a consistent read
transaction. Remote researcher authentication and multi-study database discovery
are outside this first local console.

## Verification — 2026-09-14

- 25 backend/security tests passed: old/new protocol joins, submitted answer
  counts, duplicate prevention, exact frozen wording, null/N/A/zero handling,
  caption-to-video hashes, hidden credentials, safe CSV export, remote/proxy/host
  rejection, truncated source files, and consistent concurrent WAL reads.
- 33 browser checks passed: filters and QA toggle, distributions, readable raw
  text, XSS resistance, caption/event searches, control time bases, deduplicated
  timing, both exports, scheduled refresh, stale/retry behavior, persistent
  selection and desktop/laptop/narrow layouts.
- Source preservation check: all 56 existing collection JSON/JSONL/SQLite files
  matched their pre-read hashes after overview and every participant detail read.
- Repository-wide Ruff/format checks, mypy (179 source files), JS syntax and diff
  checks passed. No new package or GPU process was required.
- Actual live read: 9 non-QA participant records, 2 survey submissions and 16
  answer values; including QA gives 17 participant records, 15 submissions and
  191 values. The completed Phase 2 record in these live sources is labelled QA.
  Empty non-QA Phase 2 plots therefore reflect the collection, not a demo.

Evidence: `/tmp/concap-dashboard-ui-final-20260914/result.json` and screenshots
in that directory; live screenshots are `/tmp/concap-dashboard-live-overview.png`
and `/tmp/concap-dashboard-live-inspector-final.png`. Browser fixture results are
synthetic old/new protocol records isolated from live data. Visual review scored
94/100 against the design contract and supplied Rerun-style reference.

Changed files: `dashboard.py`, `dashboard_data.py`, `dashboard.html`,
`dashboard.css`, `dashboard.js`, `DASHBOARD_API.md`, this guide, three dashboard
test files, root `DESIGN.md`, `Makefile`, and `deploy/README.md`. Existing study
logic and collected records were preserved. The deliberate simplification is a
read-only console with three focused views and native SVG plots, without an
additional data store, charting dependency or general-purpose visualization SDK.

### Coordination update verification

The metadata/linked-view update passed 50 additional browser checks covering full
metadata disclosure, unchanged distribution groups/counts, keyboard selection,
zero/missing clocks, exact caption boundaries, cross-video exclusion, safeguards
against equating job creation with display time, clear/reset behavior, refresh
persistence and desktop/laptop layout. The original 33 dashboard browser checks
and 25 backend/security tests also pass, as do Ruff, mypy, JS syntax and diff checks.

Final linked-view evidence: `/tmp/concap-dashboard-linked-browser-qa/result.json`.
Original regression evidence: `/tmp/concap-dashboard-coordination-regression/result.json`.
Live selection preview: `/tmp/dashboard-linked-live-final.png`.
This update changes `dashboard.js`, `dashboard.css`, root `DESIGN.md`, this guide,
`tests/browser_dashboard.mjs` and `tests/browser_dashboard_linked.mjs`. It does not
change the backend, collection records or export contents. The running service
reads these assets per request; reload the browser to receive the revision.
