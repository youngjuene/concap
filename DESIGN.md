# Design

## Source of truth
- Status: Active. Refreshed: 2026-09-14.
- Participant study: [regen design contract](src/dpo/regen/DESIGN.md).
- Researcher dashboard: the contract below; source records in `regen/log.py`,
  `regen/study_store.py`, `regen/study_instrument.py` and `regen/study_api.py`.
- Visual reference: [Rerun](https://rerun.io), specifically compact panels and
  time-aligned inspection. This app needs no Rerun dependency or 3D workspace.

## Brand
Quiet, minimal research software. Near-white canvas, white panels, charcoal text,
fine gray dividers, blue for Phase 1/acoustic detail and teal for Phase 2/context.
Trust comes from explicit sample counts, units, source versions and missingness.

## Product goals
Review collection progress, survey distributions, control changes, caption text
and generation timing. Trace aggregate findings to individual records. Preserve
all source data. No participant editing/deletion, recruitment or model control.
Success is accurate counts, useful filtering and quick movement from a survey
result to its participant and video timeline.

## Personas and jobs
Researcher inspecting collected data at a desktop/27-inch monitor. Needs to see
incomplete sessions, compare recorded answers and inspect interaction timing
without reading SQLite or JSONL manually.

## Information architecture
Three views: Overview, Responses, Session inspector. Global filters for QA,
language and participant search; distributions split by phase, instrument version,
page/video and item. Inspector links participant, response rows, steering trend,
caption exposures and event log. Export normalized responses as CSV or session JSON.

The dashboard keeps its established palette and overall sizing. Question cards
show the question, phase/video/condition and sample counts; full instrument,
flow and denominator explanations sit behind labelled details. Raw tables keep
answers and sentences prominent, with IDs and timing provenance expandable.

The session inspector pairs one compact control timeline with a selected-moment
summary, then places caption and event records beside each other. Choosing a
control, caption or event focuses its phase/video and nearby records. Playback
positions link only to playback windows; recorded timestamps link only to recorded
event times. A caption job's creation time is never treated as its display time.
Missing clocks are explained and leave the corresponding records unfiltered.
The focus is explicitly clearable, survives refresh, and clears on participant or
video changes. Timing distributions and submitted responses remain available in
expandable sections. Keyboard users can select the same moments as pointer users.

## Design principles
Overview first, detail on selection. Separate model generation duration from
elapsed time until an exposure; do not infer exact input-to-visible latency when
no comparable clock exists. Missing is not zero, N/A is not missing, old and new
instruments must not silently combine. Distinguish generated jobs from displayed
captions, authored fallbacks and superseded/unfinished jobs.

## Visual language
System sans text and monospace IDs/timestamps. 8px rhythm, subtle 8px radii,
compact tables and restrained SVG plots. Stable panel sizes; no ornamental charts,
big gradients, animated counters or required external fonts.

## Components
Reuse native form controls, buttons and accessible tables. Dashboard-specific CSS
owns its layout; no changes to participant theme. Compact metric tiles, distribution
bars with counts, axis trend plot, caption rows and sortable/filterable event table.

## Accessibility
Visible keyboard focus, labelled filters, readable contrast, textual equivalents
for plots, counts beyond color. Respect reduced motion. Escape user-supplied text;
render comments and generated sentences as text, never markup.

## Responsive behavior
Desktop first at 1920×1080 and 1366×768. Participant list beside the inspector;
stack on narrower screens. Tables may scroll within their panel. No page overflow.

## Interaction states
Initial loading and refresh states; empty collection and no matching filters;
partial-source warnings; errors retain the previous result with a stale indicator.
Optional 15-second refresh, paused while inspecting if unchecked. Selection and
filters survive refresh. No fake sample rows in the real dashboard.

## Content voice
Plain researcher labels: Phase 1, Phase 2, Submitted, In progress, Generated,
Prepared fallback, Generation time (ms), Playback position (s), Recorded at (UTC).
Survey item text is from stored snapshots where available; otherwise show the
recorded item ID and explicitly indicate historical wording is unavailable.

## Implementation constraints
Existing FastAPI/Python, SQLite read-only transactions, JSONL reads, vanilla JS/CSS
and SVG. No new packages. Serve separately on localhost:8781, reject proxy/remote
access; never mount under public Funnel. Do not emit session credentials, model
prompts or unrelated filesystem paths. Tests reconcile fixtures across protocol
versions and prove data files remain unchanged, with browser rendering/interaction QA.

## Open questions
- Remote authenticated researcher access is outside this first local dashboard.
- Historical records may lack item wording or wall-clock exposure timestamps;
  surface these limitations without synthesizing measurements.
