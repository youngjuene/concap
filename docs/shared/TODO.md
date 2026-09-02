# Open items

Deferred deliberately, not forgotten. What is built and has no production
caller belongs here with the reason. Entries leave this file when they are
resolved, and what they said goes into the commit message that resolved them,
so the record lives in `git log` rather than accumulating here. Revisit
alongside `docs/shared/study-runbook.md`.

## 1. The orphaned visual pools — one decision, still unmade

Two frozen candidate pools in `artifacts/street/` are on the visual track:
`238ad3d4…` (train) and `8e7981fa…` (validation). They predate the decision to
declare `[tracks.audio]` alone, so nothing on the live contract can read,
dedup, or export them — `candidates generate --track visual` is refused
outright — and unlike the eight audio pools they carry no `dedup-source` edge,
so they have never been through the cross-split near-duplicate pass.

One decision, not two:

* restore `[tracks.visual]` and `[backends.visual]` from git history (the
  header of `configs/study/street-audio.toml` names the exact prompt hash and
  `video_frames`), put both pools through `dpo candidates dedup`, and treat the
  visual arm as collected; **or**
* drop them at the next `dpo artifact gc`.

Leaving them is the option that later reads as a second track having been
collected. Nothing downstream is blocked either way — no training artifact
exists on either track — so this is a claim-hygiene decision, not a wiring one.

## 2. wDPO scalar maps — blocks reporting, not wiring

`objectives/wdpo.py` carries explicit, documented placeholders where
`arxiv_2603.07211v1` leaves the maps unspecified, and its `code_commit` is
unpinned. Re-verify against the pinned official code before reporting any wDPO
number — or report wDPO as excluded, with this as the stated reason.

## 3. Candidate pools need regenerating before annotation

Not a defect: a consequence, recorded so it is not rediscovered. Removing the
three dead `[tracks]` knobs changed the `tracks` section, and the `candidates`
stage declares that section in its contract slice, so the ten frozen pools in
`artifacts/street/` are keyed under a slice the contract no longer produces.
The corpus stages declare `("corpus",)` only, so the 48 registry shards, the
registry, and the corpus ingest are untouched.

Before anyone annotates, re-run the three commands that stand between the
registry and a task file — `candidates generate` (per split, audio track),
`candidates dedup`, `annotation export-tasks` — and the 212 exported tasks in
`data/annotation/` are replaced. No human work is lost: `data/annotation/responses/`
is empty, which is why the knobs were removed now rather than later.

## 4. `annotation ingest` does not refuse a duplicate annotation id

Recorded because a check that stated it was deleted, not because the check was
load-bearing. `raw_annotations.load_annotations` parsed a JSONL export of the
raw store and refused a repeated `annotation_id` ("the raw store is append-only
and ids are unique"). Nothing in `src/` ever called it — the live path is
`annotation ingest` → `annotations_from_responses` → `ingest_annotations` — so
it went with the other uncalled helpers.

The intent it carried has no other home. `reliability.py` keys a dict by
`annotation_id`, so duplicates collapse there silently, while the raw
annotations artifact keeps both rows. Two `--responses` files naming one
annotation twice would therefore publish a raw store the reliability report
disagrees with, and nothing would say so.

Whose call: the researcher, before the collection round. Either ingest refuses
a repeated id across its `--responses` inputs, or the analysis states that it
de-duplicates and the raw artifact does not.
