# Street-audio study runbook

Operational state for `configs/study/street-audio.toml` — the exact artifact
ids and the command order from annotation through training and selection to
the human study itself. Ids are
content-addressed: they stay valid until the corpus or a candidate pool is
deliberately regenerated, and every command below refuses an id minted under a
different contract slice.

## Fixed inputs

| artifact | id |
| --- | --- |
| clip registry (48 clips; train 27 / validation 7 / test 7 / study 7) | `sha256:ca6daf0e52c80d753e6f2c774bfb20355b6f610fb4f9c60ce5d4deab70c06c18` |
| audio candidate pool, train — deduped (104 candidates / 150 pairs) | `sha256:b0cdf7b437659a26bad749c021d2cbb0950c060f77b7d77bfac936ee1523e817` |
| audio candidate pool, validation — deduped (24 / 30) | `sha256:12929dba2e67ceb63ab5ed93c73f3ea1b47a007f0f2c5c39fcfb15cdf310fe2f` |

**The two pool ids above are stale.** Dropping the three dead `[tracks]`
knobs (`afdf381`) re-keyed the contract and the `candidates` slice, so
`annotation ingest` refuses both pools as minted under a different contract —
and the task files exported from them would be rated for nothing. Regenerate
the pools first (`docs/shared/TODO.md` §3), export new task files, and
replace the ids here before anyone annotates.

Sessions already exported from the deduped pools:
`data/annotation/tasks-{train,validation}.json` (176 and 36 tasks). The
pre-dedup pools (`e0d01e87…`, `7c8c8e38…`) remain in the store as the new
pools' `dedup-source` parents; do not annotate against them — the leakage
gate refuses views derived from pools with cross-split near-duplicates, and
`dpo candidates dedup` printed `leakage_audit_passed: true` for the pair
listed above. `answers-*.json` holds the attention-check keys — restricted,
never inside the served `--media-dir`.

## 1. Annotate (both authors, per split)

```bash
make annotate SPLIT=train        # serves tasks-train.json on data/live/media
# each annotator saves through the UI, then:
uv run dpo annotation ingest --workspace artifacts/street \
  --contract configs/study/street-audio.toml \
  --artifact-id <pool-id-for-the-split> --split train \
  --tasks data/annotation/tasks-train.json \
  --answers data/annotation/answers-train.json \
  --responses data/annotation/responses/responses-<annotator-a>.json \
  --responses data/annotation/responses/responses-<annotator-b>.json
```

Repeat with `SPLIT=validation` and the validation pool id. With two raters,
`judgments_per_pair = 2` is the ceiling rather than a choice, and
`min_agreement = 0.6` therefore means unanimity: the inter-rater agreement
rate is the retention rate, exactly. The budget is 212 tasks per author
(176 train + 36 validation), about 1.5 h each at 25 s/task.

Ingest publishes `dpo.reliability-report/v2` alongside the annotations —
per-annotator attention pass rate, repeat consistency, left-choice rate with
its exact binomial p, and chance-corrected agreement (Krippendorff alpha).
Exclusion needs a position lean that is both larger than `max_position_bias`
(0.2) and significant at `position_bias_alpha` (0.01); with two expert
annotators, dropping one ends the study, so read this report before
proceeding rather than after.

## 2. Train, select, lock (one 3090)

```bash
uv run dpo views derive --workspace artifacts/street --contract configs/study/street-audio.toml \
  --artifact-id <registry> --artifact-id <train-pool> --artifact-id <validation-pool> \
  --artifact-id <train-annotations> --artifact-id <validation-annotations> --track audio

uv run dpo train run --workspace artifacts/street --contract configs/study/street-audio.toml \
  --artifact-id <view-ids...> --artifact-id <flip-manifest-ids...> \
  --checkpoint-dir runs/checkpoints \
  --backend-config configs/gemma4/e4b-audio.toml --media-dir data/live/media

uv run dpo select run --workspace artifacts/street --contract configs/study/street-audio.toml \
  --artifact-id <view-ids...> --artifact-id <cell-ids...> \
  --checkpoint-dir runs/checkpoints --backend-config configs/gemma4/e4b-audio.toml \
  --media-dir data/live/media

uv run dpo report analyze --workspace artifacts/street --contract configs/study/street-audio.toml \
  --artifact-id <validation-report> --artifact-id <selection-report>
```

`views derive` prints the four view ids and, under `flip_manifests`, one id
per `[robustness].flip_rates` entry; `train run` needs all of them. The
matrix is 9 base cells plus **21 flipped retrainings** — each of the seven
pair_strict preference arms (DPO, IPO, CDPO, RDPO, DRDPO, WDPO, SFT_DPO) once
per positive rate (0.1, 0.2, 0.3) on the shared manifest — so budget roughly
3.3x the base training time. Selection scores all 30 cells but ranks and
locks the 9 base cells only. Training is resumable: rerunning skips finished
cells, flipped ones included.

`make report` prints per-variant validation accuracy, the selection ranking,
and the lock as they publish; `report analyze` publishes
`dpo.analysis-report/v1` — clip-clustered CIs at 10000 resamples
(`validation.bootstrap_samples`), paired tests vs SEED with BH correction,
Bradley-Terry, the natural-noise slices, and the flip-rate curve of every
retrained arm. It re-scores nothing: a rerun republishes to the same id.

## 3. Export the human-study stimuli

```bash
uv run dpo study export --workspace artifacts/street --contract configs/study/street-audio.toml \
  --artifact-id <lock> --artifact-id <selection-report> --artifact-id <validation-report> \
  --artifact-id <registry> --artifact-id <train-pool> \
  --track audio --checkpoint-dir runs/checkpoints \
  --backend-config configs/gemma4/e4b-audio.toml --media-dir data/live/media
```

Captions the study split with the top-ranked experiment's selected variant and
publishes `dpo.study-export/v1`: one *congruency ladder* per clip,
`[study].rungs` (5) captions ordered along the measured axis
`[logP(c|audio,video) - logP(c|audio,gray)] / |c|`. Budget about 143 s per
clip on the 3090 (33 s generating, 109 s scoring, 22.5 GiB peak) — roughly 17
minutes for the seven study clips, once.

Requires the lock (configuration freezes before held-out access), opens the
one-per-lock `human-study` capability, verifies the staged media against the
registry's derivative hashes, and refuses to publish if any caption
byte-matches a frozen training candidate. Two further refusals are specific to
the ladder: a non-monotone ladder is rejected at publish, and one whose ends
differ by less than `MIN_CONGRUENCY_SPAN` (0.02 nats/token) at selection. Both
mean the run has to be retried with more candidate spread, not overridden.

## 4. Run the study

`study serve` takes the export **document**, not an artifact id, and no command
dumps one — read it out of the store at the id step 3 printed:

```bash
cp artifacts/street/<export-id-without-the-sha256:-prefix>/payload.bin study-export.json

uv run dpo study serve --export study-export.json \
  --media-dir data/live/media --out data/userstudy/responses
```

Serves the published export at `127.0.0.1:8776`: per clip, the participant
watches with sound and writes what they heard, then gets the sentence and the
slider and rates the match. Responses are written as
`responses-<participant>.json` under `--out` in schema
`dpo.userstudy-responses/v2` — plain files, the append-only record; step 5
publishes them.

This step needs **video with sound**, which the corpus staging does not
produce — `--track both` mutes every `.mp4` it writes. Stage the unmuted
renders into an `unmuted_video/` subdirectory of the media dir first:

```bash
uv run python scripts/stage_media.py --track audio --media-source source \
  --audio-presentation unmuted_video --out data/live/media/unmuted_video \
  --condition-dir <condition-dir> --source-dir <pristine-sources> \
  --rows data/live/unmuted-rows.jsonl
```

The rows file is a by-product here, not an input: the registry is already
locked, and these renders exist only to be served.

The app looks in `<media-dir>/unmuted_video` first and falls back to
`<media-dir>` itself, so a missing subdirectory does not 404 — it serves the
corpus's **muted** render and the participant is asked to match a caption
against silence. Check one clip has sound before recruiting anyone.

## 5. Ingest the responses

```bash
uv run dpo study ingest --workspace artifacts/street --contract configs/study/street-audio.toml \
  --artifact-id <study-export> \
  --responses data/userstudy/responses/responses-P01.json \
  --responses data/userstudy/responses/responses-P02.json   # one per participant
```

Checks every response against the export (its clip, the caption at the rung it
claims, the rung's position, the rating scale, one answer per clip; the same
participant in two files is refused) and publishes `dpo.study-responses/v1`
(the validated rows, participants hashed) and `dpo.study-results/v1` (placement
and match rating on the measured axis with clip- and participant-clustered
intervals at `validation.bootstrap_samples`, the congruency–rating correlation,
order effects, per-clip and per-participant tables). Rerunning with the same
files republishes the same ids; adding a participant publishes a new pair with
the old export as the shared ancestor.

## Known cautions

- `dpo artifact gc --execute` would delete the pools: GC roots are locks and
  reports, and none exist until step 2 publishes them. Run gc only after the
  lock exists.
- The train pool shares caption text across clips — in the deduped pool now in
  use, 71 distinct texts across 104 candidates, and one caption covers 9 clips
  (validation: 19 of 24, max 2). Expect annotator disagreement to concentrate
  there. Dedup cut cross-split collisions only; this is within-split repetition
  and is out of the gate's scope by design.
- This contract declares **only `[tracks.audio]`**, deliberately: without the
  block, `candidates generate --track visual` fails rather than producing
  captions nobody will annotate. The visual pools still in the workspace
  (`238ad3d4…`, `8e7981fa…`) are orphans from an earlier two-track round —
  undeduped and unusable here. Do not treat them as a second track's data.
- Participants must stay blind: the served study document is narrowed to each
  rung's position and text. Do not hand anyone the raw
  `dpo.study-export/v1` payload — it carries the winning arm, its validation
  accuracy, and every rung's measured congruency.
