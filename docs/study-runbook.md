# Street-audio study runbook

Operational state for `configs/study/street-audio.toml` — the exact artifact
ids and the command order from annotation to the human-study export. Ids are
content-addressed: they stay valid until the corpus or a candidate pool is
deliberately regenerated, and every command below refuses an id minted under a
different contract slice.

## Fixed inputs

| artifact | id |
| --- | --- |
| clip registry (48 clips; train 27 / validation 7 / test 7 / study 7) | `sha256:ca6daf0e52c80d753e6f2c774bfb20355b6f610fb4f9c60ce5d4deab70c06c18` |
| audio candidate pool, train (108 candidates / 162 pairs) | `sha256:e0d01e87e23e1899bd0216b7f3ac6d6459284b47bf7a061471b8a6d736b5fe41` |
| audio candidate pool, validation (28 / 42) | `sha256:7c8c8e3861975f9b341fb0bd9e1e8febb210d6b09000a27d5f1873b8676ecfb1` |

Sessions already exported: `data/annotation/tasks-{train,validation}.json`
(191 and 51 tasks). `answers-*.json` holds the attention-check keys — restricted,
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
`min_agreement = 0.6` means unanimity: the inter-rater agreement rate is the
retention rate, exactly.

## 2. Train, select, lock (one 3090)

```bash
uv run dpo views derive --workspace artifacts/street --contract configs/study/street-audio.toml \
  --artifact-id <registry> --artifact-id <train-pool> --artifact-id <validation-pool> \
  --artifact-id <train-annotations> --artifact-id <validation-annotations> --track audio

uv run dpo train run --workspace artifacts/street --contract configs/study/street-audio.toml \
  --artifact-id <view-ids...> --checkpoint-dir runs/checkpoints \
  --backend-config configs/gemma4/e4b-audio.toml --media-dir data/live/media

uv run dpo select run --workspace artifacts/street --contract configs/study/street-audio.toml \
  --artifact-id <view-ids...> --artifact-id <cell-ids...> \
  --checkpoint-dir runs/checkpoints --backend-config configs/gemma4/e4b-audio.toml \
  --media-dir data/live/media
```

`make report` prints per-variant validation accuracy, the selection ranking,
and the lock as they publish. Training is resumable: rerunning skips finished
cells.

## 3. Export the human-study captions

```bash
uv run dpo study export --workspace artifacts/street --contract configs/study/street-audio.toml \
  --artifact-id <lock> --artifact-id <selection-report> --artifact-id <validation-report> \
  --artifact-id <registry> --artifact-id <train-pool> \
  --track audio --checkpoint-dir runs/checkpoints \
  --backend-config configs/gemma4/e4b-audio.toml --media-dir data/live/media
```

Requires the lock (configuration freezes before held-out access), opens the
one-per-lock `human-study` capability, verifies the staged media against the
registry's derivative hashes, and refuses to publish if any caption
byte-matches a frozen training candidate.

## Known cautions

- `dpo artifact gc --execute` would delete the pools: GC roots are locks and
  reports, and none exist until step 2 publishes them. Run gc only after the
  lock exists.
- The train pool shares caption text across clips (73 distinct of 108; one
  caption covers 10 clips) — expect annotator disagreement to concentrate
  there.
