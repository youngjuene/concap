# Running the regeneration instrument

`dpo regen`, built to [`spec-behavior.md`](spec-behavior.md). Serves on **8779**,
one past the console's 8778 and two past the session's 8777, so all three run at
once.

## What a participant does

Four pages, six steps, one direction. §2 they watch one segment under a
**prepared** caption track; §3 they answer the ART sub-factors; §4 they mark the
five-second still; §5 they pick sources out of the separated stems; §6 the track
for the other segment is written from those two reports; §7 they watch that
segment; §8 they answer ART again plus the caption measures and the PRSS.

Which segment is prepared alternates by participant sequence number, so segment
and condition are not the same variable. Nothing in the document says which is
which — one document serves everyone.

## Try it without footage

```bash
make regen-demo          # stages synthetic media, then serves on 8779
```

Two 10 s clips, four masks and three stems each, the placeholder item set. The
stems' envelopes are generated *from* the document's own `waveform` arrays, so
what the §5 lane draws is what the lane plays.

## This study's own corpus

`scripts/stage_regen_media.py` builds §10's assets out of what the street corpus
already holds — the uncaptioned 10 s pool, the response table and the Sa2VA
run's masks — for the clips that carried sound in the earlier study (c3 and c4
have no audio stream, so nobody could report on them by ear).

The first two live in the project, at `data/corpus/videos/` and
`data/corpus/tidy_data.csv`, and the flags default there. `/data/` is gitignored,
so a fresh checkout has the code and not the footage: copy them in from the
Sa2VA working tree (`/mnt/hdd/research/2026/Sa2VA/avmask/data/`) before the
first staging run. The script says so by name if they are missing, before it
encodes anything. The mask tree stays outside — 1.7 GB of per-frame PNGs
belonging to the Sa2VA run rather than to this study — and `--masks` names it.

```bash
uv run python scripts/stage_regen_media.py --pool          # the 24 candidates
uv run python scripts/stage_regen_media.py --out data/live/regen-media
uv run python scripts/stage_regen_media.py \
  --segments amsterdam_170 singapore_303 \
  --out data/live/regen-media --document data/live/regen.json
```

The pool is filed by clip id, so re-pairing costs nothing: which clip is A and
which is B is a per-document decision, and §1 flips the condition per
participant anyway. Four things it decides, each stated in the script's own
docstring: the clip comes from the **uncaptioned** pool, because the c2
deliverables carry a burnt-in Korean caption and the instrument draws its own
band; loudness is normalised to one target, because the corpus runs from −34.0
to −14.9 LUFS and §10 asks for the two segments to match; the still and the
masks are taken at **frame 00300**, one index rather than one timestamp, so a
point is matched against the frame the participant is looking at; and sound
labels are de-confused against the AudioSet ontology, so a clip never shows
`Speech` beside `Male speech, man speaking`.

Each source carries its top-level family and the run palette's colour for it,
so the §5 lanes are coloured by family and say which family they are — a
label alone can be a narrower claim than its neighbour.

Objects whose mask is empty in the five-second frame are left out. They are in
the clip somewhere but not in the picture being marked, and declaring one would
put a label in the document that no point could reach.

## A real study

### 1. Stage the assets §10 prepares

```text
<media-dir>/
  <clip>/clip.mp4     matched encoding, resolution and loudness across the pair
  <clip>/still.png    the frame at five seconds
  <clip>/masks/*.png  one binary mask per object; the file stem is the id
  <clip>/stems/*.wav  the separated sources
```

### 2. Scaffold, then author

```bash
uv run dpo regen scaffold \
  --media-dir data/live/regen-media --out data/live/regen.json \
  --session-id street-regen --study-id street2026 --corpus-id amsterdam \
  --clips amsterdam_006 amsterdam_012 --cue-slots 4
```

The scaffold fills in every asset it can see and **does not validate**. Five
things only a researcher can supply are left for them, and `scaffold` lists them
on stdout:

| Field | Why a tool cannot write it |
|---|---|
| `prepared_track[*].text` | the study's stimulus |
| `fallback_track[*].text` | what a participant reads when §6 fails; it must be defensible on its own |
| `stems[*].waveform` | the envelope, computed from the audio |
| `stems[*].gain` | measured against the original mix (§5) |
| `stems[*].colour` | which sources must stay apart on one screen |

Cue timings are scaffolded evenly across the clip. Even spacing is the one
arrangement that encodes no assumption about where the interesting sound is;
move them deliberately.

### 3. Author the items

The shipped set (`src/dpo/regen/items/default.json`) is **placeholder wording**
in the right shape. Copy it, replace the wording with the study's validated
items, set `"provenance": "authored"`, and pass `--items`. The provenance and a
digest of the wording go on every response row, so a pilot on placeholders can
never be mistaken for the study.

The ART block is defined once and asked by both survey pages. Do not duplicate
it for §8 — the change between the two askings is the study's main measure, and
two copies would drift.

### 4. Validate

```bash
uv run dpo regen validate --session data/live/regen.json --items data/live/items.json
```

Refuses on the first bad path and names it. `status: valid` reports the config
hash, the cue-slot count and the item provenance.

### 5. Serve

```bash
uv run dpo regen serve \
  --session data/live/regen.json --media-dir data/live/regen-media \
  --out data/live/regen-responses --items data/live/items.json --writer template
```

With a GPU and the study's model:

```bash
CUDA_VISIBLE_DEVICES=1 uv run dpo regen serve … \
  --writer gemma --backend-config configs/backends/gemma4-audio.toml
```

`--writer gemma` is held to the same backend pin as the pipeline and refuses a
config the contract does not pin, on any machine, before it touches CUDA.

While §6 runs, the console draws a bar over the cues in the track — the same
report the participant's waiting screen is polling, so the two cannot disagree
about where the model has got to. It is drawn only when a terminal is watching:
redirect the console to a file and you get the JSON lines and no control codes.

**The model is given `audio.wav`, not the clip.** transformers decodes an
`.mp4` or `.webm` only with `torchcodec`, and no published torchcodec works
against this project's torch: every wheel links `torch_from_blob`, which
`torch 2.10.0+cu126` renamed to `torch_create_tensor_from_blob`, so it installs
and then dies at the first decode. Staging writes a mono 16 kHz wav beside each
clip — after loudness correction, so the model hears what the participant hears
— and the document names it. A document from before this carries no `audio`
field and is refused on load rather than falling back four slots into a
session.

**Check the first regeneration of a session actually came from the model.**
Whatever the cause, a `--writer gemma` run that cannot read its audio is not
loud on the participant's screen: they get the fallback track and no notice.
It is loud in the log — `fallback: true` with a `fallback_reason` — so one
pilot run and a look at `regeneration.written` settles it before a study spends
sessions on template captions.

Launch the participant's browser with
`--autoplay-policy=no-user-gesture-required`; the viewing screens go fullscreen
on a click, but the clip must start without a second gesture.

## What lands on disk

Under `--out`, per participant:

| File | What it is |
|---|---|
| `viewings-<p>.jsonl` | **the analysis unit** — one row per viewing: key, index, condition, segment, clip, the full caption text with timings, playback start and end |
| `responses-<p>.jsonl` | one row per survey submission, carrying the `view_id` it is about, plus the item digest and provenance |
| `events-<p>.jsonl` | the full stream: every step, every point placed, moved and removed, every lane played, the regeneration with its prompt and raw output |
| `snapshot-<p>.json` | resumable state, replaced atomically |
| `roster.json` | participant identifier → sequence number |
| `captions.json` | the caption cache |

Join `responses.view_id` to `viewings.view_id`. Two viewing rows and two
response rows per completed session.

Every line carries `config_hash`, applied on the server from the configuration
in force — never from anything a browser sent.

## Things that will look like bugs and are not

**A step returns 409.** The steps run in order and neither go back nor skip
(§9.1). The body carries the step the session is actually on; the page follows
the server. A reload and a typed URL in the same tab both land on the screen
the session is actually on.

**A new tab is a new participant, not the same one resumed.** The page keeps
its identifier in `sessionStorage`, which is per tab: opening a second tab
enrols afresh and spends the next sequence number, so §1's alternation runs one
step out for everyone after it. This is the right trade for a kiosk — an
identifier outliving the tab would sit participant two inside participant one's
session — but it means *don't open a second tab mid-run*. One that was opened
is identifiable in `roster.json`: an entry with no `viewings-<p>.jsonl` against
it. Exclude it and read the sequence numbers around it as they stand.

**`/api/step/art` is refused before the first viewing ends.** The gate is on
reading a step as well as submitting it.

**The visual response never says what a point hit.** §4 matches once, on
submit. Telling the participant would turn the task into hunting for a mask.
The matched labels are in `events-<p>.jsonl`.

**A second `POST /api/regenerate` returns `cached: true`.** §6 runs once per
participant. A reload during the wait must not spend the model again, and must
not hand the participant a second, different track their §8 answers are about.

**`fallback: true`.** The model took longer than the ceiling, returned
something that failed validation, or raised. The reason is on the
`regeneration.written` event. Those viewings are not generated viewings —
separate them in the analysis.

## Archiving it

`rm -rf src/dpo/regen tests/regen docs/v3-regen src/dpo/cli/regen.py`, drop the
`register_regen` line in `src/dpo/cli/__init__.py` and the `regen-demo` target.
Nothing in `dpo.caption` or the other instruments refers to it;
`tests/caption/test_detachment.py` asserts that, and keeps passing afterwards.
