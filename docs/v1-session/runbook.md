# Caption session — operator runbook

The caption session is the participant instrument specified in
[`docs/v1-session/spec-behavior.md`](spec-behavior.md)
(behavior) and [`docs/v1-session/spec-identity.md`](spec-identity.md) (identity): a participant
watches a street-scene clip, lists the sounds they noticed, shapes each shot's
caption on the skeleton, watches the clip again with their captions placed, and
answers a few items; on a later day they complete a short follow-up in a
browser. It lives in `dpo.session` and is driven by `dpo session`.

It is a separate instrument from `dpo study serve` (the congruency slider) and
`dpo annotation serve` (pairwise preferences): different document, different
responses, nothing shared but the media directory convention.

## 1. Author the session document

The document is the configuration. Clip order is its list order; per clip it
sets whether the first viewing carries captions, shaped or control, the balance
control (`orderings` or `crossfader`), and the opening settings; per shot it
carries the sources the skeleton draws, each with its two phrases (words, never
numbers), its two weights (never sent to the browser), and its role. The
schema is spelled out in `tests/session/fixtures/session.json`, which is a
complete six-clip example.

Start from a scaffold over the clips you have media for:

```bash
uv run dpo session scaffold --media-dir data/live/media --out data/session/session.json
# or name the clips explicitly, in session order:
uv run dpo session scaffold --media-dir data/live/media --out data/session/session.json \
  --clips clip_017 clip_042 clip_003 clip_058 clip_021 clip_009
```

The scaffold writes one shot per clip (end from `ffprobe`, 10 000 ms if it
cannot be read), alternates shaped and control, and leaves `sources`, the
scene and atmosphere rows, the automatic captions, and the follow-up sound
lists EMPTY. Fill them in, then validate — the validator refuses a scaffold
until it is authored, and names the offending path when it does:

```bash
uv run dpo session validate --session data/session/session.json
# {"status": "invalid", "error": "clips[0].shots[0].automatic_caption: must not be empty", ...}
```

Rules worth knowing while authoring: shots are contiguous from 0; one to
eight sources per shot; tokens are uppercase without brackets (the interface
draws them); phrases carry no digits; roles come from `role_heads.audio` on a
shaped clip and from the clip's own three `visual_roles` on a control clip;
`followup.recognition` covers every clip and `followup.check` covers every
shaped clip; the copy fields (`poles`, `role_heads`, the attention and
criterion items, `open_item`) must match spec Table 6 verbatim.

### Derive parameter evidence from Sa2VA masks

`session link-masks` turns the Sa2VA mask tree into a reviewable
`dpo.caption-mask-link/v1` manifest. It supports both propagated masks at
`{branch}/{clip}/{label}/{frame}.png` and the one-frame-per-second release
layout at `{branch}/{clip}/{label}_{second}.png`:

```bash
uv run dpo session link-masks \
  --mask-root /mnt/hdd/research/2026/Sa2VA/avmask/runs/60fps-windowed/masks \
  --tidy-data /mnt/hdd/research/2026/Sa2VA/avmask/data/tidy_data.csv \
  --ontology /mnt/hdd/research/2026/Sa2VA/avmask/data/ontology.json \
  --out data/session/sa2va-mask-links.json
```

Add `--clips amsterdam_006 singapore_303` for a bounded calibration pass. The
manifest gives every source a `session_source` object with the same `id`,
`token`, `prose`, `phrases`, `weights`, and `role` fields the session document
expects, beside the raw evidence each was derived from.

Hand the manifest to `scaffold` and the sources arrive in the document already
filled — the audio branch on a shaped clip, the visual branch on a control
clip, and a visual category whose mask is empty in every frame left out:

```bash
uv run dpo session scaffold --media-dir data/live/media \
  --out data/session/session.json \
  --mask-links data/session/sa2va-mask-links.json
# {"status": "scaffolded", "clips": 6, "sources": 34, ...}
```

That scaffold still does not validate, and that is the point. Review every
field the manifest names in `review_required`, and expect the validator to
stop on the role first:

```bash
uv run dpo session validate --session data/session/session.json
# {"status": "invalid", "error": "clips[0].shots[0].sources[0].role: must be one of
#  ['underneath', 'stands_out', 'only_here']; it is unset because no mask measures it", ...}
```

An audio source arrives with `role: null` deliberately. The role heads say how
a sound sits in the soundscape, and a grounding mask does not measure that: an
earlier version guessed one from mask presence and called a siren *underneath*
because something siren-shaped stayed in frame. Visual roles do come through
set, because there the role is a property of the fixed category, not of the
mask. The `parameters.detail` block carries the evidence to annotate from.

The mapping is intentionally explicit:

- `final_labels` names the audio mask prompts and admission candidates.
  Repeated labels provide a normalized Ear-side prior.
- `top_level_parent_name` is an aggregate family/count constraint, not a
  label-by-label mapping. The AudioSet ontology resolves each fine label, and
  the CSV counts resolve DAG ambiguities such as Bell belonging to either
  Music or Sounds of things.
- Audio-mask presence, foreground area, and normalized bounding-box
  coordinates provide the Eye-side visibility evidence. An empty mask does
  not remove an audio admission candidate because off-screen sounds remain
  valid.
- The eight fixed visual categories provide control-task candidates. Mask
  bottom coordinate, area, and persistence produce Near/Far weights.
- Union bounding boxes, mean box centers, frame extents, timestamps, and the
  raw evidence behind every score remain in the manifest for audit.

Sa2VA audio masks ground a *possible visible source* of a tagged sound; they
do not measure loudness or acoustic onset. Consequently the suggested audio
temporal phrase and Ear weight remain researcher-calibration fields, and the
role is not suggested at all. Mask coordinates are preprocessing evidence and
never become participant stage overlays.

### What the derived weights do to the balance control

Read the orderings a shot offers before recruiting on it. Balance is a choice
among the distinct rankings the admitted sources take between the two poles,
so the *number* of those rankings is the resolution of the control, and mask
weights set it. On the two clips measured on 2026-09-01:

| Clip | Branch | Sources | Orderings |
|---|---|---|---|
| `amsterdam_006` | audio | 4 | 2 |
| `singapore_303` | audio | 2 | 2 |
| `amsterdam_006` | visual | 8 | 8 (16 before collapsing) |
| `singapore_303` | visual | 8 | 8 (17 before collapsing) |

Two things to know about that table. The audio side is coarse because the Ear
weight is `final_labels` multiplicity, which is usually 1 and quantizes to
`{0.5, 1.0}`, so most pairs never cross and the crossfader is close to a
two-position switch. That is honest about what the data supports, and it is
the ceiling until a salience measure with more resolution than a tag count
exists.

The visual side is the opposite problem. Eight sources cross into as many as
twenty-nine rankings, several separated by less than a thousandth of the axis.
`skeleton.MIN_SPAN` and `skeleton.MAX_ORDERINGS` collapse the ones a
participant could not aim at into their neighbours, keeping the widest and
always keeping the two endpoints, so the columns the kiosk draws at equal
width are all rankings that hold a real interval. The raw partition is still
available (`orderings(..., min_span=0.0)`) if you want to see what was merged.

## 2. Stage the media

The app serves `<media-dir>/unmuted_video/<clip_id>.mp4` first and falls back
to `<media-dir>/<clip_id>.mp4`. It must be video WITH sound: the participant
lists and shapes what they hear.

Real footage: stage the unmuted renders into `data/live/media/unmuted_video`
exactly as [`docs/shared/study-runbook.md` §4](../shared/study-runbook.md#4-run-the-study)
describes for the slider study:

```bash
uv run python scripts/stage_media.py --track audio --media-source source \
  --audio-presentation unmuted_video --out data/live/media/unmuted_video \
  --condition-dir <condition-dir> --source-dir <pristine-sources> \
  --rows data/live/unmuted-rows.jsonl
```

Demo (no footage): synthetic clips for every clip in a document, with distinct
test patterns and distinct tones so the follow-up's "which clip is this sound
from?" is answerable:

```bash
uv run python scripts/stage_session_demo.py \
  --session tests/session/fixtures/session.json --out data/session-demo/media
```

Both are idempotent. `make session-demo` runs the demo staging and then serves
the fixture (step 3) in one go. `--container webm` writes VP9 + Opus `.webm`
instead of H.264 + AAC `.mp4` (the app serves either suffix): the open-source
Chromium that Playwright ships decodes no H.264 or AAC, so a headless walk of
the pages needs the webm staging, while the tablet plays the mp4.

## 3. Serve the kiosk

```bash
uv run dpo session serve --session data/session/session.json \
  --media-dir data/live/media --out data/session/responses
```

Open **`http://127.0.0.1:8777/?participant=P01`** in the kiosk browser and put
it in fullscreen (Begin also requests fullscreen). The participant identifier
comes from the URL; without one the page shows a one-line message and nothing
else. Every interaction is autosaved, and reopening the same URL resumes from
the start of the interrupted step.

On a screen taller than the tablet's 820 (a desk monitor during rehearsal or
an audit) the page zooms its composition to the screen's height, so type and
targets keep their proportions instead of sitting small in the middle, and
the stage takes whatever width that leaves beyond the working column's 492
(rows beside eight ordering columns). On a 27-inch 2560×1440 the stage is
about 910 wide in the page's units against the tablet's 640. The tablet is
untouched, and the follow-up's check frame is still laid out at the tablet's
640, so a caption shaped on a wider kiosk can break across lines differently
there.

Launch the kiosk browser with autoplay allowed, so a resume into a viewing
starts the clip with sound without a tap:

```bash
chromium --kiosk --autoplay-policy=no-user-gesture-required \
  'http://127.0.0.1:8777/?participant=P01'
```

Without the flag a browser may refuse to start unmuted footage on a resume;
the page then leaves the frame focused, and the first tap on it (or Enter)
starts the viewing and is logged as `viewing.play`.

Before recruiting, check one clip has sound in the kiosk browser and that
`/api/inventory/<clip>?participant=P00` answers 403 until a listing is
submitted.

### The writer

`--writer template` (default) assembles deterministic sentence prose from the
skeleton's list: itemized captions mention each admitted source with its
temporal (or motion) phrase in order; grouped captions write one clause per
role head; scene and atmospheric return the authored rows. It needs no GPU and
is what the tests, the demo, and a dry run use. The caption's two reserved
lines at the kiosk's stage width hold about `CAPTION_MAX_CHARS` (96)
characters, and a longer caption clips behind an ellipsis in both positions.
The template writer stays within that budget where it can: when the phrased
itemized sentence would overflow it drops the phrases and keeps the sources in
order, and when a led grouped sentence would overflow it drops the head leads
and keeps the clauses in order (very long authored prose can still exceed it).
The Gemma instruction asks for the same budget and the writer enforces it in
three steps, each measured on the base E4B over the worst cases (four or five
entries): with four or more entries the first draft is told to name most of
them without notes, which fit two of six cases outright; a draft that still
overruns is requested once more as a comma list with at most two words of
notes per entry, which fit every remaining case but one while keeping every
entry in order; and what still overruns becomes the template writer's sentence
for the same list, so a caption never silently drops an admitted source
(a sentence cut is the last resort only if even that overruns). The authored captions (`automatic_caption`,
`scene.prose`, `atmosphere.prose`) are held to the budget by the validator,
since they are placed or returned verbatim.

`--writer gemma` conditions the study's Gemma 4 E4B on the shot's audio and
tells it exactly which sources to mention in which order (`GEMMA_INSTRUCTION`
and `GEMMA_LEVEL_RULES` in `dpo/session/writer.py`, decoding frozen at
temperature 0, seed 0). On a control clip the writer looks instead of
listening: it is shown one frame from the middle of the shot and asked for a
plain visual description of the listed things (`GEMMA_VISUAL_INSTRUCTION`),
so a visual skeleton never yields a sentence about what "is heard". It takes the audio backend config, the study contract
for `[tracks.audio]`, and optionally a LoRA checkpoint directory:

```bash
uv run dpo session serve --session data/session/session.json \
  --media-dir data/live/media --out data/session/responses \
  --writer gemma --backend-config configs/gemma4/e4b-audio.toml \
  --contract configs/study/street-audio.toml \
  --checkpoint checkpoints/street/<cell-directory>
```

Without a CUDA device it exits 3 (`blocked_pending_external_operation`) before
loading anything. Measured on one RTX 3090 with the unquantized base model
(`--checkpoint` omitted, 2026-08-28): 15.5 GiB resident, about 45 s to load,
then every audition for the six-clip fixture (86 captions) written in under
two minutes; a caption request answers in 0.6–1.6 s and a repeat from the
cache in about 10 ms. The model follows the skeleton — leaning the list toward
the ear puts the siren first — and on a real Amsterdam clip it names what is
there (a car engine, traffic, footsteps). Read a few captions of your own
document before recruiting anyway: the model is asked to mention only the
listed sources, and what it says about them is its own.

Either writer sits behind a cache keyed by `clip/shot/settings`: identical
settings return the identical caption, every single-source and single-head
audition is written at start-up, and a restart over the same `--out` writes
nothing again.

## 4. The follow-up, on a later day

Serve the same document, media, and `--out` (the follow-up reads the kept
captions from the kiosk's snapshot) and open **`http://<host>:8777/followup`**
on the participant's phone; pass `--host 0.0.0.0` to reach it from another
device. The page asks for the participant identifier; a participant with no
kiosk session gets a 409 and the page says so. An unfinished kiosk session
(a check clip with a shot no caption was kept for) is the same 409 to the
participant, and the response body names the first missing `clip_id` and
`shot_id` for you.

Progress is kept in the phone's browser and the page resumes at the first
task without an answer. The responses are written once: a second submission
for the same identifier (another device, a cleared browser) is refused with a
409, the first stays on disk, and the page treats it as saved.

Recognition shows a still of each clip with the document's sound list; sound
only plays excerpts cut by the server (the browser never learns which clip an
excerpt came from); the check loops each shaped clip with the kept caption
against the automatic one, A/B assigned per the document.

The check frame is the kiosk's stage (640 wide) with the placed caption at
the identity's 22 on 30, shown unscaled on any screen at least that wide (the
follow-up column is capped at 672 for this). On a narrower phone the whole
frame is scaled down as one picture, so the caption reads at about 12px on a
390-wide screen. That is a deliberate choice between two imperfect options:
a scaled picture keeps the caption exactly as it looked in the kiosk (spec 8)
and never cuts it short; reflowing identity's 22px box at phone width would
clip most captions behind an ellipsis. If a phone-sized true 22px is wanted
instead, drop the scale (`followup.css .frame`, `followup.js fitStage`) and
lower `CAPTION_MAX_CHARS` to what two lines hold at 358 wide.

## 5. What lands under `--out`

| File | What it is |
|---|---|
| `events-<participant>.jsonl` | append-only, one event per line with `received_at`; the record of the session. Never rewritten. |
| `snapshot-<participant>.json` | the browser's resumable state, replaced atomically on every save; the follow-up reads `kept` from it. |
| `followup-<participant>.json` | the follow-up responses (`dpo.caption-session-followup/v1`) with `received_at`, written once at the end; a second submission is refused. |
| `captions.json` | the caption cache, `clip/shot/settings-key` → caption, stamped with the writer that filled it (`template`, or `gemma:<model>:<checkpoint>:<instruction>:<budget>`). Safe to keep across restarts of the same writer; a file another writer wrote — a template rehearsal before a Gemma pilot, a different checkpoint — is refused at start. Delete it to force rewriting. |
| `media-cache/` | server-side cuts: `shots/` (16 kHz wav per shot, Gemma only), `stills/` (one jpeg per clip), `excerpts/` (sound-only wavs). Regenerated on demand. |

The download on the kiosk's final screen (`/api/log?participant=P01`) is the
events and the snapshot in one JSON document (`dpo.caption-session-log/v1`);
the files above are the authoritative copy.

## 6. The other instrument

`dpo console` builds the later specification in `docs/v2-console/`, which contradicts
this one on the control surface, the identity, the screens, and the copy. See
[`docs/v2-console/runbook.md`](../v2-console/runbook.md) for the comparison and its
operator steps. The two share `dpo.caption` and nothing else.
