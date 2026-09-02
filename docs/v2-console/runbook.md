# Caption console — operator runbook

The console is the participant instrument specified in
[`docs/v2-console/spec-system.md`](spec-system.md)
(the measured half) and
[`docs/v2-console/spec-uiux.md`](spec-uiux.md)
(the surfaces): a participant watches a street clip with the model's own
captions, shapes each shot's caption with three controls, watches it again with
their captions on it, then chooses between their caption and the default
policy's. It lives in `dpo.console` and is driven by `dpo console`.

**It is the second of two instruments built to two specifications that
disagree.** The first, `dpo session` (see
[`docs/v1-session/runbook.md`](../v1-session/runbook.md)), builds `docs/v1-session/`, where the
ordered list of what a caption will mention is the *only* control surface, the
identity is caption yellow on near-black, and the trial runs six clips through
a listing step, a control task, and a follow-up on a later day. This one builds
`docs/v2-console/`, where a console drives a read-only skeleton, the identity is
sodium on concrete, and one clip runs four phases ending in an in-session
check. They share `dpo.caption` — the writers, the cache, the caption budget —
and nothing else, so whichever is not adopted can be deleted whole.

| | `dpo session` | `dpo console` |
|---|---|---|
| specification | `docs/v1-session/` | `docs/v2-console/` |
| control surface | the skeleton itself | a console; the skeleton is feedback |
| balance | ordering columns with trade-place lines | segmented crossfader, widths from the regimes |
| detail | a fold handle, four levels | four detents, atmospheric grays the rest |
| weights | authored per source | computed: `r_g`, `p_g`, `c_g`, `e_g` |
| check | follow-up, on a later day, against the raw caption | in session, against the default policy |
| port | 8777 | 8778 |

## 1. Preprocess the masks

`console preprocess` runs §3 and §4 over the Sa2VA mask tree: it cuts shots on
visual composition, groups audio labels whose masks agree, and measures `r_g`
and `p_g` per shot per group.

```bash
uv run dpo console preprocess \
  --mask-root /mnt/hdd/research/2026/Sa2VA/avmask/runs/60fps-windowed/masks \
  --tidy-data /mnt/hdd/research/2026/Sa2VA/avmask/data/tidy_data.csv \
  --ontology /mnt/hdd/research/2026/Sa2VA/avmask/data/ontology.json \
  --cache-dir data/console/mask-cache \
  --out data/console/masks.json --fps 60 \
  --clips amsterdam_006 singapore_303
```

`--tidy-data` with `--ontology` resolves every audio label to its AudioSet
family, and labels from different families are never merged however their
masks overlap; without them the grouping is by mask agreement alone, and the
manifest's `grouping_constraint` says which you got. `--cache-dir` keeps the
decoded masks between runs, one compressed file per clip, so re-running the
calibration sweep reads no PNG twice and produces an identical manifest.

Both release layouts are read (`{branch}/{clip}/{label}/{frame}.png` and
`{branch}/{clip}/{label}_{second}.png`). Masks are read at 1/4 resolution by
default; the factor is in the manifest's provenance because `r_g` depends on
it.

### What the calibration does, measured on this corpus

Every number below was measured on 2026-09-01 over `amsterdam_006` and
`singapore_303`, and every one of them is a per-corpus calibration you should
re-measure on yours.

**`θ`, the composition shift that cuts a shot.** The maximum shift observed
over a 250 ms stride was **0.112** on `amsterdam_006` and **0.068** on
`singapore_303`; the medians were 0.014 and 0.018, and at the default
`θ = 0.25` neither clip is cut. Over the whole corpus (48 clips, 2026-09-02)
the default cuts **14 clips** — 8 into two shots and 6 into three, the
shortest 2.0 s — so a ten-second clip is one shot for 34 of them, not all.
Look at `frames` per shot before recruiting on a cut clip: a 2.5 s shot is a
2.5 s audio window. Lower `θ` before you have measured audio reliability and
you will shorten every analysis window for no visual reason (§5.2). The floor
on shot length binds on the last shot as well as the first, so a clip that
cannot be split into shots that all clear the floor stays whole.

**The IoU threshold.** On `amsterdam_006` the pairwise agreements were:

| pair | IoU |
|---|---|
| Traffic noise ↔ Vehicle horn | 0.725 |
| Bird ↔ Traffic noise | 0.673 |
| Bird ↔ Vehicle horn | 0.496 |
| Bird ↔ Idling | 0.128 |
| Idling ↔ Traffic noise | 0.062 |
| Idling ↔ Vehicle horn | 0.003 |

At the default 0.5 the first two pairs clear the threshold, and by mask
agreement alone transitivity would carry the third in with them: *Bird*,
*Traffic noise* and *Vehicle horn* as one source. That is evidence about the
grounding rather than about the street — Sa2VA returned nearly the same region
for three different audio prompts — and it had a cost past the grouping: the
fused source's prose ran to seventy-one characters, no sentence naming it fit
the caption budget, and every caption that mentioned it fell to the template
fallback.

Two things now stop that. Labels merge only within one AudioSet family, which
§4's own justification — labels that share a *physical source* — requires and
which an animal and a vehicle do not satisfy; on this clip that yields *Bird*
alone, *Traffic noise* with *Vehicle horn*, and *Idling* alone. And a source is
named by each label's canonical clause rather than its whole display name, so
the merged pair reads *traffic noise and vehicle horn*, thirty characters. On
`singapore_303` *Siren* ↔ *Vehicle* scored 0.001 and nothing merged. Look at
the `grouping` field of every clip before you accept a manifest all the same.

**The corpus, preprocessed whole.** With `--tidy-data`, `--ontology` and
`--provisional-salience`, all 48 clips (16 per city) scaffold into one document
that validates: 68 shots, 120 sources (two to four per shot, one shot with one),
109 labels standing alone and 11 merges of two to four labels, the longest
source name 42 characters. Served on the base E4B every audition and one
caption per shot came from the model — 390 captions, 13 of them tightened,
none from the template — so the budget holds across the corpus, not only on
the two clips above.

**`r0`, the half-saturation constant.** The default 0.02 says a source
occupying two per cent of the frame is half visible. Grounded audio masks on
this corpus are small — `r_g` between 0.0003 and 0.005 — so at `r0 = 0.02`
every source sits low on the saturating curve and the visibility axis has
little spread. Set it from the corpus with the admin calibration protocol
before interpreting any setting.

## 2. Scaffold, author, validate

```bash
uv run dpo console scaffold --manifest data/console/masks.json \
  --out data/console/session.json --study-id street2026 --corpus-id amsterdam
# {"status": "scaffolded", "config_hash": "7612638de7ea", ...}
```

The scaffold does not validate, and the reason is not an oversight:

```bash
uv run dpo console validate --session data/console/session.json
# {"status": "invalid", "error": "clips[0].shots[0].raw_caption: must not be empty", ...}
```

Three things are yours to author.

`raw_caption` is the audio-language model's own prose, played during the first
viewing. It is deliberately **not** a point in the console's reachable space
(§11): a participant who prefers it cannot steer back to it, and that
frustration is what the probe is for.

`default_caption` is what the fitted policy would produce — the thing the study
is trying to beat. It is one side of the check.

`confidence` and `energy` are `c_g` and `e_g`, and **nothing in this repository
can supply them**. `w_g = c_g · e_g` separates detection from acoustic
prominence (§5.1); a mask grounds a *possible visible source* of a tagged
sound, and measures neither its loudness nor its onset. `tidy_data.csv` carries
no per-label confidence, and no band-limited energy pass exists. The schema
refuses a document that carries only one of them, because defaulting the other
to one would silently make the audio axis detectional — the failure §5.1 names.

For a dry run before that measurement exists:

```bash
uv run dpo console preprocess ... --provisional-salience \
  --tidy-data /mnt/hdd/research/2026/Sa2VA/avmask/data/tidy_data.csv
```

which fills both from tag multiplicity and stamps `provisional_salience: true`
into the manifest and its limitations. `scaffold` carries that flag into the
document's `config`, where it is part of the stamp (§5): a dry run's
`config_hash` can never equal a measured study's, `validate` and `serve` both
report `provisional_salience` on their status line, and a document whose
config does not say either way is refused rather than assumed measured. Never
recruit on one.

## 3. Serve the kiosk

```bash
uv run dpo console serve --session data/console/session.json \
  --media-dir data/live/media --out data/console/responses
```

The command prints one status line before it starts serving — the stamp,
whether the document is a dry run (`provisional_salience`), the writer and
the URL — and nothing after it.

Open **`http://127.0.0.1:8778/?participant=P01`** in the kiosk browser. The
participant identifier comes from the URL; opened without one, the page is a
door — the title, a field for the identifier, and Continue, which reopens the
page with it in the URL. Every interaction is autosaved and a reload
resumes at the interrupted step. That identifier is the only thing the server
checks, which is right for a supervised kiosk on its own machine; if the
server is ever bound to a network address (`--host 0.0.0.0`), use identifiers
nobody could guess and a network of the study's own devices, since anyone who
knows one can read or replace that participant's record.

### The layout at kiosk width

`spec-uiux.md` §3.1 stacks the rail, the stage, the band, the console, the
skeleton and the actions top to bottom, and §1 sizes the kiosk at about 1180
by 820. Those two do not fit together: at the 760 content column the stage
alone is 428 tall and the author screen runs to about 1100, with the detail
detents and Show/Keep below the fold. As built, at 1100 and wider the stage
and the band take the left column and the console, the skeleton and the
actions stack beside them, so a shot with eight sources still fits in 820
without scrolling; narrower than that the screen lays out in §3.1's order.
The DOM order is §3.1's either way. The stage is 720 wide, which is what the
side column leaves once the four detents sit on one line; Replay shot and
Pause loop sit under the footage they act on, and the primary actions hold
one place, bottom right on the band's baseline, on every stage screen.

On a screen taller than 820 the page zooms the composition to the screen's
height, so the console keeps its hand size, and the stage takes every pixel of
width that leaves beyond the 388 side column until its 16:9 height meets the
band; the shell stops widening past that, so the console stays beside the
footage. On a 27-inch 2560×1440 that is a 972-wide stage in the page's units
against the kiosk's 720. The kiosk itself is untouched. If the researcher would rather keep the single
stack at kiosk width, the stage has to shrink to about 220 tall to fit, or
the page has to scroll.

`--writer gemma` conditions the study's Gemma 4 E4B on the shot's audio with
this instrument's own instruction — grain, the ordered sources with their band
and register, and the rule that a source out of frame is heard but not seen. It
needs a CUDA device and exits 3 without one, like the other instrument. Either
writer sits behind the same cache, keyed on shot, regime, admitted set and
grain (§8), so identical settings return the identical prose and a revisit
costs nothing. There is no reroll.

Both writers are warmed at start-up: every single-source audition is written
before the first request, so holding a token is instant. A writer that cannot
write makes the server refuse to start rather than serve errors to a
participant.

## 4. What lands under `--out`

| File | What it is |
|---|---|
| `events-<participant>.jsonl` | append-only, one event per line, each stamped with `received_at` and the configuration hash |
| `snapshot-<participant>.json` | the browser's resumable state, replaced atomically; the check reads `committed` from it |
| `captions.json` | the caption cache, `clip/shot/settings-key` → caption, stamped with the writer that filled it; a file another writer wrote (a template rehearsal, another checkpoint) is refused at start — delete it or serve from another `--out` |
| `media-cache/` | server-side cuts: shot audio for the Gemma writer, clip stills |

Two things in the event stream are written by the server and were never in a
browser. `shot.measured` carries §9 — `C_anch`, `D` and `ρ_g` — written once
per shot per configuration; those quantities are the predictor side of the §12
model and are hidden from participants entirely. And every line's
`config_hash` is applied on receipt from the configuration the server runs
under, never from anything the page sends, so a stale tab cannot stamp a result
with an old calibration.

## 5. The stamp

Method constants and calibrations freeze into a versioned artifact carried in
the document, and its hash goes on every caption, every endpoint response, and
every log line (§10). Change `r0`, a band phrase, the IoU threshold, the
corpus id, or where `c_g` and `e_g` came from (`provisional_salience`) and the
hash changes; two studies cannot then share a stamp.

The method constants in the artifact are *declarations* of what
`dpo.console.quantities` computes. Nothing dispatches on them, but they are
hashed, and a document whose declarations differ from the running build's is
refused rather than served — a result stamped with a hash that no longer
describes how it was computed is worse than no result. If you change a formula,
change the declaration beside it.

## 6. What is deferred

Table 5 of the system document lists what remains unsettled. As built:

- `r0` is a flag on `scaffold`; θ, the shot floor and the IoU threshold are
  flags on `preprocess`, and `scaffold` stamps the document with the values the
  manifest was computed under — all defaulted and hashed, awaiting the admin
  calibration protocol;
- the Korean wording of the criterion sentence is a construct decision, and
  every string sits in `dpo/console/copy.py` so it changes in one place;
- `w_g = c_g · e_g` is blocked on an acoustic measurement, as above;
- cross-shot persistence is deferred to what the probes fill with;
- saturation direction is recorded in the artifact as
  `ordering_saturates: true` — a revisable theory choice, not a fact, and the
  document keeps `r_g` beside `v_g` so the linear alternative can be measured
  against it later.
