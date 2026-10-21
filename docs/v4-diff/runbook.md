# v4-diff runbook — Stage 1, G0, phase 1 and live phase 2

How an operator runs what exists today:
- the Trida-7B and Qwen generator (Trida replaced Dream on 2026-10-12);
- G0, Korean feasibility (its human rating was waived on 2026-10-11);
- the live phase-2 app (§6).

The model change and the live redesign are at the top of [`plan.md`](plan.md).

Run every command from the repository root. Inside `src/dpo/regen`, regen's
`copy.py` shadows the standard library. There are two environments:

- `uv run python -m diff …` — the repository's environment. diff ships in its
  editable wheel. Use it for sources, plans, checks and the G0 report.
- `PYTHONPATH=src uv run --project envs/diff-generator --frozen python -m diff …` —
  the generator environment (transformers 4.57.3, einops, torch 2.5.1+cu124). Use it
  for model fetch, plan freeze, generation and the worker. Run `uv sync
  --project envs/diff-generator --frozen` once.

There is no `diff` console script. Use `python -m diff`, because a `diff`
executable in the venv would shadow the system `diff`.

## 1. Models

```sh
PYTHONPATH=src uv run --project envs/diff-generator --frozen python -m diff \
  model fetch --profile trida-7b@c37755e8 --dest artifacts/diff/models/trida-7b-c37755e8
PYTHONPATH=src uv run --project envs/diff-generator --frozen python -m diff \
  model fetch --profile qwen2.5-7b-instruct@a09a3545 --dest artifacts/diff/models/qwen2.5-7b-instruct-a09a3545
```

`fetch` downloads the pinned revision, hashes every file against
`src/diff/model_profile.py`, and writes `<dest>.verification.json`. Any
mismatch stops it. `model verify --no-weights` re-checks the small files
quickly. Each later load re-hashes the remote code before
`trust_remote_code`, and the code Python actually imported after.

## 2. Development sources (required before G0)

G0 reads **development-clip material only**: never a participant clip, and
never one of the three long videos. Write a `diff.caption-sources/v1` file;
`tests/diff/fixtures/diff-smoke-sources.json` shows the shape. Each cue needs:

- `"material": "development"`;
- its 원문 (`text`, at most 96 characters and 2 lines);
- structured `evidence.events`: source, category, count, action, qualities from
  the closed vocabulary, location or relation, certainty;
- `provenance`.

A cue is one timed caption line; regen's 10 s clips carry one each. At least 3
cues are needed, so the 12 corner-cell prompts exist.

The current set is `artifacts/diff/sources/c2-development-draft.json`. It holds
six of regen's twelve c2 clips, leaving out the live pair (`bangkok_034`,
`singapore_286`) and the clips plan Q15 excludes. Its 원문 comes from
`scripts/c2-korean-captions.json`. Its evidence is an unreviewed draft built
from each clip's AudioSet labels and descriptions. Check a sources file with:

```sh
uv run python -m diff sources validate --sources path/to/development-sources.json
```

`sources import-regen --material development` can start a draft from a regen
document, but the evidence must be authored. The live viewing manifest's
evidence is visual context only and is never development material.

## 3. G0

```sh
W=artifacts/diff/g0-$(date +%Y%m%d)
uv run python -m diff g0 plan --sources path/to/development-sources.json --out $W/plan.json
PYTHONPATH=src uv run --project envs/diff-generator --frozen python -m diff plan freeze --plan $W/plan.json \
  --trida-dir artifacts/diff/models/trida-7b-c37755e8 \
  --qwen-dir artifacts/diff/models/qwen2.5-7b-instruct-a09a3545 --out $W/frozen.json
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src uv run --project envs/diff-generator --frozen python -m diff generate \
  --plan $W/frozen.json --run-dir $W/run --device cuda:0 \
  --trida-dir artifacts/diff/models/trida-7b-c37755e8 \
  --qwen-dir artifacts/diff/models/qwen2.5-7b-instruct-a09a3545
uv run python -m diff g0 report --plan $W/frozen.json --run-dir $W/run \
  --sources path/to/development-sources.json --out-dir $W/g0
```

What each step does:

- **`g0 plan`** writes every development cue with the phase-1 prompt, plus 12
  corner-cell prompts, as `trida_tau_070`, `trida_tau_090`, `trida_tau_100` and `qwen`.
- **`plan freeze`** hashes each prompt's token IDs with the model's own chat
  template, and refuses prompts that break the 2,048-token context or contain a
  mask token or BOS.
- **`generate`** needs a GPU with at least 20 GB free, so check `nvidia-smi`
  first. On this machine, GPU 0 has been held since 2026-09-03 by the xart
  project's GPU API container (`xart-gpu-api-1`), not by regen. Each job:
  - runs in a supervised worker with a 60 s deadline;
  - records the resolved config, forward calls against `steps`, realized
    forward calls, and the RNG check (no call may consume randomness);
  - gets one identical retry only after a transient worker or device failure.
- **`g0 report`** writes the automatic report and stops there:
  - `rater/packet.csv` and `rater/ratings.csv` go to the rater;
  - `researcher/` holds the key, the report and the run manifest, and never
    goes to the rater.

**Rating.** One fluent Korean reader fills `naturalness` (1–5) for every row of
`rater/ratings.csv`, seeing only the packet.

**Decision.**

```sh
uv run python -m diff g0 decide --out-dir $W/g0 --ratings path/to/ratings.csv --decided-by "<name>"
```

This applies the provisional rule over the rated condition (`trida_tau_090`) and writes
`researcher/decision-<id>.json`, an `AnalysisDecision` naming the run
manifest's hash. Qwen's numbers are reported beside it. The rule:

- every job terminal, with no unresolved technical failure;
- at least 95 % of all rated-condition jobs passing the script check; an
  incomplete, empty or missing output counts as a failure (decided 2026-10-10),
  and so does an exact copy of the cue's 원문 (decided 2026-10-11);
- at least 3/4 rated 3 or higher.

When the automatic criteria already fail, no rating can change the outcome:
leave out `--ratings` and the fail is recorded with naturalness unrated
(`null`). Without ratings, `decide` refuses a run whose automatic criteria
pass, because a pass always needs the rating.

To record a waiver instead of a rating, as on 2026-10-11:

```sh
uv run python -m diff g0 waive --out-dir $W/g0 --decided-by "<name>" --notes "<reason>"
```

The record's outcome is `waived`, never `pass`. Only a step that names it in
`accept` takes it.

**After the decision.**
- **Fail:** stop. The researcher chooses between revised prompts on development
  material (a new protocol version) and another model.
- **Pass:** lattice generation (`plan build --phase 2 --g0-decision …`) and,
  later, the bundle builder accept the decision (AC-20).

## 4. Smoke runs

`g0 plan --smoke` builds the same plan over **synthetic** material, such as the
test fixture. Use it to exercise fetch → freeze → generate → report on real
hardware. A smoke run is never G0, and `g0 decide` refuses it.

## 5. Checks without a GPU

```sh
uv run pytest tests/diff
uv run ruff check src/diff tests/diff && uv run ruff format --check src/diff tests/diff
uv run mypy
```

`make check` runs these over the whole repository and also checks
`envs/diff-generator/uv.lock`.

## 6. Phase-1 setup (researcher review)

Phase 1 profiles the participant on regen's 12 c2 clips with regen's pages
(researcher's decisions of 2026-10-18). Its material is drafted from the
prepared metadata and then approved by the researcher:

```sh
uv run python -m diff phase1 draft-setup      # → artifacts/diff/phase1/setup-draft.json
uv run python -m diff phase1 show --setup artifacts/diff/phase1/setup-draft.json
uv run python -m diff phase1 approve --setup artifacts/diff/phase1/setup-draft.json \
  --by <name> --clip amsterdam_012 [--clip …] [--practice]
```

**What `draft-setup` fills in:**
- media, frames and masks, from `data/live/regen-media`;
- the prepared Korean captions, copied word for word from
  `scripts/c2-korean-captions.json` (`--captions`). They are shown raw, with
  their line breaks and no length or line limit;
- `orig_text` and every AudioSet label, from `data/corpus/tidy_data.csv`;
- each label's family, using the AudioSet ontology, as the reference for the
  options' families;
- nothing about wording: P0 and every page title are the questionnaire
  sheet's, word for word (`diff.workbook_instrument`, the sheet as of
  2026-10-18).

The options are written into the draft by hand and marked
`assistant-draft-unreviewed`. They keep `orig_text`, so R2's answer key is
relative to `orig_text`, not to the shown caption:
- **A1s:** one option per sound `orig_text` names. Each option has a Korean
  label, a family, and `orig_phrase`, the words of `orig_text` it renders.
- **R2:** the same sounds marked `mentioned` with their phrases, plus
  distractors that quote nothing.
- **Practice:** its options come from the practice clip's `orig_text`.

`show` and `approve` refuse a phrase that is not in `orig_text`.

The setup file holds clip keys, `orig_text`, families and the R2 answer key,
so it stays on this machine.

**Approval:**
- `show` prints each part, with every problem for participant sessions.
- `approve` binds an approval to a hash of the part's content, so any later
  edit voids it.
- Participant sessions are refused until every clip and the practice are
  approved.
- An optional `consent_text` in the file puts a consent checkbox on P0.

## 7. Participant sessions: phase 1, then live phase 2

The participant app runs in the repository environment. It starts one Trida
worker per GPU in the generator environment, so `uv sync --project
envs/diff-generator --frozen` must have run.

```sh
uv run python -m diff live serve \
  --phase1-setup artifacts/diff/phase1/setup-draft.json \
  --sources path/to/phase2-sources.json \
  --video <video_id>=path/to/video.mp4 \
  --items path/to/items.json \
  --trida-dir artifacts/diff/models/trida-7b-c37755e8 \
  --gpus 0 --log-dir artifacts/diff/live-$(date +%Y%m%d) --port 8782
```

**The inputs:**
- `--phase1-setup` is the approved setup (§6). It is required for
  participants. Smoke runs may leave it out, and then an entry goes straight to
  the watch with midpoint sliders and tuner slots reading "없음".
- `--sources` is a `diff.caption-sources/v1` file with phase-2 cues (unit kind
  `video`). Each cue holds its starting caption (`text`, at most 96 characters,
  or `serve` refuses it) and its `start_ms` and `end_ms`. No evidence is needed.
  Build it from a captions table, one row per video window:

  ```sh
  uv run python -m diff sources import-captions --input data/phase2/captions.csv \
    --text-column ko --out data/phase2/phase2-cues.json \
    --end viewing-1=299.679 --end viewing-2=300.259 --end viewing-3=300.142 \
    [--reviewed-by <researcher>]
  ```

  - The table needs `video`, `window` (0, 1, …), `start_s`, `end_s` and the
    caption column. Every other column is kept in each cue's note.
  - Each video's windows must run contiguously from 0 s, and each caption must
    fit in 96 characters. Otherwise nothing is written and the problems are
    listed.
  - `--end` gives a video's real end, so its last window reaches it.
  - Pass `--reviewed-by` only once every caption in the table has been read;
    without it the cues are drafts, which only smoke runs serve.

  The user test's long videos are `data/phase2/viewing-1..3.mp4` (not in git),
  served as `--video viewing-1=data/phase2/viewing-1.mp4` and so on, in watch
  order.
- `--video` gives one `ID=PATH` per video, in watch order. Settings carry over
  between videos.
- `--items` holds the post-watch items:
  `{"items": [{"id", "text", "min", "max", "anchors": [low, high]}]}`.
  The text is Korean.
- `--store` is the session store. Participants default to
  `artifacts/diff/participants.sqlite`, one store for the whole study, because
  IDs are unique across it. Smoke runs get `<log-dir>/smoke-sessions.sqlite`,
  so a test `P1` never blocks a real one.
- `--gpus` lists the cards to use, one live session per card. `serve` refuses
  to start unless every listed card has at least 17 GB free
  (`LIVE_MIN_FREE_MIB`). Other jobs on this machine also use GPU 0, so check
  `nvidia-smi` and list only the free cards.
  - A session holds its card from entry until its last video ends. With one
    card, participants therefore come strictly one after another: a second ID
    entered meanwhile is told to wait ("잠시 후", 503).
- `--purpose participant` is refused in any of these cases:
  - the tuner role text is provisional (`rewrite_prompts.TUNER_SYSTEM`);
  - the nine live prompts are provisional (`rewrite_prompts.LIVE_CELL_PROMPTS`);
  - the second-caption prompt is provisional (`rewrite_prompts.TWEAK_P1`);
  - the setup has unapproved parts;
  - a phase-2 starting caption is not reviewed: every cue needs status
    `reviewed` and text provenance that is human or reviewed.

  Use `--purpose smoke` until then.

**Before a study day,** run the same arguments through `live preflight`
(without `--purpose`, `--log-dir` or `--port`). It serves nothing and lists, all
at once, everything that would stop a participant session:
- the gates: the tuner text, the second-caption prompt, the setup's approvals
  and unreviewed starting captions;
- each video's cues against the video's real length (with `ffprobe`);
- the post-watch items, the Trida snapshot's small files, and each listed GPU's
  free memory.

It exits 1 while anything is listed and prints "ready for participant
sessions" otherwise. `serve` itself now reports every unmet gate at once too.

**Running a session:**
- Open `http://<host>:8782/` and type the participant ID (`P1`, `P2`, …).
  Typing the same ID again resumes an unfinished session, and a finished ID is
  refused. To switch participants on a browser that still holds a session,
  open `/#enter`.
- The flow is:
  - P0 (the introduction, plus consent if there is consent text), then the
    practice;
  - stage A: for each of the 12 clips, in a random order recorded per
    participant, the first viewing (the raw prepared caption), then P2–P5;
  - the stage-B interstitial ("첫 번째 시청을 마쳤습니다"), phase 1's one natural
    break;
  - stage B: the same clips in the same order, the second viewing (Trida's
    caption), then P7, P8, P9, P10 (S1–S3) and P11 (R1/R2);
  - the phase-2 briefing ("이제 긴 영상 N편을 봅니다"), whose "2스테이지 시작"
    builds the tuner's calibration from the phase-1 answers and opens the
    watch at `/watch/`;
  - the videos, then the post-watch items;
  - the end screen, which asks the participant to stay and call you for the
    interview (P12) and debrief (P13), both outside the app. The session is
    already finished and saved. Hold "연구자 확인 · 다음 참가자" for one
    second to return to the entry.
- **Second captions:** when a clip's P5 is submitted, Trida writes its second
  caption on the session's GPU. It works from the raw caption and that
  participant's step-1 answers (`diff.phase1_tweak`), while the participant
  goes on with stage A.
  - A participant who reaches P6 before it is ready sees "다음 영상을 준비하고
    있습니다…" until it is.
  - A caption that fails its check, or whose call fails, skips that clip's
    second step automatically, with the reason recorded. There are no
    quality retries; one identical retry follows a lost worker or device
    error.
- Phase 1 copies regen's look, and the watch shares it (UI review of
  2026-10-19), including the caption band. Every page but the watch shows the
  whole session's progress in six segments.
- **The watch:** each slider has three stops, the steps its axis really has.
  A move shows a chip saying the caption is being rewritten. The rewrite
  replaces the caption on screen about a second later, mid-window, and the
  chip says the change is applied. Late in a window, the change lands in the
  next window, and until then the chip says that window shows a caption from
  the earlier settings. Hiding the tab pauses the video behind phase 1's
  overlay.
- **After a server restart:** phase 1 continues exactly where it was, and any
  second caption still pending is queued again when the participant's page
  next talks to the server. A watch
  in progress restarts its current video from the start, with its settings
  (logged as `session_resumed`, `video_restarted: true`).

**GPUs:**
- Each GPU in `--gpus` must have 17 GB free at start, or `serve` refuses.
  Trida takes about 15 GB.
- A session claims its GPU when the ID is typed, and holds it through phase 1
  (which needs it for the second captions) and the videos. It releases the GPU after the last video, before the items.
  When no GPU is free, the entry says so before anything starts.
- The researcher's midsentence renders share these cards, so check
  `nvidia-smi` first.

**Logs, under `--log-dir/sessions/`:**
- `session-<id>.jsonl`: settings, playback, exposures (each with the job it
  showed), video ends and ratings;
- `live-<id>.jsonl`: every Trida call, with its job, system message, token
  hash, latency, output and checks. Live captions also carry `L` and the
  target. Second captions are marked `kind: "tweak"`, with the clip, the
  attempt, the slots and degrees, and the outcome.

`verification-trida.json` and `worker-gpu<N>.log` sit beside them. Phase-1
answers, viewings, second captions, skipped steps and the calibration are in
the store.

**Behaviour to expect:**
- Trida at τ 0.9 on the 96-token live canvas (protocol v6) takes about 0.4 s a
  caption on an RTX 3090 for short starting captions, and about 0.8 s (p95
  1.4 s, at most 2.7 s) for the M1 list captions of the long videos. A second
  caption, on its 128-token canvas, took 0.53 s (median; 2.2 s at most) in the
  smoke walk of 2026-10-19.
- The texture slider has three steps (0–2), and the context slider has three.
- A change shows as soon as it is written: about 1–1.5 s after the move (the
  300 ms debounce, Trida's ~0.6 s, the page's 0.5 s poll), in the window that
  is playing. Every caption stays on screen at least 3 s (`MIN_READ_MS`):
  a rewrite that arrives sooner waits, and one that would have less than 3 s
  of its window left is not swapped in. When less than 3.5 s of the playing
  window would be left at `p + L` (`MIN_SHOWN_MS`), the change lands in the
  next window instead. If the worker is busy, it appears one call later; a
  model call cannot be cancelled.
- Every window shows a caption about its own audio: Trida's under the current
  settings; else its caption for that window under earlier settings (after a
  late move, or while the worker is busy or restarting); else the window's
  starting caption. A caption whose check failed or flagged a repetition
  (about 9% on the M1 captions) is never shown.
- A worker that overruns 30 s is restarted, which reloads the model; windows
  meanwhile show their captions under earlier settings or their starting
  captions.

**Walking the interface without a GPU:** add `--fake-worker --purpose smoke`
and leave out `--trida-dir`. The captions are placeholders marked "시험용",
never research data. Playwright's Chromium has no H.264, so a headless walk
needs WebM copies of the clips in a walk-only copy of the setup.

## 8. Export

```sh
uv run python -m diff export --log-dir artifacts/diff/live-20261020 [--log-dir …] \
  --phase1-setup artifacts/diff/phase1/setup-draft.json --out artifacts/diff/export-$(date +%Y%m%d)
```

It reads the store (`--store`, default the participant store) and every
`--log-dir`, and writes into a new, empty directory:
- `participants.jsonl`: each session's full record, including every
  second-caption attempt;
- `phase1.csv`: one row per participant × clip × item, practice excluded.
  `viewing` (`P1` or `P6`) says which viewing an answer belongs to. It
  carries:
  - the clip key;
  - for A1s and A2s, the chosen sound's label, family and `orig_phrase`;
  - for V1 and V2, the matched objects;
  - for R2, the intended key, the sounds the second-caption prompt named;
- `second_captions.csv`: one row per participant × clip:
  - the status, and the reason for a skipped step;
  - the first and second captions;
  - the attended sound and object, and the four degrees;
  - the intended R2 key, the template hash and whether it was provisional;
  - the job and the check flags;
- `post_watch.csv`;
- `manifest.json`, with file hashes.

This is the only way data leave the app.

## 9. Audit before analysis

```sh
uv run python scripts/diff_session_audit.py check --store artifacts/diff/participants.sqlite \
  --log-dir artifacts/diff/live-20261020 [--log-dir …] \
  --phase1-setup artifacts/diff/phase1/setup-draft.json --sources path/to/phase2-sources.json \
  --export artifacts/diff/export-20261020 --out artifacts/diff/audit-20261020.json
```

Run it on the pilot and on every participant session before analysis. It
opens the store read-only and exits 1 on any mismatch. From each
participant's answers it recomputes:
- the object under each P2 point, by re-reading the masks;
- each direction's degree and the other tweak slots;
- the tuner's heard families and rating means.

diff's own prompt functions render what those values should produce, and the
audit compares that with every logged call. It also checks that:
- each second caption is what Trida wrote and what P6 showed, or that a
  failed one was skipped;
- every slider change got a live call;
- every caption shown maps to a call;
- the export matches the store.

Audit with the commit the sessions ran: a session run on another template is
reported once, as a version mismatch.

When the model, its tokenizer or the generator environment changes, also
check that the logged prompts are what the worker tokenized:

```sh
envs/diff-generator/.venv/bin/python scripts/diff_session_audit.py tokens \
  --trida-dir artifacts/diff/models/trida-7b-c37755e8 --log-dir artifacts/diff/live-20261020
```

## 10. Comparing prompt wordings

`scripts/diff_prompt_bench.py` runs candidate wordings of live-p2, tweak-p1
and the tuner on fixed participant answers, at the study's settings. Its
docstring gives the candidate file's format. A live candidate replaces any of
the nine cell prompts (`"cells": {"0-broad": "…{cue_text}…", …}`).
- `inputs` keeps a run's answers and the slider steps of its live calls.
- `run`, in the generator env on one GPU, rebuilds every prompt from those
  answers with the checkout's code (or a candidate's wording) and writes it
  with Trida.

It reports:
- how often an output copies its starting caption;
- whether texture words appear as each texture step asks;
- check failures and flags, on the shown text;
- how often Trida's own text has 「」 or a leading label, which the display rule removes;
- a rough count of visibly broken text ("소" cut from "소리", ",,", "..");
- how many second captions change between 0% and 100% degrees.

The broken-text count misses typos, so a fluent reader still rates the chosen
wording before it is approved.

The baseline inputs are `artifacts/diff/bench/inputs-smoke-20261019.json`,
from the 2026-10-19 smoke run with three agent pseudo participants. The file
is not tracked, because it holds the long videos' draft cues.
