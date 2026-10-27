# ConCap: Comparative Preference Alignment for Audiovisual Congruence Captioning

The repository holds two packages:

- **`diff`** (`src/diff`, `python -m diff`): **the study instrument in use.** It
  is one participant app in two phases. Phase 1 shows 12 street clips twice
  each, the second time with an audio caption rewritten from that
  participant's own responses. Phase 2 plays long videos whose captions are
  rewritten live as the participant moves two sliders. See
  [the `diff` study](#the-diff-study-phase-1-and-phase-2) below, and
  [`docs/v4-diff/`](docs/v4-diff/) for the operator runbook and the plan.
- **`dpo`** (`src/dpo`): the preference-alignment pipeline ConCap began as,
  with its earlier participant instruments. It is described from
  [the `dpo` pipeline](#the-preference-alignment-pipeline-dpo) on.

## 0. Install and verify

Requirements: Linux, Python 3.11, `uv >= 0.11`, Git. The `diff` app needs no
GPU, only an Anthropic API key for its caption writer
([below](#the-diff-study-phase-1-and-phase-2)). `dpo`'s real training
additionally requires the locked CUDA/PyTorch stack. One RTX 3090-class card
(24 GB) is the reference setup and fits a contract declaring one caption track;
a contract declaring two loads a base per track in one process, so it needs a
card per track until those bases are shared.

```bash
uv sync --dev
make check      # ruff + mypy --strict + pytest + lockfile check
make smoke      # offline end-to-end canary
```

`make check` covers both packages, the `diff` suite (`tests/diff`) included.
`make smoke` is `dpo`'s complete health check: a cold canary executes every
pipeline stage on synthetic fixtures with the tiny CPU backend (all matrix
cells train with real optimizer steps, the flipped-label retrainings
included), and a warm rerun must reuse the same report artifact with zero
recomputation. The live commands gate themselves: without a CUDA device or a
wired backend, `train`, `select`, `candidates generate`, and `study export`
refuse with exit 3 and no side effects. For a persistent, inspectable
workspace run the underlying commands yourself:

```bash
uv run dpo canary run --workspace artifacts/canary --contract configs/study/canary.toml
uv run dpo artifact verify --workspace artifacts/canary --all
```

## The `diff` study: phase 1 and phase 2

### Purpose

The study asks how an audio caption that answers to a listener's own
experience of a street changes that experience, toward attention restoration.

- **Phase 1** shows 12 short street clips, two viewings each. The first
  viewing shows the clip's prepared caption as it was written. The second shows
  a caption rewritten for this participant from their answers to the first:
  - it centres the sound they said they attended to most;
  - it carries the restorative quality they perceived most in that clip:
    fascination, being away, or extent and coherence (Payne, 2013), each
    centred on their own mean over the clips;
  - it is written in one of two modes. *Scene-aware* ties the sounds to what
    is on screen; *acousmatic* describes them as heard, without the picture.
    A Bayesian binding model recommends a mode for each clip from three
    inputs: how visible the noticed sound's source is (measured from Sa2VA
    masks), whether they pointed at it on a still, and their ratings of how
    the sounds fit the scene. A balanced plan then follows or counters that
    recommendation. That contrast is what phase 1 tests.
- **Phase 2** plays three five-minute videos. Their captions are rewritten
  live while the participant moves two sliders:
  - texture: how the sounds are described, from none to a little to a lot;
  - context: broad kinds, specific sources, or places and relations.

  The rewriter's role text (the tuner) is built from that participant's
  phase-1 answers. Items after the videos close the session. Testing how the
  restorative emphasis modulates the experience is planned for phase 2, under
  the same rule, once phase 1 is confirmed; it is not built yet.

Every caption is written by Claude Sonnet 5.5 through the Claude API
(`diff.claude_writer`), at low effort. Trida-7B wrote them until 2026-10-27, and
Qwen3.5-9B phase 1's for a day; why they were replaced is in the plan. Nothing
in the app runs on a GPU.

### A session

The experimenter types the participant ID on the start screen. Typing the
same ID resumes an unfinished session, and a finished ID is refused. The
pages follow the study's questionnaire sheet, in Korean:

1. **Introduction, consent and a practice clip.**
2. **Stage A, the 12 clips in a random order recorded per participant.** For
   each clip:
   - the first viewing, with the prepared caption;
   - where on a still the sound seemed to come from (V1);
   - which sound they attended to most (A1s);
   - the restorative appraisal (P4);
   - the remaining first-step items (P5).
3. **Second captions.** When stage A closes, each clip's mode is decided, and
   its second caption is written and checked. A gate (`diff.phase1_usable`)
   refuses:
   - broken or cut-off text;
   - copies of the first caption, and lists of names;
   - a missing attended sound, or a sound that is not in the audio;
   - the questionnaire's own wording;
   - a place in the wrong mode.

   A refused caption is written again at the next rung of a retry ladder:
   the first wording twice, then two others. A clip whose last rung is
   refused skips its second step.
4. **Stage B, the same clips in the same order.** For each clip:
   - the second viewing, with the participant's caption;
   - the point on a still again (V2);
   - the sound attended to again (A2s);
   - the appraisal again (P9);
   - ratings of the caption (S1–S3);
   - recall of the caption (R1, and R2: which sounds it named).
5. **Phase 2.** A short briefing opens the live watch. On each slider move the
   caption on screen is rewritten within about 2–3 s. After the videos come
   the researcher's post-watch items.
6. **The end screen.** The interview and debrief happen outside the app.

### Running it

The app runs in the repository environment. The writer reads the API key
from the shell. Put the key in `~/.bashrc`, above its "If not running
interactively" guard, so scripts and non-interactive shells see it too, and
keep the file private (`chmod 600 ~/.bashrc`):

```bash
export ANTHROPIC_API_KEY=…   # never in the repository; the key's owner pays for the calls
```

Build and approve the phase-1 setup and the phase-2 cues first (runbook §6
and §7), then list everything that would stop a participant session, and
serve:

```bash
uv run python -m diff live preflight \
  --phase1-setup artifacts/diff/phase1/setup-draft.json \
  --sources data/phase2/phase2-cues.json \
  --video viewing-1=data/phase2/viewing-1.mp4 --video viewing-2=data/phase2/viewing-2.mp4 \
  --video viewing-3=data/phase2/viewing-3.mp4 --items path/to/items.json

uv run python -m diff live serve \
  --phase1-setup artifacts/diff/phase1/setup-draft.json \
  --sources data/phase2/phase2-cues.json \
  --video viewing-1=data/phase2/viewing-1.mp4 --video viewing-2=data/phase2/viewing-2.mp4 \
  --video viewing-3=data/phase2/viewing-3.mp4 --items path/to/items.json \
  --sessions 1 --log-dir artifacts/diff/live-$(date +%Y%m%d) --port 8782
```

Open `http://<host>:8782/`. Notes:
- `--sessions` sets how many participants can run at once.
- `--purpose smoke` runs before the gates are met, on its own store.
- Add `--fake-worker` to walk the interface with placeholder captions, without
  the API.
- `serve` makes one short API call at start, and stops if the key or the
  network fails.

**What leaves the machine:** every caption prompt goes to Anthropic. A
prompt holds the clip's caption and sound words, the sound the participant
attended to, the emphasis their ratings chose, or the phase-2 tuner text. It
holds no participant ID and no raw answer.

**After sessions:**

```bash
uv run python -m diff export --log-dir artifacts/diff/live-20261028 \
  --phase1-setup artifacts/diff/phase1/setup-draft.json --out artifacts/diff/export-20261028
uv run python scripts/diff_session_audit.py check --store artifacts/diff/participants.sqlite \
  --log-dir artifacts/diff/live-20261028 --phase1-setup artifacts/diff/phase1/setup-draft.json \
  --sources data/phase2/phase2-cues.json --export artifacts/diff/export-20261028
```

- **The export** is the only way data leave the app. It writes CSVs for the
  phase-1 answers, the second captions, blind R2 coding and the post-watch
  items, plus every session's full record.
  - The second-caption file names the mode, the follow or counter plan, the
    emphasis and the model that wrote each caption.
- **The audit** recomputes every session from its answers. It checks:
  - the modes, the emphases and the prompts;
  - every logged request against the job it should have sent;
  - the retry ladder and what each screen showed;
  - the export.

**Benches** check caption delivery before a study day. Each API call is
billed, and Claude's output varies between runs.
- `scripts/diff_second_captions.py` writes every second caption a participant
  can reach down the retry ladder.
- `scripts/diff_prompt_bench.py` scores candidate wordings of the live, second-
  caption and tuner prompts on fixed answers.

### Status

Participant sessions are refused until the researcher approves:
- the nine live prompts and the tuner's role text;
- the second-caption wordings and the gate;
- the binding model's parameters and plan;
- the phase-1 setup;
- every phase-2 starting caption.

`live preflight` lists whatever is still open. The post-watch items are the
researcher's to write. What has been measured, decided and left open is in
[`docs/v4-diff/plan.md`](docs/v4-diff/plan.md). The operator steps are in
[`docs/v4-diff/runbook.md`](docs/v4-diff/runbook.md).

## Earlier and retired instruments

- **`dpo regen` is retired** (2026-10-19). `diff` replaced it as the study
  instrument. The code stays in `src/dpo/regen`, frozen, with
  [`docs/v3-regen/`](docs/v3-regen/), and `make open`/`deploy/edge` still
  describe its old public door.
- **The `dpo session` and `dpo console` instruments**, and **the congruency-
  slider study** (`dpo study serve`), belong to the `dpo` pipeline below.
  [`docs/README.md`](docs/README.md) maps every instrument's documents.

## The preference-alignment pipeline (`dpo`)

ConCap (package: `dpo`) is a pipeline with five jobs, in order:

1. **Collect** human caption preferences through a local web UI: prepared
   video clips are shown muted (visual track) or as audio — with a per-clip
   opt-in to unmuted video — under two selectable captions.
2. **Train** the comparison matrix on the collected preferences: SFT plus
   seven preference objectives (DPO, IPO, cDPO, rDPO, Dr.DPO, wDPO, and
   SFT-to-DPO warm start), with in-contract hyperparameter sweeps.
3. **Compare** the results: per-variant validation accuracy, winner selection,
   a locked configuration, and the inferential layer over them — all as
   content-addressed artifacts.
4. **Export** the held-out study split's captions from the locked winner, along
   a *measured* audiovisual congruency axis — one caption per slider stop.
5. **Run** the human study: a participant watches each clip and drags that
   slider until the sentence fits what they hear.

Every input is a typed, content-addressed artifact; one study contract owns
every result-affecting knob; and the same commands run the synthetic canary on
CPU and the real study on GPU — which backend executes is a contract field,
not a different code path.

## The `dpo` study, end to end

The `dpo` pipeline's study is `configs/study/street-audio.toml` — Gemma 4 E4B on the
**audio caption track only** (see the contract header for why), over 48 street
clips split train 27 / validation 7 / test 7 / study 7. Its working data:

```text
data/video/        the collected c1..c4 condition renders (read-only stimuli)
data/live/media/   staged corpus: {clip_id}.mp4 (muted) + {clip_id}.wav per clip
data/annotation/   exported sessions; answers-*.json holds the attention-check
                   keys and must never live inside the served --media-dir
artifacts/street/  the study's content-addressed store
runs/checkpoints/  per-cell adapters, resumable by content hash
data/userstudy/    human-study responses, one JSON per participant
```

| phase | needs | run |
| --- | --- | --- |
| **1. Annotate** | two authors, 212 tasks each (~1.5 h at 25 s/task) | `make annotate SPLIT=train`, then `SPLIT=validation`, then `dpo annotation ingest` per split |
| **2. Train and select** | one 3090 | `dpo views derive` → `dpo train run` → `dpo select run` |
| **3. Compare** | — | `make report` and `dpo report analyze` |
| **4. Export stimuli** | one 3090, ~17 min | `dpo study export` |
| **5. Run the study** | participants | `dpo study serve` |
| **6. Analyze the study** | — | `dpo study ingest` |

```bash
make annotate SPLIT=train        # serve tasks-train.json on data/live/media
make annotate SPLIT=validation   # both splits must be annotated before step 2
make report                      # published validation/selection/lock reports
```

[`docs/shared/study-runbook.md`](docs/shared/study-runbook.md) carries the exact artifact ids
and the full command sequence with real paths;
[`docs/shared/TODO.md`](docs/shared/TODO.md) records what is still open and
which of those calls are the authors' to make — including the one thing to do
before anyone annotates.

## 1. Collect preferences

```bash
# Media: one file per clip per track, named {clip_id}.mp4 / {clip_id}.wav, plus
# the clip rows to ingest. --media-source source stages the pristine footage
# instead of a render (use it when a render burns a caption into the frame,
# which would anchor an annotator who is judging captions); --mute drops the
# audio stream so visual-track isolation holds in the file, not just the player.
# --track both stages a muted mp4 AND a wav per clip under one row, which is
# what a contract declaring more than one track needs; pass --condition-dir once
# per condition to collect the full clip set.
uv run python scripts/stage_media.py --condition-dir data/video/c1_video_audio \
  --source-dir data/source --track both --media-source source \
  --out media/ --rows data/clips.jsonl

# Corpus: one JSONL row per clip (media_hash, derivatives, optional
# audio_presentation = "audio_only" | "unmuted_video"). Splits are grouped by
# source_video_id, so every condition render of one clip stays on one side.
uv run dpo corpus ingest      --workspace "$W" --contract "$C" --input data/clips.jsonl
uv run dpo corpus lock-splits --workspace "$W" --contract "$C" --artifact-id "$INGEST_ID"

# Caption pairs: generated by the contract's frozen seed policy C0.
# Once per track the contract declares, and per split. A track the contract
# does not declare is refused here.
uv run dpo candidates generate --workspace "$W" --contract "$C" \
  --artifact-id "$REGISTRY_ID" --track audio --split train --dataset-version study/v1

# Pre-enforce the leakage gate BEFORE anyone annotates: views derive refuses a
# candidate whose text near-duplicates one across a split boundary, and finding
# that out afterwards wastes the annotation. This drops both sides of every
# cross-pool collision, re-pairs, and re-freezes with dedup-source lineage.
uv run dpo candidates dedup --workspace "$W" --contract "$C" \
  --artifact-id "$REGISTRY_ID" --artifact-id "$TRAIN_POOL" --artifact-id "$VALIDATION_POOL" \
  --dataset-version study/v2

# Annotator session: tasks with randomized display order, repeats, and
# attention checks per the contract's [annotation] fractions.
uv run dpo annotation export-tasks --workspace "$W" --contract "$C" \
  --artifact-id "$POOL_ID" --artifact-id "$REGISTRY_ID" \
  --out-tasks tasks.json --out-answers answers.json     # answers.json is restricted

uv run dpo annotation serve --tasks tasks.json --media-dir media/ --out responses/

# After each annotator saves (--responses repeats, once per annotator):
uv run dpo annotation ingest --workspace "$W" --contract "$C" \
  --artifact-id "$POOL_ID" --split train --tasks tasks.json --answers answers.json \
  --responses responses/responses-alice.json --responses responses/responses-bob.json
```

The UI serves each clip by its track and per-clip flag — muted video for
visual captions, audio-only (default) or unmuted video for audio captions —
and every judgment records which presentation it saw, so any cross-modal
contamination stays measurable. Choices are recorded against the displayed
order and resolved to canonical candidates exactly once at ingest;
reliability screening (attention checks, repeat consistency, position bias,
response time) applies the contract's preregistered exclusion rules before
aggregation. Position bias takes two knobs, and drops an annotator only when
both fire: the lean must exceed `max_position_bias` *and* an exact two-sided
binomial test must reject a fair coin at `position_bias_alpha`. Either test
alone convicts the wrong people — the effect size alone convicts eight
decisive judgments out of eleven, which chance produces about a fifth of the
time, and the p-value alone convicts a 0.55 lean that distorts nothing.

The report also carries **Krippendorff's α** (nominal, over canonical choices).
It is the study's only real agreement number: an aggregated pair's `agreement`
is the modal share, so at `judgments_per_pair = 2` it can only be 0.5 or 1.0
and credits nothing to chance — a coin scores 0.5 there. α is computed over
primary judgments only, since attention checks have a right answer and repeats
measure the same annotator twice, and it is read through `canonical_choice()`,
because two annotators who both pressed the left button under opposite display
orders endorsed opposite candidates.

## 2. Run the experiments

The nine-condition matrix is code-owned
(`dpo.contracts.study_contract.EXPERIMENT_MATRIX`): the preference arms start
from the frozen SEED reference, SFT_DPO warm-starts from SFT, and a contract
can set hyperparameters but can never move an experiment's initialization,
reference, or objective family. One frozen preference dataset regenerates
every training view (`D_sft`, `D_pair_strict`, `D_pair_all`) bit-identically.

The contract owns every result-affecting knob — seed model identity and init
seed, the C0 decoding mixture and generation seed, view gates, training
seeds and budgets, and per-experiment hyperparameters. **Sweep axes**: a
sweepable knob (`beta`, `epsilon`, `beta_prime`) may be a list
(`beta = [0.1, 0.3]`); each value trains as its own variant, every variant is
validated identically, and selection picks one winner per experiment and
track. Because artifact identities are stage-scoped, extending a sweep axis
recomputes only the new cells — collected preferences are never touched.

**Robustness axis**: `[robustness].flip_rates` adds one more dimension to
the matrix. `views derive` publishes one shared flip manifest per rate — the
exact train pair ids whose chosen/rejected labels swap — and `train run`
retrains every preference arm that trains on `D_pair_strict` (DPO, IPO, CDPO,
RDPO, DRDPO, WDPO, SFT_DPO) once per positive rate on that manifest, so every
method meets exactly the same corrupted labels. Selection scores each
retraining on the same *unflipped* validation pairs as its base cell but
ranks base cells only; the flip-rate curve comes out of `report analyze`.
Three positive rates cost three extra cells per arm; `flip_rates = [0.0]` is
the honest way to opt out.

```bash
# Derive every training view from the frozen preferences, once per declared
# track. Both train and validation splits must already be annotated. Prints
# the view ids and, under "flip_manifests", one id per contract rate.
uv run dpo views derive --workspace "$W" --contract "$C" \
  --artifact-id "$REGISTRY_ID" --artifact-id "$TRAIN_POOL" --artifact-id "$VALIDATION_POOL" \
  --artifact-id "$TRAIN_ANNOTATIONS" --artifact-id "$VALIDATION_ANNOTATIONS" --track audio

# Train every matrix cell, flipped retrainings included. Takes the views AND
# the flip manifests. Resumable: rerunning skips finished cells.
uv run dpo train run --workspace "$W" --contract "$C" \
  --artifact-id "$VIEW_IDS..." --artifact-id "$FLIP_MANIFEST_IDS..." \
  --checkpoint-dir runs/checkpoints \
  --backend-config configs/gemma4/e4b-audio.toml \
  --media-dir media/

# Score every cell, pick one winner per experiment and track, lock.
uv run dpo select run --workspace "$W" --contract "$C" \
  --artifact-id "$VIEW_IDS..." --artifact-id "$CELL_IDS..." \
  --checkpoint-dir runs/checkpoints --backend-config configs/gemma4/e4b-audio.toml \
  --media-dir media/
```

Which backend runs is the contract's `models.seed.implementation`: the
deterministic tiny CPU backend needs no `--backend-config`/`--media-dir` and
runs the whole pipeline offline; the Gemma backend requires one backend TOML
per declared track plus the media directory, and verifies the CUDA device, the
contract's `[backends]` hash pins, and full media coverage **before** touching
the store. LoRA comes from the contract's `[training.lora]` and every attach
is checked to keep all trainable parameters inside the language model — the
media towers stay frozen (peft cannot wrap their `Gemma4ClippableLinear`
modules, so this guard is load-bearing, not decorative). A checkpoint
directory is screened for pickle-family files and symlinks before peft is
allowed to open it.

Training is **resumable by content**: each cell writes its adapter plus a
`cell.json` holding the semantic hash of everything it trained from, so a
crash resumes where it stopped and a changed view or hyperparameter retrains
exactly the cells it affects. Reference log-probabilities are precomputed and
the reference released before the policy trains, so a second model never
occupies device memory. At selection, SEED and SFT are scored once per track
and reused as both reference and policy, halving the run's forward passes.

The backend TOMLs under `configs/gemma4/` carry runtime shape only (model
pin, quantization, gradient checkpointing); the contract can pin their
hashes via `[backends]`.

## 3. Compare the results

```bash
uv run dpo report show    --workspace "$W"                     # or: make report
uv run dpo report analyze --workspace "$W" --contract "$C" \
  --artifact-id "$VALIDATION_REPORT" --artifact-id "$SELECTION_REPORT"
```

`report show` prints every validation report (per-variant accuracy by track
and experiment), selection report (ranking, selected variants with their
hyperparameters), lock manifest, and analysis report in the workspace.

`report analyze` publishes `dpo.analysis-report/v1`, the inferential layer
over the per-pair scores the validation report persists: clip-clustered
bootstrap confidence intervals per experiment at the contract's
`validation.bootstrap_samples`, exact paired sign tests against SEED with
Benjamini-Hochberg correction across the preference arms, a Bradley-Terry fit
over per-pair contests between the selected variants, the preregistered
natural-noise slices for the ranked winner, and the flip-rate robustness
curve of every arm that was retrained on flipped labels. It re-scores
nothing, so the report is a pure function of its two parents: a rerun
republishes to the same id, and every number in it has lineage.

Artifacts are the ground truth — `dpo artifact trace` walks any result back
through its full lineage.

## 4. Export the human-study stimuli

```bash
uv run dpo study export --workspace "$W" --contract "$C" \
  --artifact-id "$LOCK" --artifact-id "$SELECTION_REPORT" --artifact-id "$VALIDATION_REPORT" \
  --artifact-id "$REGISTRY_ID" --artifact-id "$TRAIN_POOL" \
  --track audio --checkpoint-dir runs/checkpoints \
  --backend-config configs/gemma4/e4b-audio.toml --media-dir media/
```

Captions the held-out **study** split with the top-ranked experiment's selected
variant and publishes `dpo.study-export/v1` — the stimuli the human study
serves. Each clip gets a *congruency ladder*: one caption per slider stop
(`[study].rungs` of them — the slider's resolution is the study's independent
variable, so the contract owns it), ordered along a measured axis (below).
Cost on one 3090, measured: about
**143 s per clip** — 33 s generating, 109 s scoring — so ~17 minutes for a
seven-clip study split, once, at a 22.5 GiB peak. Scoring dominates because
each candidate is scored twice against a full-length video (see the ablation
below); the audio-only second pass it replaced was roughly half the price and
measured the wrong thing.

### Congruency is measured, not asserted

For a caption `c` on one clip:

```
congruency(c) = [ logP(c | audio, video) - logP(c | audio, gray) ] / |c|   nats/token
```

How much does *seeing* the clip help explain this sentence? A description of
sound alone gains nothing from the video and scores near zero; one naming the
thing visibly making the sound scores strongly positive; one naming something
not on screen scores **negative**, which is a principled incongruent end rather
than a staged one. It reuses `completion_logprobs` — the same likelihood that
scores preference pairs. In the literature this contrast is a length-normalized
conditional pointwise mutual information, the same quantity contrastive
decoding methods use to *decode*; here it selects stimuli instead.

`gray` is the second pass's ablation, not a deletion: a video of the clip's own
resolution, frame rate and frame count carrying a flat mid-grey field
(`dpo.evaluation.ablation`). Dropping the video instead would remove ~2500 soft
tokens along with the picture, and a model's absolute log-probability moves with
how much context precedes a completion whatever that context contains — so a
per-clip offset of unknown size would ride on every score. Rung order and
spacing survive such an offset because every candidate on a clip shares it; the
claim that does not survive is the one about the axis's zero, which is exactly
what calling a rung *incongruent* asserts.

Candidates are over-generated across conditionings, wordings, and temperatures;
each is scored; and the rungs are **selected** for even spacing on that measured
axis. Writing prompts of increasing "tie sound to sight" strength and declaring
that ordering to be the axis is an assumption no output verifies — the model may
answer rung three less congruently than rung two, leaving the study's
independent variable silently unordered. Two refusals enforce it: a non-monotone
ladder is rejected at publish, and one whose ends differ by less than
`MIN_CONGRUENCY_SPAN` is rejected at selection, because a slider that changes
wording without changing meaning is not a control.

Building the ladder deliberately steps outside the audio track's modality
isolation — congruency is defined against what is actually in frame, and an
audio-only model asked to name the visible source invents one. That path runs
through `stimulus_messages` and `generate_stimulus` / `score_stimulus`, named
for their one purpose so the exception is visible at every call site. Nothing
scored, trained on, or compared against preference data goes through them.

Four further properties are deliberate:

- It requires the **lock**, not just the selection report: configuration
  freezes before any held-out access.
- Reading study clips takes a fenced `human-study` capability, reserved once
  per lock (idempotent, so a crashed run retries; a new lock forces a new
  fence, so captions from two configurations cannot be mixed). Every
  precondition runs *before* the fence opens.
- The protected read verifies each staged media file against the registry's
  recorded derivative hashes, so a study cannot ship captions generated from
  files the corpus never saw.
- The publish **fails** if any caption byte-matches a frozen training
  candidate — otherwise the study would measure memorization and report it as
  caption quality.

The published payload is capability-exempt (`PUBLIC_DERIVED_TYPES`), so a study
web process can read the captions without holding a study capability while the
protected ancestry stays sealed.

## 5. Run the study

```bash
uv run dpo study serve --export study-export.json \
  --media-dir data/live/media --out data/userstudy/responses
```

Each clip runs in two steps. First the participant watches it with sound and
writes what they heard, in their own words, with no sentence on screen — that
answer is only worth collecting before a caption has told them what to have
heard. Then the sentence and the slider appear, and they drag until it best fits
the clip and rate the match. The slider **generates nothing at interaction
time** — it indexes the pre-built ladder, so a drag costs a five-element scan
(~20 ns) rather than a model call; the study document is fetched once at boot
(~4 KB for seven clips).

The control's geometry *is* the measurement: stops sit at their measured
congruency, and the tick marks are placed at those same positions rather than
distributed evenly, so the distance dragged between two captions is their
distance on the axis. Marks pull answers toward themselves, so a mark anywhere
but on its own stop is worse than no mark at all. What reaches the browser is
narrowed to position and text — the winning arm, its validation accuracy, each
rung's score, and the conditioning that produced it all stay in the artifact,
because a participant who can read which stop scored highest has been handed the
answer.

Clip order is drawn per participant from their own participant code, and
recorded with each response as `presentation_index`. The export stays sorted, so
the artifact is unchanged; the order varies where it matters and stays stable
across a reload or a resumed draft.

This is a **separate instrument** from the annotation UI in `dpo.annotation`:
different question, different response schema (`dpo.userstudy-responses/v2`),
different people. Kept apart so neither study's validator can be satisfied by
the other's data. It serves the clip *with* its soundtrack, so
`--media-dir` needs `unmuted_video/` renders staged alongside the corpus.

### The caption instruments

Two further participant instruments elicit caption preferences, built to two
specifications that disagree about the surface. Both serve the same
`unmuted_video/` media, write an event log per participant, and share
`dpo.caption` — the writers, the cache, and the caption budget — so whichever
is not adopted can be deleted whole
([`docs/shared/caption-stack.md`](docs/shared/caption-stack.md) says how). The
documentation is split the same way the code is:
[`docs/README.md`](docs/README.md) is the map.

`dpo session serve` builds [`docs/v1-session/`](docs/v1-session/): the skeleton of the sounds a
caption could mention is the only control surface, a participant shapes each
shot on it and watches the clip again with their captions placed, and a
follow-up on a later day checks recognition and preference. Sources can be
filled from Sa2VA masks (`session link-masks`, then `scaffold --mask-links`).
Operator steps are in [`docs/v1-session/runbook.md`](docs/v1-session/runbook.md);
`make session-demo` serves the fixture over synthetic clips.

`dpo console serve` builds [`docs/v2-console/`](docs/v2-console/): a console of three
controls drives a read-only skeleton, and the caption preference is checked in
session against the default policy rather than on a later day. Its quantities
are computed rather than authored — shots cut on visual composition, audio
labels grouped by mask agreement, saturating visibility against a calibrated
`r0` — and frozen into a hashed configuration artifact that stamps every
caption and log line. Operator steps are in
[`docs/v2-console/runbook.md`](docs/v2-console/runbook.md); `make console-demo`
serves its fixture on port 8778, so both instruments can run side by side.

## 6. Analyze the study

```bash
uv run dpo study ingest --workspace "$W" --contract "$C" --artifact-id "$STUDY_EXPORT" \
  --responses data/userstudy/responses/responses-P01.json \
  --responses data/userstudy/responses/responses-P02.json
```

Reads every participant's saved file, checks each response against the export
it was collected under — the clip must be one the export carries, the caption
must be the rung the response claims at the position the export gave it, the
rating must be on the instrument's scale, one answer per clip — and publishes
two artifacts. `dpo.study-responses/v1` is the record: every validated row,
participants hashed exactly as annotators are. `dpo.study-results/v1` is a
pure function of it: placement on the measured axis (mean chosen position and
congruency, the rung histogram), match rating overall and per rung, the
correlation between a chosen caption's measured congruency and its rating,
placement by presentation order, and per-clip and per-participant tables —
each interval from the cluster bootstrap twice over, resampling clips and
then participants, because both are random effects of this design. A
different analysis republishes the results with the same record as parent;
the record is never rewritten, and the raw files stay on disk as the
append-only source.

## Repository structure

`src/diff/` is the study app. It holds:
- the phase-1 flow, the binding model, and the second caption's steering and
  gate;
- the live phase-2 scheduler and server;
- the Claude writer, the session store and the export.

Its design is in [`docs/v4-diff/`](docs/v4-diff/). `src/dpo/` is the pipeline:

```text
src/dpo/
├── cli/           # one module per command (`ls` here reads as the command
│                  # list); _shared plumbing, _backend live-model resolution
├── core/          # content-addressed artifact store, identity, atomic IO,
│                  # fenced access, text-safety screening
├── contracts/     # study_contract (vocabularies, matrix, validator),
│                  # visual_caption / audio_caption compliance screens
├── data/          # split manifest, D_sft / D_pair_strict / D_pair_all,
│                  # noise calibration, flip manifests, weighting, leakage audit
├── candidates/    # candidate_records + C0 policy, generation, evidence-free
│                  # audits, pair sampler, cross-pool dedup, freeze
├── annotation/    # raw_annotations, collection_tasks, webapp (FastAPI UI),
│                  # aggregation, reliability + exclusions
├── userstudy/     # the human study's page + app: congruency slider over a
│                  # published study export (a separate instrument), and the
│                  # validated reader of what participants saved
├── models/        # shared completion logprob, modality-isolated batches,
│                  # visual_media / audio_media builders, tiny CPU backend,
│                  # gemma4/ (adapter, backend_config, tokenization safety,
│                  # training_backend = real QLoRA with the frozen-tower guard)
├── objectives/    # base protocol + dpo, ipo, cdpo, rdpo, drdpo, wdpo, sft
├── trainers/      # one preference trainer for every preference arm,
│                  # SFT trainer, diagnostics
├── evaluation/    # preference accuracy, caption reuse (the memorization
│                  # gate), ablation, congruency (the measured audiovisual axis)
├── analysis/      # compare (the `report analyze` layer), Bradley-Terry,
│                  # clip-cluster bootstrap + BH correction, robustness slices
│                  # and flip curves, the human study's analysis
└── pipeline/      # stage registry (artifact types + contract slices),
                   # publishing, per-stage modules (corpus/candidate/
                   # annotation/view/training/selection/study/study_results),
                   # sweep expansion, live_runner (resumable matrix over a
                   # backend seam), offline matrix runner, lock manifest,
                   # offline canary
```

File naming follows one rule: every basename is globally unique and says what
the module contains. `uv run dpo stage list` prints the stage registry; the
CLI validates its inputs against the same registry, so documentation cannot
drift from enforcement.

## Reproducibility and integrity

- Artifact identities are hashes of semantic manifests — never timestamps,
  paths, hosts, or PIDs. Each stage's identity covers exactly the contract
  sections it declares in the registry, so an unrelated tweak cannot
  invalidate it.
- A training artifact cannot have validation, test, or study exposure
  anywhere in its recursive ancestry (enforced at publish time); test and
  study split payloads stay sealed behind fenced capabilities.
- Golden tests snapshot one training step per objective
  (`tests/golden/golden_values.json`; regenerate deliberately with
  `make golden` and review the diff).
- wDPO is experimental: pinned to `arxiv_2603.07211v1`, with placeholder
  scalar maps documented in `dpo/objectives/wdpo.py` that must be
  re-verified against the pinned official code before any confirmatory run.
- The heavier confirmatory machinery (automated evidence auditing with claim
  ledgers, the one-shot test reservation) was deliberately removed from the
  default path and is recoverable from git history when that phase starts.
  Nothing of it is stubbed in the tree: the test split stays sealed by
  construction — it is a capability-read role with no capability scope, so
  no reservation can open it — and every artifact type the registry names
  has a producer.

See [`docs/shared/pipeline.md`](docs/shared/pipeline.md) for the invariants and claim
limits, [`docs/shared/study-runbook.md`](docs/shared/study-runbook.md) for the live study's
artifact ids, and [`docs/shared/TODO.md`](docs/shared/TODO.md) for deferred wiring.
