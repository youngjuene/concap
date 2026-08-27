# ConCap — Comparative Preference Alignment for Audiovisual Congruence Captioning

**Research plan and work programme**
Repository: `youngjuene/concap` · package/CLI `dpo` · drafted 2026-07-30 ·
revised 2026-08-01 (§4.5 rewritten: the user study that exists is a
congruency-ladder study; the three-condition design moved to §4.6)

---

## 1. Summary

Preference optimization methods for language models — DPO and its many
successors — are almost always compared on text-only preference data, on
different datasets, with different base models, under different
hyperparameter budgets. Whether their reported differences survive a
controlled comparison is unknown, and whether any of them holds under
**multimodal** preference data, where the judgment depends on media the model
must actually perceive, is untested.

ConCap proposes to answer that in one controlled setting. We will collect a
frozen corpus of human **sound-caption** preferences over real urban scenes —
judged from the audio alone, so the preference depends on a percept the model
must actually hear — and train a nine-condition matrix of preference
objectives from one checkpoint, one frozen dataset, and one hyperparameter
regime. Every result-affecting knob is owned by a single study contract;
every intermediate is a content-addressed artifact whose identity is a hash
of its semantic manifest. The apparatus already exists and is GPU-proven; the
remaining work is a human data collection round and the experimental runs it
enables.

The study runs **one modality track**. The apparatus is built for two — the
visual and audio tracks are isolated end to end, with separate processors,
contracts, checkpoints and reports, and a visual track was measured and exists
in git history — but the live contract declares only `[tracks.audio]`, on
purpose. The exclusion is enforced rather than remembered: without the block,
`candidates generate --track visual` fails on this contract, so no scene
caption can be generated that nobody has agreed to annotate. What this costs
is a cross-modal claim; what it buys is a collection round two people can
actually finish (§4.2).

The programme then ends where the caption is actually used. A model that wins
on held-out preference accuracy has matched annotators comparing two captions
side by side; it has not been shown to do anything for a viewer. So the
selected models generate **sound captions** — captions whose job is to convey
what can be heard — for the reserved study split, and a user study puts them
in front of viewers on a *measured* axis. Each clip carries a ladder of
captions ordered by how much seeing the clip helps explain the sentence, from
sound-only description to one that names the visible source; the participant
watches with sound, says what they heard, then drags a slider until the
sentence best fits and rates the match (§4.5). The axis is a quantity computed
from the model's own log-probabilities, not an ordering asserted by the prompts
that produced it. A companion design that holds the caption fixed and moves the
*sound* instead — **V+C / V+A+C / V+A′+C** — is specified in §4.6 and is not
built.

The secondary contribution is the apparatus itself: a reproducible,
leakage-gated comparison harness in which "we changed one thing" is
mechanically enforceable rather than asserted in a methods section.

---

## 2. Background and motivation

**The comparison problem.** DPO, IPO, cDPO, rDPO, Dr.DPO and wDPO each claim
an advantage over a predecessor — better calibration, robustness to label
noise, robustness to distribution shift. Those claims come from different
papers with different data, models, and tuning effort. A practitioner
choosing among them has no controlled evidence. The confound is not subtle:
initialization, reference model, view of the data, and per-objective tuning
budget all move together in the published record.

**Why multimodal captioning is the right testbed.** A caption preference is a
judgment about correspondence between a description and a percept. Unlike a
text-only helpfulness rating, it has an external referent, so a wrong
preference is wrong about something. It also exposes a failure mode
text-only benchmarks cannot: a model can score a caption without ever
conditioning on the media. (We found and fixed exactly this defect in our own
Gemma scoring path, and now regression-test against it — a completion's
log-probability must shift when its clip changes.)

**Why audio, and why isolated.** Judging a caption from sound alone is the
harder half of the problem and the less studied one: the judgment cannot be
recovered from the frame, so a preference recorded under audio-only
presentation is evidence about a percept the model must have encoded. Keeping
that isolation in the *file* — a demuxed 16 kHz wav, not a muted player —
means a cross-modal shortcut is not merely discouraged but unavailable, and
the unmuted-video opt-in that does exist is recorded per judgment so its
contamination stays measurable (A3).

The apparatus keeps the visual track's machinery intact and symmetric —
identical clips, seeds, budgets and candidate counts, separate processors,
contracts and reports — so modality could become a within-subject factor if a
second annotation round is ever funded. That symmetry is why consolidating on
a single checkpoint (Gemma 4 E4B) mattered: a future cross-track difference
must not be confounded with model identity. The decision was taken 2026-07-30,
after measuring that the alternatives with a larger language model reach it by
discarding the audio encoder, and hallucinate speech on non-speech street
recordings at every precision. The live study still declares one track (§1).

**Why street scenes.** The stimulus corpus derives from a prior urban
soundscape study: locations in Amsterdam, Bangkok and Singapore selected by
Local Moran's *I* spatial clustering, independently in a video feature space
and an MFCC (audio) feature space, sampling both high-high and low-low
clusters. This gives scenes that are heterogeneous along *both* modalities by
construction, and it gives them geographic and cultural spread — a corpus
where visual and auditory salience genuinely dissociate rather than covary.

**Why the caption must finally be tested on viewers, and what to vary.**
Held-out preference accuracy measures agreement with annotators who compared
two captions against a percept. It does not measure what a caption *does* for
someone watching a scene, which is the only reason to align a captioner at
all. So the programme ends at the point of use — and the question is which
side of the correspondence to move.

Moving the *sound* is the obvious design and the one the parent soundscape
study's own conditions suggest (`c1` V+A, `c2` V+A+C, `c3` V, `c4` V+C): show
the same caption with no audio, with the scene's own audio, and with a
substituted soundtrack, and whatever effect survives substitution is not an
effect of correspondence. It also requires fabricating a stimulus — a scene
paired with audio that is not its own — and staging that nothing in the
apparatus does yet. It is specified in §4.6 and deferred.

Moving the *caption* answers a closely related question with stimuli the
pipeline can already produce and verify. Congruency — how much seeing the clip
helps explain the sentence — is computable from the model's own
log-probabilities, so a clip can carry a ladder of captions from
sound-only description to one naming the visible source, ordered by measurement
rather than by the prompt that produced each rung. Where a viewer places
themselves on that ladder, and how well they say it matches, is a statement
about correspondence made against real audio the scene actually has. That is
§4.5, and it is built.

---

## 3. Specific aims

**A1 — Collect a frozen, modality-isolated human sound-caption preference
dataset.** Elicit forced-choice preferences between candidate captions from
the frozen collection policy C0, presented **audio-only** with a per-clip
opt-in to unmuted video, under preregistered reliability screening. Freeze it
as a versioned artifact chain. Modality isolation holds in the file, not
merely in the player: the audio track's media is a demuxed 16 kHz wav, and a
visual media batch can never carry an audio tensor or the reverse.

**A2 — Run the nine-condition comparison.** Train the code-owned matrix
(SEED, SFT, DPO, IPO, cDPO, rDPO, Dr.DPO, wDPO, SFT→DPO warm start) on that
one dataset, with in-contract hyperparameter sweeps, and select one winner per
experiment on held-out validation preference accuracy. One track, one seed
checkpoint, one hyperparameter regime: the only thing that differs between
cells is the objective (and, for SFT_DPO, the initialization, whose extra
compute is reported).

**A3 — Characterize robustness and contamination.** Measure each objective
under calibrated synthetic label-flip rates (0.0/0.1/0.2/0.3, train labels
only) against the corpus's own natural disagreement rate; and separately
analyze judgments made under the unmuted-video opt-in, which are by definition
not modality-isolated. Note what this aim does *not* include: with one track
declared, there is no cross-modal ranking to test. Whether an objective's rank
survives a change of modality is a question this design cannot answer, and
answering it means restoring `[tracks.visual]` and `[backends.visual]` from
git history and paying for a second annotation round.

**A4 — Reproduce wDPO's pinned revision.** wDPO ships with placeholder scalar
maps and is pinned to `arxiv_2603.07211v1`. Reproduce the official
implementation before wDPO enters any reportable result; until then it is
labelled experimental in every output.

**A5 — Restore the one-shot test reservation and run a confirmatory phase.**
Bradley-Terry ability estimation with clip-cluster bootstrap is no longer
missing — it was rebuilt into `dpo report analyze`, which also carries the
paired tests and the natural-noise slices. What remains removed is the sealed
test reservation itself: the artifact types are reserved and no command
produces them, so every number the pipeline reports today is exploratory.
Restore that path from git history and run the sealed test split exactly once.

**A6 — Measure where viewers place a generated sound caption on the
audiovisual congruency axis.** On the reserved study split, generate a ladder
of sound captions per clip from the selected audio-track model, ordered by a
measured congruency score, and have blinded participants drag a slider to the
caption that best fits the clip they are watching and hearing, rating the
match. The manipulated variable is the caption's congruency with what is on
screen; the clip and its soundtrack are held constant. This is the aim that
turns a preference-accuracy win into a statement about captions doing their
job (§4.5). Its companion — holding the caption fixed and moving the sound —
is §4.6, and is deferred.

---

## 4. Research design

### 4.1 Stimulus corpus

48 street scenes from three cities, balanced by construction. The
annotator-facing condition renders are 4K/60fps and ~10.5 s; the pristine
uncaptioned source clips they derive from — and which the training corpus is
staged from — are 720p and ~10.0 s, a prefix of the same footage starting at
the same instant.

| Factor | Levels | n |
|---|---|---|
| City | Amsterdam / Bangkok / Singapore | 16 each |
| Cluster feature space | video / MFCC | 24 each |
| Cluster type (Local Moran's *I*) | high-high / low-low | 24 each |

Every clip is distinct; no footage appears twice. Four presentation condition
renders exist, 12 clips each — `c1` **V+A** (video with its own audio), `c2`
**V+A+C** (audio plus a Korean audio-description caption burned into the
frame), `c3` **V** (muted video, no audio stream), `c4` **V+C** (muted plus
caption). These are the parent study's own conditions, and §4.6 borrows two of
them by name. They cannot be reused as *stimuli* by either user study, for two
independent reasons: each condition directory holds different footage, so no
clip exists in more than one condition; and their captions are the prior
study's hand-authored text burned into the frame, not a model's. Both study
designs therefore stage from the pristine sources — §4.5 needs one unmuted
render per study clip carrying no caption at all, since its captions are DOM
text a slider swaps; §4.6 would need one render per condition.

The **training corpus** is staged separately from the pristine uncaptioned
source pool, not from the condition renders, for two reasons: the renders burn
captions into pixels (which would anchor an annotator judging captions), and
`c1`/`c3` sample disjoint footage, so no single condition directory can supply
both modalities for the same clip. Staging demuxes each source into a muted
`.mp4` (visual track) and a 16 kHz `.wav` (audio track), so modality isolation
holds in the file, not merely in the player. All 48 clips are staged and
ingested (96 media files, `data/live/clips.jsonl`).

Splits are computed group-atomically on `source_video_id` before any
downstream stage exists — 55/15/15/15 train/validation/test/study, which on
this corpus resolves to **27 train, 7 validation, 7 sealed test, 7 study** in
the locked registry.

**Scale-up path.** The corpus is small. `data/video/metadata/` catalogues 415
locations with significant Moran's *I* across the six city×feature-space
combinations, of which 48 are currently rendered. Expansion is a rendering
task against the parent dataset, not a redesign; because artifact identities
are stage-scoped, extending the corpus recomputes only what it touches.

### 4.2 Preference elicitation

Candidates come from the frozen collection policy **C0**: the contract's seed
model under a fixed decoding mixture (1 greedy, 2 sampled at T=0.7/p=0.9, 1
controlled-error), fixed generation seed, 4 candidates per clip, 3–6 pairs per
clip. After annotation opens, candidate text, ids, pair mappings, source
checkpoints and decoding configurations are immutable; any change requires a
new dataset version and a new collection round.

Collection runs through a local FastAPI instrument. Design features, all
enforced in code:

- **Presentation is recorded per judgment.** Visual = muted video; audio =
  audio-only by default with a per-clip `unmuted_video` opt-in. Cross-modal
  contamination therefore stays measurable rather than assumed absent. The
  contract-level vocabulary is exactly those three (`muted_video`,
  `audio_only`, `unmuted_video`); `substituted_audio_video` survives only as a
  scoped-only presentation string in the annotation web app, with no contract
  support behind it — the A′ manipulation it was reserved for belongs to the
  deferred study of §4.6.
- **Order-blind resolution.** Choices are recorded against the *displayed*
  order and resolved to canonical candidate identity exactly once, at ingest.
- **Response vocabulary:** `a_better / b_better / tie / both_unacceptable`,
  with tie subtypes, preference strength (decisive only), confidence, and
  reason tags.
- **Reliability screening** before aggregation, on preregistered thresholds:
  attention checks (5% of tasks, ≥0.8 pass), repeat consistency (12% of tasks,
  reported per annotator but not an exclusion rule), position bias, and
  minimum response time (1500 ms). Position bias excludes only when the lean
  is both larger than 0.2 *and* significant at α=0.01 on an exact two-sided
  binomial test — either test alone convicts the wrong annotators, and with a
  two-person expert panel an exclusion ends the round. The published
  `dpo.reliability-report/v2` carries those per-annotator rates plus
  chance-corrected agreement (nominal Krippendorff α). Excluded rows stay in
  the raw store; raw annotations are append-only and never overwritten by an
  aggregate.
- **2 judgments per pair** — the panel is the two authors, so two is the
  ceiling rather than a choice. It follows that `min_agreement = 0.6` means
  unanimity (agreement over two raters is 0.5 or 1.0), and the inter-rater
  agreement rate *is* the retention rate.

Volume for the current corpus, audio track: 176 train + 36 validation = **212
tasks per author**, repeats and attention checks included — about 1.5 h each at
25 s/task. That budget is the reason the contract declares one track: two
would double it, and a session two expert annotators cannot finish is a
dataset that does not exist. (Visual-track pools from an earlier two-track
round are still in the workspace; they are orphaned under this contract —
neither deduped nor exportable — and belong to a restored visual arm or to
garbage collection.) Scaling the panel beyond the authors raises
`judgments_per_pair` and makes `min_agreement` a real threshold rather than a
unanimity rule; that is a contract change, not a code change.

### 4.3 Experimental matrix

Nine conditions, **code-owned** (`dpo.contracts.study_contract.EXPERIMENT_MATRIX`).
A contract may set hyperparameters; it can never move an experiment's
initialization, reference, objective family, or data view.

| Condition | Init | Reference | View | Objective |
|---|---|---|---|---|
| SEED | — | — | — | unaligned seed |
| SFT | SEED | — | `D_sft` | sft |
| DPO | SEED | frozen SEED | `D_pair_strict` | dpo |
| IPO | SEED | frozen SEED | `D_pair_strict` | ipo |
| CDPO | SEED | frozen SEED | `D_pair_strict` | cdpo |
| RDPO | SEED | frozen SEED | `D_pair_strict` | rdpo |
| DRDPO | SEED | frozen SEED | `D_pair_strict` | drdpo |
| WDPO | SEED | frozen SEED | `D_pair_strict` / `D_pair_all` | wdpo |
| SFT_DPO | SFT | frozen SFT | `D_pair_strict` | dpo (warm start) |

SFT_DPO's extra compute is reported in every output, so its warm start is
never a hidden advantage.

Three views regenerate bit-identically from the one frozen preference pool:
`D_sft` (positively endorsed, deduplicated, train-split completions only),
`D_pair_strict` (clear preference above minimum strength and agreement), and
`D_pair_all` (full outcome probabilities and difficulty). Clip-level inverse
weighting keeps pair-rich clips from dominating.

**Sweeps.** `beta`, `epsilon` and `beta_prime` may be list-valued; each value
trains as its own variant, every variant is validated identically, and
selection picks one winner per experiment (per declared track) before the
lock. This is how each objective gets a comparable tuning budget instead of an
accidental one.

**Verified objective identities** (already tested): cDPO(ε=0) ≡ DPO,
rDPO(ε=0) ≡ DPO (with the divergent ε→0.5 excluded), Dr.DPO equals its
hand-computed logsumexp reference, and wDPO with both stages disabled ≡ DPO.
Golden tests snapshot one training step per objective.

### 4.4 Training and evaluation protocol

Base model **Gemma 4 E4B-it**, pinned by revision and lock hash, bf16
unquantized, on one RTX 3090 (24 GB). LoRA (r=16, α=32, dropout 0.05) attaches
to language-model attention and MLP projections only; `assert_text_only_lora_scope`
re-checks after every attach that all trainable parameters lie inside the
language model. **The vision and audio towers stay frozen** — an objective that
was allowed to retune the audio encoder would be compared against ones that
were not, and the comparison would be about perception rather than preference.
This guard is load-bearing, not decorative: the towers' 232
`Gemma4ClippableLinear` modules cannot be wrapped by peft on our pins, so a
naive target spec fails outright.

All preference methods share one completion-only sequence log-probability
implementation: prompt tokens and padding masked, float32 log-softmax with
NaN/overflow guards, and the *whole processor encoding* reaching the forward
pass so completions are scored conditioned on their clip's media.

Reference log-probabilities are precomputed and the reference released before
the policy trains, so two models never occupy device memory. Each cell writes
its adapter plus a semantic hash of everything it trained from, so a crash
resumes where it stopped and a changed input retrains exactly the cells it
affects.

Evaluation: per-variant preference accuracy on held-out validation,
compliance screening against the caption contracts, and caption generation
under fixed decoding (T=0, top-p 1.0, 48 new tokens). The inferential layer is
`dpo report analyze` — clip-clustered bootstrap intervals at the contract's
`validation.bootstrap_samples` (10000 on the live study; 200 only on the
canary, where it is a speed fixture), exact paired sign tests against SEED
with BH correction, a Bradley-Terry fit over per-pair contests, and the
preregistered natural-noise slices, and the flip-rate robustness curve of
every arm retrained on flipped labels. It publishes `dpo.analysis-report/v1`
with the two reports as parents, so every inferential number has an artifact
identity and a lineage; it re-scores nothing, and a rerun republishes to the
same id.

### 4.5 Congruency-ladder user study (built)

This is the only place in the programme where a caption is consumed as a
caption rather than scored as a completion. It runs after A2's selection and
lock, on models that have finished training and validation. Both halves —
`dpo study export` and `dpo study serve` — are implemented and tested.

**The measured axis.** The study's independent variable is how much *seeing*
the clip helps explain the sentence. For a caption `c` on one clip:

```
congruency(c) = [ logP(c | audio, video) - logP(c | audio, gray) ] / |c|    nats/token
```

A caption describing sound alone gains nothing from the video and scores near
zero; one naming the thing visibly making the sound scores strongly positive;
one naming something not on screen scores **negative** — a principled
incongruent end rather than a staged one. It reuses `completion_logprobs`, the
same likelihood that scores preference pairs, so nothing here redefines what
likelihood means. In the literature the contrast is a length-normalized
conditional pointwise mutual information; contrastive decoding methods use it
to decode, and here it selects stimuli.

`gray` is an ablation, not a deletion: a video of the clip's own resolution,
frame rate and frame count carrying a flat mid-grey field. Dropping the video
instead would remove ~2500 soft tokens along with the picture, and a model's
absolute log-probability moves with how much context precedes a completion
whatever that context contains — so a per-clip offset of unknown size would
ride on every score. Rung *order* would survive such an offset; the claim that
would not is the one about the axis's zero, which is exactly what calling a
rung incongruent asserts.

**Why measured rather than prompted.** Writing five instructions of increasing
"tie sound to sight" strength and declaring that ordering to be the axis is an
assumption no output verifies — the model may answer rung three less
congruently than rung two, leaving the independent variable silently
unordered. So candidates are over-generated across conditionings, wordings and
temperatures, each is scored, and the rungs are **selected** for even spacing
on the measured axis. Two refusals enforce it: a non-monotone ladder is
rejected at publish, and one whose ends differ by less than
`MIN_CONGRUENCY_SPAN` (0.02 nats/token) is rejected at selection, because a
slider that changes wording without changing meaning is not a control.

**Stimuli.** The reserved study split — 7 clips, group-atomic on
`source_video_id`, sealed behind a fenced `human-study` capability reserved
once per lock, and absent from every training, validation and test artifact's
ancestry. Each clip gets one ladder of `[study].rungs` captions from the
top-ranked experiment's selected variant — the slider's resolution is the
study's independent variable, so the contract owns it. Measured cost on one
3090: ~143 s per clip
(33 s generating, 109 s scoring) at a 22.5 GiB peak — about 17 minutes for the
seven-clip split, once. Publication fails if any caption byte-matches a frozen
training candidate, so the study cannot measure memorization and report it as
caption quality.

Building a ladder deliberately steps outside the audio track's modality
isolation: congruency is defined against what is in frame, and an audio-only
model asked to name the visible source invents one. The exception is confined
to `stimulus_messages` / `generate_stimulus` / `score_stimulus`, named for that
one purpose so it is visible at every call site; nothing scored, trained on, or
compared against preference data goes through them. This is a stated
limitation, not a leak — the stimulus builder sees the video, the compared
models never do.

**Response.** Each clip runs in two steps. First the participant watches it
with sound and writes, in their own words, what they heard — with no sentence
on screen, because that answer is only worth collecting before a caption has
told them what to have heard. Then the sentence and the slider appear and they
drag until it best fits, and rate the match on a five-point scale. Each
response records the caption shown and its position on the axis, the free-text
recall, the match rating, slider moves, time to first move, response time,
replay count, and `presentation_index` — where the clip fell in that
participant's order, so order can be modelled rather than assumed away.

**Design and blinding.** The slider generates nothing at interaction time; it
indexes the pre-built ladder, so a drag costs a five-element scan rather than a
model call. The control's geometry *is* the measurement: stops sit at their
measured congruency and tick marks are placed at those same positions, so the
distance dragged between two captions is their distance on the axis. What
reaches the browser is narrowed to position and text — the winning arm, its
validation accuracy, and each rung's score stay in the artifact, because a
participant who can read which stop scored highest has been handed the answer.
Clip order is drawn per participant from their own participant code and stays
stable across a reload or a resumed draft.

The instrument is deliberately separate from the annotation UI: different
question, different people, different response schema
(`dpo.userstudy-responses/v2`), so neither study's validator can be satisfied
by the other's data. It serves the clip *with* its soundtrack, so the media
directory needs `unmuted_video/` renders staged alongside the corpus.

**Analysis.** `dpo study ingest` validates every participant's saved file
against the export it answers and publishes two artifacts:
`dpo.study-responses/v1`, the record (participants hashed), and
`dpo.study-results/v1`, a pure function of it — placement on the measured
axis (mean chosen position and congruency, the rung histogram), match rating
overall and per rung, the correlation between a chosen caption's measured
congruency and its rating, placement by presentation order, and per-clip and
per-participant tables, each interval from the cluster bootstrap in
`dpo.analysis` resampling clips and then participants, because both are
random effects of this design. The mixed-effects fit with participant and
clip as random effects is downstream work over the persisted rows. The
free-text recall gives an independent handle on what a participant actually
heard, collected before any caption could prime it.

**Two decisions still open:**

1. **Caption language.** Both caption contracts prompt for one short *English*
   sentence, while the parent corpus's own caption renders carry Korean
   audio-description text. If the panel is Korean-speaking, either the contract
   prompt or a translation step has to be chosen — and a translation step puts
   a translator between the model and the measured effect, which should be
   stated as a limitation rather than absorbed silently.
2. **Systems under test and panel size.** The export captions one clip set
   from the top-ranked variant, so the default study measures *one* system
   across the congruency axis rather than ranking systems. Comparing arms means
   one export per arm and a decision about whether a participant sees more than
   one — currently unspecified. The rung count is already a contract key
   (`[study].rungs`), so two exports cannot silently differ in resolution.

### 4.6 Three-condition sound-caption study (V+C, V+A+C, V+A′+C) — deferred

The complement to §4.5: hold the caption fixed and move the sound. Specified
here because it answers a question the ladder cannot, and deferred because
nothing in the repository implements it — no A′ staging path, no per-condition
stimulus rendering, and `substituted_audio_video` exists only as a string in
the annotation web app.

| Condition | Video | Audio | Caption | What it isolates |
|---|---|---|---|---|
| **V+C** | yes | none | generated | the caption as *substitute* — the viewer's only access to the soundscape |
| **V+A+C** | yes | the clip's own | generated | the caption as *complement* — the viewer can hold it against what they hear |
| **V+A′+C** | yes | substituted | generated | the caption under *incongruence* — described sound and heard sound disagree |

Two contrasts follow directly: **V+C vs V+A+C** is how much of the caption's
effect depends on the sound being available, and **V+A+C vs V+A′+C** how much
depends on the sound being the *right* sound. An effect flat across all three
is an effect of prose, not of correspondence. Condition would be
between-subjects at the session level, each participant seeing each clip once
with system assignment rotated (a Latin square over clip × system); for *N*
systems, 3*N* participants per condition gives three responses per
clip × system × condition cell, ≈7 responses and ~10 minutes per session.

Before it could run, three things need deciding, and the first is a build:

1. **What A′ is.** The default is a donor recording from the parent dataset
   that is *not* one of the 48 corpus clips: real street audio, with the
   stimulus still derived from exactly one corpus source clip, so the
   group-atomic guard keyed on `source_video_id` has no edge case. An in-corpus
   donor would make one stimulus depend on two corpus clips and needs an
   explicit rule; synthetic ambience avoids the guard entirely but is the least
   ecologically valid.
2. **Caption language** — as §4.5.
3. **Systems under test and panel size.** Nine (SEED plus eight winners) at 27
   participants per condition is 81; a reduced set of SEED, SFT and the winning
   preference objectives at four systems needs 12 per condition, 36 total. The
   reduced set is the default, with SEED as the unaligned floor.

---

## 5. Existing apparatus

The instrument is built. ~15.6k lines across `src/dpo/`; 255 tests (252
passing, 3 skipped behind the live-GPU gate), `ruff` and `mypy --strict`
clean, and a `make smoke` health check that runs every pipeline stage
end-to-end on synthetic fixtures on CPU — the flipped-label retrainings
included — and verifies a warm rerun reuses the same report artifact with zero
recomputation. Test counts re-measured 2026-08-25.

Governing invariants:

- **Contract-owns-everything.** Seed model identity and init seed, C0 decoding
  mixture and generation seed, view gates, training seeds and budgets,
  per-experiment hyperparameters, frame budget — all in one TOML. Backend
  TOMLs carry runtime shape only (model pin, quantization, checkpointing) and
  the contract pins their hashes.
- **Content-addressed identity.** Artifact ids are hashes of semantic
  manifests — never timestamps, paths, hosts, or PIDs. Each stage's identity
  covers exactly the contract sections it declares, so an unrelated tweak
  cannot invalidate it. `dpo artifact trace` walks any result to its roots.
- **Leakage is a publish-time error.** A training artifact cannot have
  validation, test, or study exposure anywhere in its recursive ancestry. Test
  and study payloads stay sealed behind fenced capabilities. Candidate text is
  screened for near-duplicate leakage across split boundaries.
- **One code path for canary and study.** Which backend executes is a contract
  field. The deterministic CPU backend and the real QLoRA backend run
  identical matrix semantics, asserted cell-for-cell.

**GPU-proven to date:** live C0 generation on real media; real E4B SFT and DPO
cells training, checkpointing, round-tripping, and resuming from a fresh
runner; the full 24 GB memory budget measured (see §8).

**Current live workspace** (`artifacts/street/`): 48 clip-registry shards, one
locked registry, one corpus ingest, and ten frozen candidate pools — the two
deduped audio pools now in use, their pre-dedup parents, four earlier audio
rounds, and two visual pools orphaned by the single-track decision. Annotation
sessions are exported (212 tasks per author) but no annotations are ingested;
no views, no cells, no reports, no lock — collection has not yet run.

**What the user study can already lean on, and what it cannot.** Built and
tested: the measured congruency axis and its gray-video ablation
(`dpo.evaluation.congruency`, `dpo.evaluation.ablation`), ladder selection with
its monotonicity and span refusals, `dpo study export` behind the lock and the
fenced capability, and `dpo study serve` with the participant app and the v2
response schema. Not built: any consumer of the responses (they are files, not
artifacts), A′ staging for §4.6 (`scripts/stage_media.py` offers `audio_only`
and `unmuted_video` only), and per-condition stimulus rendering. Note also that
the ladder study needs `unmuted_video/` renders staged for serving, which the
training corpus staging does not produce — it mutes every video it writes.

---

## 6. Work programme

Durations are working estimates, not commitments.

### Phase 0 — Apparatus (complete)
Pipeline, contracts, objectives, trainers, both backends, live generation and
training proven on-GPU, stimulus corpus staged and ingested.

### Phase 1 — Close the E4B consolidation (complete, 2026-08-26)
The 24 GB training budget had been measured against a 4-bit base; the audio
track requires unquantized bf16 (4-bit E4B collapses to boilerplate on
non-speech audio — measured, root-caused to the quantized language model, not
the tower).

- Re-measured at bf16 on the declared (audio) track: at the contract's
  `batch_size = 4`, every preference arm trains at a 20.85 GiB peak on one
  3090 — identical across arms, with about 2.7 GiB of headroom
  (`configs/gemma4/e4b-audio.toml`).
- `[backends.audio]` config hash re-pinned in the same commit.
- No artifact regeneration was needed: `backends` appears in no stage slice
  but the whole-contract hash, and every live artifact still verifies.

**Deliverable met:** a live contract whose preference arms are measured to fit.

### Phase 2 — Collection round (~2–3 weeks)
Freeze the candidate pool, export tasks, recruit and run annotators, ingest,
screen, aggregate, freeze the preference dataset. The pools are deduped and
the sessions are already exported (212 tasks per author); what remains is the
annotation itself and the ingest.

**Deliverables:** frozen preference artifact chain; the
`dpo.reliability-report/v2` (attention pass rates, repeat consistency,
position bias with its binomial p, Krippendorff α, exclusions); the corpus's
natural disagreement rate.

### Phase 3 — Exploratory matrix (~2–3 weeks, GPU-bound)
Derive the three views; train all cells across the sweep; validate; select one
winner per experiment; lock. Then `dpo report analyze` for the inferential
layer.

**Deliverables:** per-variant validation accuracy by experiment; selection
report with the winning hyperparameters; lock manifest; the analysis document
(intervals, paired tests vs SEED, Bradley-Terry ranking). First answer to the
primary question — do these objectives differ at all on multimodal preference
data, once they are given the same data, checkpoint, and tuning budget?

### Phase 4 — Robustness and wDPO (~2 weeks, parallel with Phase 3)
Synthetic flip sweep at 0.0/0.1/0.2/0.3 against the natural noise
calibration. Separate analysis of unmuted-video audio judgments. Reproduce
wDPO's pinned revision (A4) and replace the placeholder scalar maps.

The wiring is in place: `train run` retrains every pair_strict preference arm
once per positive `[robustness].flip_rates` entry on the shared flip
manifest, `select run` scores each retraining on the unflipped validation
pairs, and `report analyze` publishes the flip-rate curves. On the live
contract that is 21 extra E4B cells (three rates × seven arms) on top of the
9 base cells; `flip_rates = [0.0]` opts out honestly if the curves will not be
reported.

### Phase 5 — Confirmatory and congruency-ladder user study (~3–4 weeks)
Two strands, one sealed-data event each.

*Confirmatory (A5).* Restore the one-shot test reservation from git history —
Bradley-Terry and the clip-cluster intervals are already in `report analyze`,
so this strand is now the sealed-split machinery alone. Run the test split
exactly once.

*User study (A6).* Fix the two open decisions of §4.5; stage `unmuted_video/`
renders; run `dpo study export` (~17 min of GPU for seven clips); build the
response reader and the `dpo.study-results/v1` producer, which is the one
missing piece of the path; recruit an independent blinded panel; run and
analyze. Everything but the export itself is off the critical path.

**Deliverables:** confirmatory result with intervals, plus the full artifact
lineage from raw clip to reported number; and the user-study result — where
participants place the caption on the measured axis, their match ratings, and
the free-text recall collected before any caption was shown.

### Phase 6 — Release
Paper, released preference dataset, released apparatus with the canary as its
executable specification.

---

## 7. Resources

| Item | Requirement |
|---|---|
| GPU | 2× RTX 3090-class (24 GB); one is the reference configuration. Phases 1, 3, 4 are GPU-bound. |
| Storage | Content-addressed workspace plus checkpoints per matrix cell, plus one `unmuted_video/` render per study-split clip for serving; the 48-clip corpus is small, expansion is the driver. |
| Annotators (Phase 2) | The two authors, 212 tasks ≈ 1.5 h each; scales linearly with clip count, and doubles if a visual arm is ever restored. |
| Participants (Phase 5) | Independent blinded panel for §4.5, ~10 min per session (7 clips); panel size follows from the open decision on systems under test. §4.6, if it is ever built, needs 3*N* per condition — 36 for the four-system default, 81 for all nine. |
| Software | Linux, Python 3.11, `uv ≥ 0.11`, locked CUDA/PyTorch stack. Fully locked (`uv.lock`, checked in `make check`). |
| Model access | `google/gemma-4-E4B-it` pinned at revision `a4c2d58`; cached locally. |

---

## 8. Risks and mitigations

**Corpus size limits statistical power.** ≈26 training clips is small for
preference optimization. *Mitigation:* clip-cluster bootstrap for intervals
(already implemented in `dpo.analysis`); pre-planned expansion from the 415
catalogued cluster locations; and honest reporting that Phase 3 is exploratory.
This is the single largest threat to the strength of the conclusions and
should be revisited before Phase 5.

**The user study inherits that size.** 7 study clips is a narrow base, and
clip is the unit that generalizes, not response. *Mitigation:* every
participant sees every clip, so participants buy independent replications of
each clip rather than more clips; intervals come from the clip-cluster
bootstrap; and expanding the study split is a staging task, not a redesign. If
the corpus grows before Phase 5 (§4.1), the study split grows with it.

**The ladder can refuse to build.** A clip whose candidates score non-monotone,
or whose ends fall within 0.02 nats/token, is rejected rather than shipped —
correct behaviour, but on 7 clips a repeated refusal is a study that cannot
run. *Mitigation:* the refusals are per clip and the candidate set is
over-generated across conditionings, wordings and temperatures, so the first
response is more spread rather than a threshold edit; a threshold that has to
move should move before any participant is recruited, not after.

**Temporal density, if a visual arm is restored.** Not a live risk under the
single-track contract, but the reason restoring one is not free. The processor
emits exactly 70 soft tokens per frame at any input resolution, and 70 is the
floor of the allowed per-frame budget — so sequence length is
`frames × 70 + text` and frame count is the *only* lever. Two frames fit the
whole matrix on 24 GB; four OOM. Two frames over a 10 s clip is two stills,
while the visual caption contract asks for actions and scene changes.
*Mitigations, in order of preference:* (a) shorten clips so the same two frames
cover less time; (b) more VRAM; (c) a model preparation path that avoids
upcasting the 2 GB embedding to fp32 — the one untried memory lever, and one
that needs care rather than a patch. Gradient checkpointing and frame
resolution are measured and closed. The ladder study's export does pay this
cost today (it scores against full-length video), and it fits: 22.5 GiB peak.

**A′ has no staging path, and the wrong choice would contaminate a split.**
`scripts/stage_media.py` can stage audio-only and unmuted video, not a
substituted soundtrack. This now costs nothing until §4.6 is built.
*Mitigation:* the default there (an out-of-corpus donor recording) keeps each
stimulus derived from one corpus source clip, so the existing group-atomic
guard covers it unchanged; an in-corpus donor is allowed only with an explicit
rule that forbids study-split audio over any other split's video and vice
versa.

**Stimulus building sees the video the audio models never do.** Ladder
construction conditions on the frame by design (§4.5), which is a deliberate
exception to the audio track's modality isolation. *Mitigation:* the exception
lives in three named functions used only for stimuli, nothing on the
preference path calls them, and it is reported as a property of the study
rather than left for a reader to discover.

**Caption language may not match the panel.** The caption contracts produce
English; the parent renders carry Korean. *Mitigation:* decided before the
study runs (§4.5), and whichever way it goes it is reported — a translated
caption is a translated caption, and the claim is limited accordingly.

**wDPO is not yet reproduced.** *Mitigation:* A4 gates it out of every
reportable result until the pinned revision is reproduced; it is labelled
experimental in the code and in every report.

**Annotator disagreement swamps the signal.** *Mitigation:* the natural noise
calibration from Phase 2 tells us this *before* Phase 3 spends GPU time;
`D_pair_strict` gates on agreement and strength; the flip sweep quantifies how
much noise each objective tolerates.

**Cross-modal contamination on the audio track.** *Mitigation:* the unmuted-
video opt-in is recorded per judgment and analyzed separately; audio-only is
the default presentation for collection.

**Silent media-blindness in scoring.** *Mitigation:* found once, fixed, and
now regression-tested — the shared log-probability path must condition on
media.

---

## 9. Scope and claim limits

Stated up front, and enforced in `docs/pipeline.md`:

- The comparison supports claims about **these nine conditions, under this
  frozen preference dataset, under this one audio caption contract** — not
  about preference optimization in general, and not about how the ranking
  behaves in another modality: one track is declared, so modality is a
  constant here, not a factor.
- Without the evidence-audit layer (removed from the default path),
  factuality claims rest on human preference and the deterministic compliance
  screens alone.
- Audio-track judgments made under the unmuted-video presentation are not
  modality-isolated and are reported separately.
- Synthetic flip results are calibrated on train labels only and never mix
  into validation labels — and until a flip-manifest consumer exists, no
  robustness curve is derivable from the default path at all.
- The user study supports claims about **where this panel places these
  captions on the measured congruency axis, and how well they rate the match**,
  on the reserved study split, in the caption language actually shown. The axis
  is a model-internal quantity: a rung's congruency is what the scoring model
  finds explainable given the frame, which is why participant placement against
  it is the finding rather than a manipulation check.
- The user study measures a caption's effect on viewers, not its factuality;
  a caption can be judged to fit and still be wrong about the scene.
- No claim about presentation conditions. V+C / V+A+C / V+A′+C (§4.6) is not
  built, so nothing in this programme currently speaks to how a caption's
  effect depends on the sound being present or being the right sound.
- **Confirmatory claims require Phase 5.** Until the sealed test reservation is
  restored and run, every number is exploratory — computed on validation — and
  will be labelled as such.

---

## 10. Status at a glance

| Aim / component | State |
|---|---|
| Pipeline, contracts, objectives, trainers | Complete; 252/255 tests passing (3 live-gated), lint + `mypy --strict` + offline canary green |
| Stimulus corpus staged and ingested | Complete — 48 clips, media staged for both modalities, in `artifacts/street/` |
| Track scope | **One track declared** (`[tracks.audio]`); the visual block and its backend are removed from the live contract and recoverable from git history |
| Live C0 generation on real media | Proven on-GPU |
| Live training (SFT, DPO cells) | Proven on-GPU under a 4-bit base |
| E4B at bf16 unquantized | Decided 2026-07-30; measured 2026-08-26 — 20.85 GiB peak per preference arm at `batch_size = 4` on one 3090, `[backends.audio]` hash re-pinned (Phase 1 closed) |
| Candidate pools | Audio train + validation deduped and exported as sessions (212 tasks/author); visual pools from an earlier two-track round are orphaned under this contract |
| Human preference collection | **Not started** (Phase 2) — the critical path |
| Nine-condition matrix result | Blocked on Phase 2 |
| Inferential analysis (`report analyze`) | Built — clip-clustered CIs, paired tests + BH, Bradley-Terry, noise slices, flip-rate curves; publishes `dpo.analysis-report/v1` with lineage |
| Robustness axis (flip manifests) | Wired — `train run` retrains each pair_strict arm per rate, `select run` scores them, `report analyze` draws the curves; 21 extra E4B cells on the live contract |
| wDPO reproduction | **Not started** (A4) |
| Congruency-ladder user study (A6, §4.5) | **Built** — measured axis, gray ablation, ladder refusals, `study export` behind the lock, `study serve` + v2 responses, `study ingest` → `dpo.study-responses/v1` + `dpo.study-results/v1`; `[study].rungs` is a contract key. Still an operator step: `unmuted_video/` staging for serving. Two decisions open |
| Three-condition study (§4.6) | **Not built**, deferred — no A′ staging, no per-condition rendering |
| Confirmatory test reservation | Removed by design; recoverable from git history (Phase 5) |

The critical path is Phase 2; everything downstream of it is compute, and the
compute is already wired — the user study's response reader and results
producer included.
