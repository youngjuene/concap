# Pipeline invariants, gates, and claim limits

## Lineage and leakage

Artifact identities are hashes of semantic manifests, not timestamps, paths,
hosts, or PIDs. Execution facts are separate receipts. Each stage's cache
identity covers exactly the contract sections it declares in the stage
registry, and each training cell keys on its resolved variant, so a
hyperparameter tweak invalidates only what read it.

Clip roles come only from the locked registry: splits are computed
group-atomically (one source video, one split; near-duplicate/continuity
links merge groups) before any candidate, annotation, or training stage
exists, and an asserted role can only confirm the seeded assignment. A
training artifact cannot have validation, test, or study exposure in its
recursive ancestry; test and study payloads stay sealed behind fenced
capabilities. Candidate text is screened for near-duplicate leakage across
split boundaries, and SFT rows may draw completions from the train split
only.

## Construction

The two caption tracks use identical clips, seeds, budgets, and candidate
counts, and are isolated end to end: separate input processors (a visual
media batch can never carry an audio tensor and vice versa), separate caption
contracts, checkpoints, and reports. Chosen and rejected completions score
against one structurally shared media input.

Which tracks exist at all is a contract decision, and an undeclared track is
refused rather than merely unused: `candidates generate --track visual` fails
on a contract without `[tracks.visual]`, and a `[backends.*]` entry serving an
undeclared track is refused too. The live street study declares
`[tracks.audio]` only, so everything below describes a one-track run on
two-track machinery.

Candidates come from the frozen collection policy C0 — the contract's seed
model under the contract's decoding mixture and generation seed — with
deterministic compliance and cross-modal lexicon screens (the automated
claim-ledger audit was removed from the default path; human preference is
the quality signal). After annotation begins, candidate text, ids, pair
mappings, source checkpoints, and decoding configurations are immutable; any
change requires a new dataset version and a new collection round.

A pool carrying cross-split near-duplicates is repaired before annotation
rather than discovered after it: `dpo candidates dedup` cuts both sides of
every collision under the leakage audit's own threshold constant and
republishes the pool with a `dedup-source` edge, so the gate that would have
refused `views derive` passes in-band instead. Within-split duplicates are
out of the gate's scope by design.

## Preference data

Raw annotations are append-only and never overwritten by an aggregate. The
response vocabulary is `a_better / b_better / tie / both_unacceptable` with
tie subtypes, preference strength (decisive only), confidence, and reason
tags; choices are recorded against the displayed order and resolved to
canonical candidate identity exactly once. Every judgment records the
presentation it was made under (muted video; audio-only; opt-in unmuted
video), so cross-modal contamination on the audio track stays measurable.

Reliability screening runs before aggregation and publishes a
`dpo.reliability-report/v2` a reviewer can check: per annotator, the judgment
and attention-check counts, attention pass rate, repeat consistency,
left-choice rate with its exact two-sided binomial p-value, fast-response
count, and the exclusion reasons; plus chance-corrected agreement over the
whole round (nominal Krippendorff alpha). Three preregistered rules exclude
an annotator — attention pass rate below `min_attention_pass`, a position
lean that is both larger than `max_position_bias` *and* significant at
`position_bias_alpha`, and fast responses on at least half their rows.
Repeat consistency is reported, not enforced. Excluded rows remain in the raw
store.

Derived views are pure functions of the frozen pool plus retained
annotations: `D_sft` (positively endorsed, deduplicated), `D_pair_strict`
(clear preference, minimum strength and agreement), `D_pair_all` (full
outcome probabilities and difficulty), the natural-noise calibration (train
split only), and one shared flip-index manifest per synthetic rate (train
labels only). Clip-level weighting keeps pair-rich clips from dominating.

## Training and comparison

All preference methods share one completion-only sequence log-probability
implementation (`dpo.models.logprob`): prompt tokens and padding masked,
float32 log-softmax with NaN/overflow guards. On a multimodal backend the
whole processor encoding reaches the forward pass, so a completion is scored
conditioned on its clip's media — scoring from token ids alone would be
silently media-blind and is regression-tested against that. One preference
trainer serves every preference arm through the objective registry;
references are frozen structurally and receive no gradients (tested).

One matrix runner drives every backend through a single seam
(`pipeline.live_runner`): the deterministic CPU backend and the real QLoRA
backend execute identical matrix semantics, asserted cell-for-cell against
the offline runner. Cells are resumable by content — each writes its adapter
plus the semantic hash of everything it trained from, so a crash resumes and
a changed input retrains exactly what it affects. Reference log-probabilities
are precomputed and the reference released before the policy trains, so no
second model occupies device memory. On the real backend every LoRA attach is
checked to keep all trainable parameters inside the language model: the media
towers must stay frozen for the comparison to mean anything, and the tower
modules are ones the adapter library cannot wrap.

The matrix is code-owned: the six direct preference arms initialize from
SEED against a frozen SEED reference on `D_pair_strict` (wDPO may use the
metadata view); SFT_DPO initializes from SFT against a frozen SFT reference
on the same pair view as DPO, with its extra SFT compute visible in every
report. Sweepable loss knobs (beta, epsilon, beta_prime) may be list-valued;
each value trains as its own variant, every variant is validated
identically, and selection picks one winner per experiment and track before
the lock. The wDPO stage toggles are contract keys, so its ablation arms are
expressible without code edits. Verified objective identities: cDPO(eps=0)
== DPO, rDPO(eps=0) == DPO with the divergent eps -> 0.5 excluded, Dr.DPO
equals its hand-computed logsumexp reference, and wDPO with both stages
disabled equals DPO. wDPO remains experimental until the pinned revision is
reproduced.

The matrix carries a robustness axis the contract declares in
`[robustness].flip_rates`. `views derive` publishes one shared flip manifest
per rate — the strict-view pair ids whose chosen/rejected labels swap,
train labels only — and the train stage retrains every preference cell that
trains on `D_pair_strict` once per positive rate on that manifest, so every
method meets exactly the same corrupted labels. Each retraining is a matrix
cell of its own, keyed by its rate, with the manifest as a parent; a WDPO
variant on the metadata view has no flip series, because the manifest is
built over a different pair set. Validation scores every retraining on the
same *unflipped* validation pairs against the same reference as its base
cell, and persists those scores beside the base scores; selection ranks and
locks base cells only. A model trained on corrupted labels is a measurement
of robustness, never a candidate for the lock.

## Analysis

`dpo report analyze` takes one validation report and the selection report
derived from it and publishes `dpo.analysis-report/v1`: clip-clustered
bootstrap intervals per experiment at the contract's
`validation.bootstrap_samples` resample count, exact paired sign tests
against SEED with Benjamini-Hochberg correction, a Bradley-Terry fit over
per-pair contests between the selected variants, the preregistered
natural-noise slices for the ranked winner, and the flip-rate curve of every
selected variant that was retrained on flipped labels — accuracy and log loss
at each rate, rate 0.0 being the base cell. It re-scores nothing: every
number derives from the per-pair scores the validation report persists, so
the report is a pure function of its two parents and a rerun republishes to
the same id. An inferential number therefore has an artifact identity and a
lineage, and `dpo artifact trace` walks it back to the cells.

## Study export and the human study

The human study measures a caption's placement on an audiovisual congruency
axis, and the axis is measured rather than asserted. For a caption `c` on one
clip, congruency is `[logP(c | audio, video) - logP(c | audio, gray)] / |c|`
in nats per token, computed through the same `completion_logprobs` primitive
that scores preference pairs. `gray` is an ablation, not a deletion: a video
of the clip's own resolution, frame rate and frame count carrying a flat
mid-grey field, so the second pass keeps the soft-token count of the first and
no per-clip offset of unknown size rides on the difference.

`dpo study export` over-generates candidates per clip across conditionings,
wordings and temperatures, scores each, and *selects* rungs for even spacing
on that measured axis. Two refusals keep the slider a control rather than a
label: a non-monotone ladder is refused at publish, and one whose ends differ
by less than `MIN_CONGRUENCY_SPAN` is refused at selection.

Gates on the export path: it requires the lock, not merely the selection
report, so configuration freezes before any held-out access; study clips are
read under a fenced `human-study` capability reserved once per lock
(idempotent on retry, and a new lock forces a new fence); staged media is
verified against the registry's derivative hashes; and the publish fails if
any caption byte-matches a frozen training candidate, so the study cannot
measure memorization and report it as caption quality. The published payload
is capability-exempt, so the study web process reads captions while the
protected ancestry stays sealed.

Building a ladder deliberately steps outside the audio track's modality
isolation — congruency is defined against what is in frame, and an audio-only
model asked to name the visible source invents one. That exception is confined
to `stimulus_messages` / `generate_stimulus` / `score_stimulus`, named for
that one purpose so it is visible at every call site; nothing scored, trained
on, or compared against preference data goes through them.

`dpo study serve` is a separate instrument from the annotation UI: different
question, different people, and a different response schema
(`dpo.userstudy-responses/v2`), so neither study's validator can be satisfied
by the other's data. Responses are written as files under `--out`: the
append-only record, never rewritten.

## Claim limits

- The comparison supports claims about these nine conditions under this
  frozen preference dataset and these caption contracts — not about
  preference optimization in general.
- Without the evidence-audit layer, factuality claims rest on human
  preference and the compliance screens alone; audio-track judgments made
  under the unmuted-video presentation are not modality-isolated and must be
  analyzed separately from audio-only judgments.
- Synthetic flip results are calibrated on train labels only and never mix
  into validation labels: a flip-rate curve compares retrainings on corrupted
  train labels against one fixed, clean validation set. The curve says how
  each objective degrades under symmetric label noise at the contract's
  rates on this dataset; it is not a statement about the natural noise rate,
  which `dpo.noise-calibration/v1` estimates separately from annotator
  disagreement.
- The **one-shot test reservation** remains out of the default path: the test
  role is sealed by construction (a capability-read role with no capability
  scope), no command opens it, and no artifact type in the tree refers to a
  confirmatory phase. Every number the pipeline currently reports is
  exploratory, computed on validation. Bradley-Terry ability estimation, the
  clip-clustered intervals, and the robustness curve are published in
  `dpo.analysis-report/v1` with full lineage, but they remain validation
  numbers.
