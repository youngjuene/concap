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

Candidates come from the frozen collection policy C0 — the contract's seed
model under the contract's decoding mixture and generation seed — with
deterministic compliance and cross-modal lexicon screens (the automated
claim-ledger audit was removed from the default path; human preference is
the quality signal). After annotation begins, candidate text, ids, pair
mappings, source checkpoints, and decoding configurations are immutable; any
change requires a new dataset version and a new collection round.

## Preference data

Raw annotations are append-only and never overwritten by an aggregate. The
response vocabulary is `a_better / b_better / tie / both_unacceptable` with
tie subtypes, preference strength (decisive only), confidence, and reason
tags; choices are recorded against the displayed order and resolved to
canonical candidate identity exactly once. Every judgment records the
presentation it was made under (muted video; audio-only; opt-in unmuted
video), so cross-modal contamination on the audio track stays measurable.
Reliability screening (attention checks, repeats, position bias, response
time) applies the preregistered exclusion rules before aggregation; excluded
rows remain in the raw store.

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

## Claim limits

- The comparison supports claims about these nine conditions under this
  frozen preference dataset and these caption contracts — not about
  preference optimization in general.
- Without the evidence-audit layer, factuality claims rest on human
  preference and the compliance screens alone; audio-track judgments made
  under the unmuted-video presentation are not modality-isolated and must be
  analyzed separately from audio-only judgments.
- Synthetic flip results are calibrated on train labels only and never mix
  into validation labels.
- The confirmatory apparatus (one-shot test reservation, blinded final human
  study, Bradley-Terry analysis integration) is intentionally out of the
  default path; confirmatory claims require restoring it from git history
  first.
