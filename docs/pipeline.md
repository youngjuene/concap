# Pipeline invariants, gates, and claim limits

## Lineage and leakage

Artifact identities are hashes of semantic manifests, not timestamps, paths,
hosts, or PIDs. Execution facts are separate receipts. Clip roles come only
from the locked registry: splits are computed group-atomically (one source
video, one split; near-duplicate/continuity links merge groups) before any
evidence, candidate, pair, annotation, or training stage exists, and an
asserted role can only confirm the seeded assignment. A training artifact
cannot have validation, test, or study exposure in its recursive ancestry.
Candidate text is screened for near-duplicate leakage across split
boundaries, and SFT rows may draw completions from the train split only.

Protected test and study pools require a fenced capability. Confirmatory test
authority reserves the lock-manifest/test-split/generation/evaluation/analysis
semantic hash before the first protected read and permits only an identical
resume.

## Construction

The two caption tracks use identical clips, seeds, budgets, and candidate
counts, and are isolated end to end: separate input processors (a visual
media batch can never carry an audio tensor and vice versa), separate
evidence schemas and providers (an audio request has no visual key, hash, or
vocabulary — and the reverse), separate caption contracts, checkpoints, and
reports. Chosen and rejected completions score against one structurally
shared media input.

Candidates come from the frozen collection policy C0 (normally the
task-capable seed SEED, bootstrapped on a disjoint corpus if the gate fails),
with at least four candidates per clip, 20-30% challenge candidates, and
content-derived candidate ids. After annotation begins, candidate text, ids,
pair mappings, source checkpoints, decoding configurations, and audit
versions are immutable; any change requires a new dataset version and a new
collection round.

The evidence layer never defines a reference caption. Claims are proposed
from provider output with `uncertain` status and move only through explicit
human audits with provenance; automated outputs are never silently promoted.
Deterministic lexical screens (cross-modal lexicons, ledger-form matching)
are recall aids and structural gates — claim semantics stay human.

## Preference data

Raw annotations are append-only and are never overwritten by an aggregate.
The response vocabulary is `a_better / b_better / tie / both_unacceptable`
with tie subtypes, preference strength (decisive only), confidence, and
reason tags; choices are recorded against the displayed order and resolved to
canonical candidate identity exactly once. Reliability screening (attention
checks, repeats, position bias, response time) applies the preregistered
exclusion rules before aggregation; excluded rows remain in the raw store.

Derived views are pure functions of the frozen pool plus retained
annotations: `D_sft` (positively endorsed, audit-gated, deduplicated —
less-bad outputs from both-unacceptable pairs never qualify, and `chosen`
never automatically implies `sft_eligible`), `D_pair_strict` (clear
preference, minimum strength and agreement, zero both-bad), `D_pair_all`
(full outcome probabilities and difficulty), the natural-noise calibration
(train split only — robust hyperparameters are never estimated from test),
and one shared flip-index manifest per synthetic rate (train labels only).
Clip-level weighting (inverse pair count or cap) keeps pair-rich clips from
dominating.

## Training and comparison

All preference methods share one completion-only sequence log-probability
implementation (`dpo.models.logprob`): prompt tokens and padding masked,
float32 log-softmax with NaN/overflow guards, total sequence logprob by
default and length normalization only as a flagged ablation. One preference
trainer serves DPO, IPO, CDPO, RDPO, DRDPO, WDPO, and SFT_DPO through the
objective registry; references are frozen structurally and receive no
gradients (tested), and reference logps may be precomputed.

The matrix is code-owned: DPO, IPO, CDPO, RDPO, DRDPO, and WDPO initialize
from SEED against a frozen SEED reference on `D_pair_strict` (WDPO may use the
metadata view); SFT_DPO initializes from SFT against a frozen SFT reference
on the same pair view as DPO, and its extra SFT compute is visible in every
report (pipeline and compute-matched comparisons). Sweepable loss knobs
(beta, epsilon, beta_prime) may be list-valued in the contract; each value
trains as its own variant cell keyed on its resolved hyperparameters, every
variant is validated identically, and selection picks one winner per
experiment and track before the lock — the wDPO stage toggles
(enable_correction/enable_winsorization) are contract keys, so its ablation
arms are expressible without code edits.
Verified objective identities: cDPO(eps=0) == DPO, rDPO(eps=0)
== DPO with the divergent eps -> 0.5 excluded and the debiasing distinguished
from label smoothing by test, Dr.DPO equals its hand-computed logsumexp
reference and is batch-level by construction, and wDPO with both stages
disabled equals DPO with correction weights capped at 0.5 and all tail
statistics detached. wDPO remains experimental until the pinned revision is
reproduced; the placeholder scalar maps are documented in the module and
snapshotted by golden tests.

## Evaluation, selection, and the study

One generation adapter with one frozen decoding budget serves every model;
generated captions must not byte-match frozen training candidates (reuse rate
is reported). Automated checks are secondary to human preference and claim
support: compliance (length, sentence form, duplicates, repetition,
modality-violation rate), claim-grounded metrics against the audited ledger,
and typed external boundaries for CLIPScore/FENSE/CLAP that report
`blocked_pending_external_operation` instead of proxying. Reports distinguish
"no unique reference caption exists" from "no auditable evidence exists" via
ledger audit coverage.

Validation reports preference accuracy, log loss, calibration, and margins,
sliced by difficulty, agreement, and track. The lock manifest freezes
checkpoints, processor, preprocessing, decoding, metric versions, the study
interface, exclusion criteria, and the statistical plan before any test read.
The study is a blinded incomplete-block design over all 36 experiment pairs with
balanced exposures, deterministic A/B randomization, clip-disjoint blocks,
and a blinding lexicon scan; model identity exists only in the restricted
randomization manifest. Analysis is Bradley-Terry with clip-clustered
bootstrap intervals and Benjamini-Hochberg correction; annotator-aware mixed
models are an approved extension, not a silent substitute.

## Claim limits

- The comparison supports claims about these nine conditions under this
  frozen preference dataset and these caption contracts — not about
  preference optimization in general.
- Automated similarity metrics never overrule human preference or claim
  support.
- Synthetic flip results are reported separately for known-epsilon and
  estimated-epsilon conditions and never mix train-time flips into
  validation or test labels.
- LLM/provider outputs may construct evidence and candidates; final claim
  statuses, exclusions, metrics, and statistics remain human/deterministic
  with no LLM-evaluator ancestry.
