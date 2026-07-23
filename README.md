# ConCap: Comparative Preference Alignment for Audiovisual Congruence Captioning

ConCap is a reproducible pipeline (package: `dpo`) for comparing nine
preference-alignment conditions (SEED, SFT, DPO, IPO, CDPO, RDPO, DRDPO,
WDPO, SFT_DPO) on two independent captioning tracks — **visual captions**
from video with the audio removed, and **audio captions** from the audio
stream alone — using **one frozen human-preference dataset**. Every production
input is a typed, content-addressed artifact; arbitrary paths to datasets,
checkpoints, provider responses, or protected payloads are rejected.

The pipeline order is normative and deliberately puts candidate construction
and preference collection BEFORE any experimental fine-tuning:

```text
contract
--> corpus / immutable group-level splits
--> modality-isolated evidence + claim ledger
--> audited candidate construction (frozen policy C0)
--> frozen candidate pool
--> human preference collection
--> preference audit + derived dataset views
    --> SEED    task-capable seed          (never trained)
    --> SFT     chosen-only SFT            (D_sft)
    --> DPO     direct DPO                 (D_pair_strict, from SEED)
    --> IPO     direct IPO                 (D_pair_strict, from SEED)
    --> CDPO    direct cDPO                (D_pair_strict, from SEED)
    --> RDPO    direct rDPO                (D_pair_strict, from SEED)
    --> DRDPO   direct Dr.DPO              (D_pair_strict, from SEED)
    --> WDPO    direct wDPO                (strict or metadata view, from SEED)
    --> SFT_DPO SFT-to-DPO                 (same pair view as DPO, from SFT)
--> common validation and model selection
--> configuration lock
--> test-once
--> confirmatory human study (blinded incomplete block)
--> statistical analysis (Bradley-Terry, clip-clustered bootstrap)
```

The initialization rules are code-owned (`dpo.contracts.study_contract.EXPERIMENT_MATRIX`):
DPO, IPO, CDPO, RDPO, DRDPO, and WDPO start from SEED with a frozen SEED
reference; SFT_DPO is the explicit warm-start condition (SEED → SFT → SFT_DPO)
with a frozen SFT reference; a study contract can set hyperparameters but can
never move an experiment's initialization, reference, or objective family.

## 0. Install and verify

Requirements: Linux, Python 3.11, `uv >= 0.11`, Git. Real training
additionally requires NVIDIA drivers compatible with the locked CUDA/PyTorch
stack (two RTX 3090-class GPUs are the reference setup).

```bash
uv sync --dev
uv sync --project providers
make check      # ruff + mypy --strict + pytest + lockfile checks
```

## 1. Run the offline end-to-end canary first

```bash
make smoke
```

`make smoke` runs a cold canary in a fresh workspace, a warm canary in the
same workspace, verifies every artifact, and checks that the warm run reuses
the same report artifact with `provider_calls: 0`. It also exercises one live
boundary and expects JSON status `blocked_pending_external_operation` with no
side effects. Equivalent manual commands:

```bash
uv run dpo canary run --workspace artifacts/canary --contract configs/study/canary.toml
uv run dpo artifact verify --workspace artifacts/canary --all
```

The cold canary executes every stage on synthetic fixtures with the tiny CPU
backend: contract lock, corpus ingest, immutable splits, pinned fixture
evidence for both tracks, audited claim ledgers, frozen per-split candidate
pools, synthetic annotation (repeats, attention checks, counterbalanced
display), reliability screening, derived views (`D_sft`, `D_pair_strict`,
`D_pair_all`, noise calibration, shared flip manifests), the leakage audit,
all 18 matrix cells (9 experiments x 2 tracks) with real optimizer steps,
validation scoring and selection, the configuration lock, a fenced test-once
read, a blinded study export, and a Bradley-Terry analysis. Two integrity
oracles must also pass: a training artifact with test ancestry is rejected,
and payload corruption is detected.

## 2. The study contract

`configs/study/canary.toml` is the synthetic-canary contract. For live work,
copy it, set `execution_class = "live"`, and replace every fixture model,
hash, and policy with an approved immutable value. Never edit a locked
contract after seeing results.

```bash
uv run dpo contract validate --contract configs/study/canary.toml
uv run dpo contract lock --contract configs/study/canary.toml --out runs/contract.lock.json
```

The contract owns every result-affecting knob: split seed and fractions, both
caption contracts (prompt, 8-30 word bounds, clip duration), the seed model
identity and init seed, the C0 decoding mixture, generation seed, and
challenge composition, pair-sampling bounds, annotation quality thresholds
and preregistered exclusion rules, view gates (agreement/confidence/strength),
training seeds (at least 3) and budgets, per-experiment hyperparameters
(beta, epsilon, beta_prime, the wDPO stage toggles and pinned revision),
frozen validation decoding (also used for the study export), the test-once
policy, the study design, synthetic flip rates, and the statistical plan.
The backend TOMLs under `configs/gemma4/` carry runtime concerns only
(model pin, quantization, gradient checkpointing); an optional `[backends]`
section pins their hashes for provenance.

**Sweep axes.** A sweepable hyperparameter (`beta`, `epsilon`, `beta_prime`)
may be a list: `beta = [0.1, 0.3]` expands the experiment into one trained
variant per value inside one workspace. Every variant is validated
identically; selection picks one winner per experiment and track (recorded
with its hyperparameters in the selection report) and only winners reach the
lock, the test, and the study. Artifact identities are keyed per pipeline
stage on exactly the contract sections that stage reads — and per training
cell on its resolved variant — so extending a sweep axis recomputes only the
new cells and their downstream reports; corpus, evidence, candidate, and
annotation artifacts (the expensive live-class stages) are reused untouched.

## 3. Splits before anything else

```bash
uv run dpo corpus ingest --workspace "$STORE" --contract "$CONTRACT" \
  --input data/clips.jsonl | tee runs/01-ingest.json
uv run dpo corpus lock-splits --workspace "$STORE" --contract "$CONTRACT" \
  --artifact-id "$INGEST_ID" | tee runs/02-registry.json
```

Clip rows carry `clip_id`, `source_video_id`, `media_hash`, `start_ms`,
`end_ms`, `derivative_hashes` (muted-video and audio-only derivatives), an
optional `link_group` (near-duplicate/continuity assertions from an external
detector, with provenance), and an optional `asserted_role` that can only
confirm — never override — the seeded assignment. All clips of one source
video and all link-grouped clips land in one split; visual and audio tracks
share the clip's assignment; the PRD-shaped split manifest (with its sha256)
is embedded in the registry artifact and printed by the command.

## 4. Live boundaries fail closed

Live evidence extraction, real Gemma training, live scoring, and study
recruitment are typed external gates: they exit with code 3 and
`blocked_pending_external_operation`, never a partial local imitation.

```bash
uv run dpo evidence run --workspace "$STORE" --contract "$CONTRACT" \
  --track visual --invoke-external   # exit 3 until a pinned provider publishes artifacts
```

The real backend is Gemma 4 QLoRA (`configs/gemma4/12b.toml` for the visual
track, `configs/gemma4/e4b.toml` for the audio track — runtime shape only;
every training hyperparameter comes from the study contract; language-model
LoRA only, media towers frozen, `world_size=1`). The offline comparison uses
the deterministic tiny backend in `dpo.models.tiny` — the same trainers, the
same objectives, the same artifact discipline.

## 5. Test-once and the human study

Before any test payload is readable: validation completes, the selection
report is approved, and the lock manifest freezes checkpoints, processor,
preprocessing, generation configuration, metric versions, the study
interface, exclusion criteria, and the statistical plan.

```bash
uv run dpo test reserve --workspace "$STORE" --contract "$CONTRACT" \
  --artifact-id "$LOCK_MANIFEST_ID" --artifact-id "$TEST_SHARD_1" ...
```

One semantic hash (lock manifest + test split + generation + evaluation +
analysis) gets one reservation; only an identical resume is permitted.
Corrupted files, hardware failures, and deterministic crashes justify a
resume; a low score, an unexpected ranking, or a desire to change decoding do
not. The study export produces blinded incomplete-block tasks over all 36
model pairs with balanced exposures, randomized A/B position, clip-disjoint
participant blocks, and a blinding scan; model identity exists only in the
restricted randomization manifest.

## Repository structure (PRD section 22, adapted)

```text
src/dpo/
├── core/          # content-addressed artifact store, identity, atomic IO,
│                  # fenced access, GPU leases, text-safety screening
├── contracts/     # study_contract (vocabularies, matrix, validator),
│                  # visual_caption / audio_caption compliance screens
├── data/          # split manifest, D_sft / D_pair_strict / D_pair_all,
│                  # noise calibration, flip manifests, weighting, leakage audit
├── evidence/      # visual_evidence / audio_evidence schemas, claim ledger,
│                  # pinned providers
├── candidates/    # candidate_records + C0 policy, deterministic audits,
│                  # pair sampler, freeze
├── annotation/    # raw_annotations, aggregation, reliability + exclusions
├── models/        # shared completion logprob, modality-isolated batches,
│                  # visual_media / audio_media builders, tiny CPU backend,
│                  # gemma4/ (adapter, backend_config, tokenization safety)
├── objectives/    # base protocol + dpo, ipo, cdpo, rdpo, drdpo, wdpo, sft
├── trainers/      # one preference trainer for every preference arm,
│                  # SFT trainer, diagnostics
├── evaluation/    # compliance, preference accuracy, factuality, track metrics,
│                  # caption_generation, blinded study export
├── analysis/      # Bradley-Terry, clip-cluster bootstrap + BH, robustness
└── pipeline/      # stage registry (artifact types + contract slices),
                   # publishing (the one artifact publisher), corpus_stage,
                   # experiment resolution + sweep expansion, matrix runner,
                   # lock manifest + test-once resolver, offline canary
```

File naming follows one rule: every basename is globally unique and says what
the module contains. `uv run dpo stage list` prints the stage registry and the
artifact-typed lineage; the CLI validates its inputs against the same
registry, and artifact cache identities derive from its declared contract
slices, so documentation cannot drift from enforcement.

## Reproducibility and integrity

- Artifact identities are hashes of semantic manifests — never timestamps,
  paths, hosts, or PIDs; execution facts live in separate receipts. Each
  stage's identity covers exactly the contract sections it declares in the
  stage registry, so an unrelated contract tweak cannot invalidate it.
- One frozen raw preference dataset regenerates every trained experiment's
  view bit-identically; aggregation-rule changes create a new view version,
  never a new raw collection.
- A training artifact cannot have validation, test, or study exposure
  anywhere in its recursive ancestry (enforced at publish time).
- Golden tests snapshot one training step per objective
  (`tests/golden/golden_values.json`; regenerate deliberately with
  `make golden` and review the diff).
- wDPO is experimental: pinned to `arxiv_2603.07211v1`, with placeholder
  scalar maps documented in `dpo/objectives/wdpo.py` that must be
  re-verified against the pinned official code before any confirmatory run.

See [`docs/pipeline.md`](docs/pipeline.md) for the invariants and claim
limits.
