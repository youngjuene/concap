"""The resumable experiment-matrix runner behind one backend seam.

``LiveMatrixRunner`` owns everything that is backend-independent: the matrix
semantics (SEED never trained, SFT on the SFT view, preference arms against a
frozen reference, SFT_DPO warm-started from the SFT checkpoint), batching,
trainer construction, diagnostics, the checkpoint layout, and resumption. A
``TrainingBackend`` owns everything model-specific: what a seed policy is, how
to clone it trainable, how to persist and reload it, how to free its memory,
and what a media batch of one track looks like. The same runner therefore
drives the tiny CPU backend and the real Gemma QLoRA backend, and the offline
canary's ``OfflineMatrixRunner`` stays untouched as the frozen reference of
what the semantics must be.

Resumption is content-addressed, not timestamped: each cell directory holds
the saved adapter plus a ``cell.json`` carrying the published cell document
and the semantic hash of everything the cell trained from. A rerun recomputes
that hash and reuses the cell only when it matches, so a crash mid-matrix
never redoes completed cells and a changed view, hyperparameter, seed, or
training section retrains exactly the cells it affects.

References never occupy a second model in VRAM: reference log-probabilities
are precomputed once per cell and the reference adapter is released before the
policy trains.
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch

from dpo.contracts.study_contract import ContractError, StudyContract
from dpo.core.identity import canonical_bytes, semantic_hash
from dpo.data.derive_pairs import MetadataPair, StrictPair
from dpo.data.derive_sft import SftExample
from dpo.models.base import MediaBatch, ModelAdapter
from dpo.models.tiny import TinyAdapter, encode_text, synthetic_media
from dpo.pipeline.experiments import (
    ExperimentVariant,
    ResolvedExperiment,
    expand_experiment,
    resolve_experiment,
)

# The one orientation of the metadata-rich view for training. Importing the
# offline runner's implementation (rather than restating it) keeps the live and
# offline matrices from ever disagreeing about what pair_all trains on.
from dpo.pipeline.run_matrix import DEFAULT_MEDIA_DIM, CellResult, _oriented_from_metadata
from dpo.pipeline.stage_inputs import parse_cell_result
from dpo.trainers.callbacks import REQUIRED_PREFERENCE_KEYS, DiagnosticsLog
from dpo.trainers.preference_trainer import (
    PreferenceTrainer,
    TrainerConfig,
    build_pair_batches,
    precompute_reference_logps,
)
from dpo.trainers.sft_trainer import SftTrainer, build_sft_batches

CELL_FILE = "cell.json"
CELL_KEY = tuple[str, str, str, int]


@runtime_checkable
class TrainingBackend(Protocol):
    """The seam between matrix semantics and one concrete model backend."""

    def seed_adapter(self, track: str) -> ModelAdapter:
        """The frozen SEED policy of one track; shared, never trained."""

    def clone_trainable(self, track: str, source: ModelAdapter | None) -> ModelAdapter:
        """A trainable copy of ``source``, or of SEED when ``source`` is None."""

    def save(self, adapter: ModelAdapter, directory: Path) -> None:
        """Persist a trained policy so a later process can reload it."""

    def load(self, track: str, directory: Path) -> ModelAdapter:
        """Reload a policy saved by ``save``."""

    def release(self, adapter: ModelAdapter) -> None:
        """Free whatever device memory the adapter holds; idempotent."""

    def media_batch(self, track: str, clip_ids: Sequence[str]) -> MediaBatch:
        """The track's media features for these clips."""


@runtime_checkable
class CompletionTokenCounter(Protocol):
    """Optional backend hook: count completion tokens with the real tokenizer."""

    def count_completion_tokens(self, track: str, text: str) -> int: ...


def _cuda() -> bool:
    return bool(torch.cuda.is_available())


_UNSAFE_PATH = re.compile(r"[^A-Za-z0-9._=+-]+")


def _slug(value: str) -> str:
    """A filesystem-safe, collision-free rendering of a variant id."""
    safe = _UNSAFE_PATH.sub("-", value)
    if safe != value:
        safe = f"{safe}-{semantic_hash({'variant': value}).removeprefix('sha256:')[:8]}"
    return safe or "variant"


def cell_directory(root: str | Path, *, track: str, experiment_id: str, variant_id: str, seed: int) -> Path:
    """``<root>/<track>/<experiment>__<variant>__seed<N>`` — the one layout."""
    return Path(root) / track / f"{experiment_id}__{_slug(variant_id)}__seed{seed}"


def _write_document(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp"
    temporary.write_bytes(canonical_bytes(document) + b"\n")
    os.replace(temporary, path)


def read_cell_checkpoint(directory: Path, *, inputs_hash: str | None = None) -> CellResult | None:
    """The completed cell in ``directory``, or None when absent or stale."""
    path = Path(directory) / CELL_FILE
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(document, Mapping):
        return None
    if inputs_hash is not None and document.get("inputs_hash") != inputs_hash:
        return None
    cell = document.get("cell")
    if not isinstance(cell, Mapping):
        return None
    return parse_cell_result(cell)


def scan_checkpoints(root: str | Path) -> list[tuple[CellResult, Path]]:
    """Every completed cell under a checkpoint root, in deterministic order."""
    found: list[tuple[CellResult, Path]] = []
    for path in sorted(Path(root).glob(f"*/*/{CELL_FILE}")):
        cell = read_cell_checkpoint(path.parent)
        if cell is not None:
            found.append((cell, path.parent))
    return found


@dataclass(frozen=True)
class PolicyLocation:
    """Where one cell's policy lives: a saved checkpoint, or the shared SEED."""

    track: str
    directory: Path
    trained: bool


class CheckpointPolicies(Mapping[CELL_KEY, ModelAdapter]):
    """Lazily materialized policies, at most one checkpoint resident at a time.

    Selection scores eighteen cells; loading eighteen QLoRA policies at once
    would exhaust the device. Each lookup materializes its policy from the
    checkpoint directory and releases the previously materialized one — the
    backend's ``release`` is what decides how much that actually frees.
    """

    def __init__(self, backend: TrainingBackend) -> None:
        self._backend = backend
        self._entries: dict[CELL_KEY, PolicyLocation] = {}
        self._resident: tuple[CELL_KEY, ModelAdapter] | None = None

    def register(self, key: CELL_KEY, location: PolicyLocation) -> None:
        self._entries[key] = location

    def __getitem__(self, key: CELL_KEY) -> ModelAdapter:
        location = self._entries[key]
        if not location.trained:
            # SEED is shared and never released; it has no checkpoint of its own.
            return self._backend.seed_adapter(location.track)
        if self._resident is not None and self._resident[0] == key:
            return self._resident[1]
        if self._resident is not None:
            self._backend.release(self._resident[1])
            self._resident = None
        adapter = self._backend.load(location.track, location.directory)
        self._resident = (key, adapter)
        return adapter

    def __iter__(self) -> Iterator[CELL_KEY]:
        return iter(sorted(self._entries))

    def __len__(self) -> int:
        return len(self._entries)


def load_checkpoint_policies(
    backend: TrainingBackend, root: str | Path
) -> tuple[CheckpointPolicies, dict[tuple[str, str, str], CellResult]]:
    """Rebuild the lazy policy map (and the cells) from a checkpoint root."""
    policies = CheckpointPolicies(backend)
    cells: dict[tuple[str, str, str], CellResult] = {}
    for cell, directory in scan_checkpoints(root):
        policies.register(
            (cell.experiment_id, cell.variant_id, cell.track, cell.seed),
            PolicyLocation(track=cell.track, directory=directory, trained=cell.trained),
        )
        cells[(cell.experiment_id, cell.variant_id, cell.track)] = cell
    return policies, cells


@dataclass
class TinyBackend:
    """The deterministic CPU backend: the offline model, checkpointed to disk."""

    contract: StudyContract
    media_dim: int = DEFAULT_MEDIA_DIM
    _seed: dict[str, TinyAdapter] = field(default_factory=dict)

    def seed_adapter(self, track: str) -> ModelAdapter:
        adapter = self._seed.get(track)
        if adapter is None:
            seed_model = self.contract.raw["models"]["seed"]
            implementation = str(seed_model["implementation"])
            if implementation != "dpo.models.tiny.TinyAdapter":
                raise ContractError(
                    f"the tiny backend only serves the tiny seed model; models.seed names {implementation!r}"
                )
            precision = str(self.contract.training["precision"])
            if precision != "fp32":
                raise ContractError(
                    f"the tiny backend is fp32-only; training.precision={precision!r} applies"
                    " to the live backend"
                )
            adapter = TinyAdapter(
                track=track,
                prompt=self.contract.tracks[track].prompt,
                media_dim=self.media_dim,
                seed=int(str(seed_model["init_seed"])),
            )
            for parameter in adapter.model.parameters():
                parameter.requires_grad_(False)
            self._seed[track] = adapter
        return adapter

    def clone_trainable(self, track: str, source: ModelAdapter | None) -> ModelAdapter:
        base = self.seed_adapter(track) if source is None else source
        if not isinstance(base, TinyAdapter):
            raise ContractError("the tiny backend can only clone a tiny adapter")
        return base.clone_trainable()

    def save(self, adapter: ModelAdapter, directory: Path) -> None:
        if not isinstance(adapter, TinyAdapter):
            raise ContractError("the tiny backend can only save a tiny adapter")
        adapter.save(directory)

    def load(self, track: str, directory: Path) -> ModelAdapter:
        adapter = TinyAdapter.load(directory)
        if adapter.track != track:
            raise ContractError(f"checkpoint {str(directory)!r} holds a {adapter.track} adapter")
        return adapter

    def release(self, adapter: ModelAdapter) -> None:
        del adapter

    def media_batch(self, track: str, clip_ids: Sequence[str]) -> MediaBatch:
        return synthetic_media(track, clip_ids, media_dim=self.media_dim)


@dataclass
class LiveMatrixRunner:
    """One resumable matrix cell at a time, over any ``TrainingBackend``."""

    contract: StudyContract
    backend: TrainingBackend
    checkpoint_dir: Path
    strict_pairs: Mapping[str, tuple[StrictPair, ...]]
    metadata_pairs: Mapping[str, tuple[MetadataPair, ...]]
    sft_rows: Mapping[str, tuple[SftExample, ...]]
    policies: CheckpointPolicies = field(init=False)
    # (experiment, variant, track, seed) -> "trained" | "resumed"; the first
    # outcome wins, so a warm-start dependency cannot double-count its cell.
    outcomes: dict[CELL_KEY, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.policies = CheckpointPolicies(self.backend)

    # -- backend-facing helpers ------------------------------------------------

    def seed_adapter(self, track: str) -> ModelAdapter:
        return self.backend.seed_adapter(track)

    @property
    def seed_adapters(self) -> dict[str, ModelAdapter]:
        return {track: self.backend.seed_adapter(track) for track in sorted(self.contract.tracks)}

    def cell_directory(self, experiment_id: str, variant_id: str, track: str, seed: int) -> Path:
        return cell_directory(
            self.checkpoint_dir,
            track=track,
            experiment_id=experiment_id,
            variant_id=variant_id,
            seed=seed,
        )

    @property
    def trained_count(self) -> int:
        return sum(1 for outcome in self.outcomes.values() if outcome == "trained")

    @property
    def resumed_count(self) -> int:
        return sum(1 for outcome in self.outcomes.values() if outcome == "resumed")

    def was_resumed(self, experiment_id: str, variant_id: str, track: str, seed: int) -> bool:
        return self.outcomes.get((experiment_id, variant_id, track, seed)) == "resumed"

    # -- shared training mechanics --------------------------------------------

    def _count_tokens(self, track: str, text: str) -> int:
        if isinstance(self.backend, CompletionTokenCounter):
            return self.backend.count_completion_tokens(track, text)
        # Tiny tokenization is bytes + EOS, exactly as the offline runner counts.
        return len(encode_text(text)) + 1

    def _check_completion_budget(self, track: str, texts: Sequence[str]) -> None:
        budget = int(str(self.contract.training["max_completion_tokens"]))
        for text in texts:
            tokens = self._count_tokens(track, text)
            if tokens > budget:
                raise ContractError(
                    f"completion of {tokens} tokens exceeds training.max_completion_tokens={budget}"
                )

    def _trainer_config(self, *, seed: int, batches: int, preference: bool) -> TrainerConfig:
        training = self.contract.training
        epochs = float(str(training["epochs"]))
        rate_key = "learning_rate" if preference else "sft_learning_rate"
        return TrainerConfig(
            learning_rate=float(str(training[rate_key])),
            total_steps=max(1, math.ceil(epochs * batches)),
            max_grad_norm=float(str(training["max_grad_norm"])),
            seed=seed,
            gradient_accumulation_steps=int(str(training["gradient_accumulation_steps"])),
        )

    def _pairs_for_view(self, view: str, track: str) -> list[StrictPair]:
        if view == "pair_strict":
            return list(self.strict_pairs.get(track, ()))
        if view == "pair_all":
            return _oriented_from_metadata(list(self.metadata_pairs.get(track, ())))
        raise ContractError(f"unknown training view {view!r}")

    def _shuffled(self, rows: Sequence[StrictPair], seed: int) -> list[StrictPair]:
        shuffled = list(rows)
        random.Random(seed).shuffle(shuffled)
        return shuffled

    def _batch_size(self) -> int:
        return int(str(self.contract.training["batch_size"]))

    # -- identity of a cell's training inputs ----------------------------------

    def _row_fingerprint(self, resolved: ResolvedExperiment, track: str) -> dict[str, object]:
        rows: dict[str, object] = {}
        needs_sft = resolved.spec.objective == "sft" or resolved.spec.policy_init == "SFT"
        if needs_sft:
            rows["sft"] = [
                {
                    "example_id": row.example_id,
                    "candidate_id": row.candidate_id,
                    "completion": row.completion,
                    "weight": row.consensus_weight,
                }
                for row in self.sft_rows.get(track, ())
            ]
        if resolved.training_view in {"pair_strict", "pair_all"}:
            rows["pairs"] = [
                {
                    "pair_id": row.pair_id,
                    "chosen_id": row.chosen_id,
                    "rejected_id": row.rejected_id,
                    "chosen_text": row.chosen_text,
                    "rejected_text": row.rejected_text,
                    "weight": row.weight,
                }
                for row in self._pairs_for_view(str(resolved.training_view), track)
            ]
        return rows

    def inputs_hash(self, resolved: ResolvedExperiment, *, track: str, seed: int) -> str:
        """Everything this cell trains from, as one semantic hash.

        It is deliberately a superset of the published cell's cache identity
        (``training_stage.training_cell_slice``) plus the training rows: if the
        artifact identity could change while this hash did not, a rerun would
        publish a new artifact from a stale checkpoint.
        """
        return semantic_hash(
            {
                "schema": "dpo.cell-inputs/v1",
                "experiment_id": resolved.experiment_id,
                "variant_id": resolved.variant_id,
                "hyperparameters": dict(resolved.hyperparameters),
                "training_view": resolved.training_view,
                "seed": seed,
                "track": track,
                "track_contract": dict(self.contract.raw["tracks"][track]),
                "training": dict(self.contract.raw["training"]),
                "models_seed": dict(self.contract.raw["models"]["seed"]),
                "rows": self._row_fingerprint(resolved, track),
            }
        )

    # -- cells -----------------------------------------------------------------

    def _diagnostics(self, summary: Mapping[str, object]) -> dict[str, object]:
        diagnostics = dict(summary)
        if _cuda():
            # Only a cell that actually allocated device memory reports it, so a
            # CPU-backend cell has the same diagnostics on every machine.
            peak = int(torch.cuda.max_memory_allocated())
            if peak:
                diagnostics["peak_vram_bytes"] = peak
        return diagnostics

    def _finish(self, cell: CellResult, directory: Path, inputs_hash: str, *, resumed: bool) -> CellResult:
        key = (cell.experiment_id, cell.variant_id, cell.track, cell.seed)
        # First outcome wins, so the summary answers "did THIS run have to train
        # the cell?". A warm start that trains SFT keeps it counted as trained
        # even though the matrix's own SFT cell later reuses that checkpoint;
        # "resumed" therefore means reused from an earlier session.
        self.outcomes.setdefault(key, "resumed" if resumed else "trained")
        self.policies.register(
            key, PolicyLocation(track=cell.track, directory=directory, trained=cell.trained)
        )
        if not resumed:
            _write_document(directory / CELL_FILE, {"cell": cell.document(), "inputs_hash": inputs_hash})
        return cell

    def _release(self, *adapters: ModelAdapter | None) -> None:
        for adapter in adapters:
            if adapter is not None:
                self.backend.release(adapter)
        gc.collect()
        if _cuda():
            torch.cuda.empty_cache()

    def _sft_checkpoint(self, track: str, seed: int) -> tuple[CellResult, Path]:
        """The SFT cell of this (track, seed), training or resuming it as needed."""
        variants = expand_experiment(self.contract, "SFT")
        variant = variants[0]
        cell = self.run_cell("SFT", track=track, seed=seed, variant=variant)
        return cell, self.cell_directory("SFT", variant.variant_id, track, seed)

    def run_cell(
        self,
        experiment_id: str,
        *,
        track: str,
        seed: int,
        variant: ExperimentVariant | None = None,
    ) -> CellResult:
        resolved = resolve_experiment(self.contract, experiment_id, track=track, seed=seed, variant=variant)
        directory = self.cell_directory(experiment_id, resolved.variant_id, track, seed)
        inputs_hash = self.inputs_hash(resolved, track=track, seed=seed)
        cached = read_cell_checkpoint(directory, inputs_hash=inputs_hash)
        if cached is not None:
            return self._finish(cached, directory, inputs_hash, resumed=True)
        if _cuda():
            torch.cuda.reset_peak_memory_stats()
        if resolved.spec.objective is None:
            cell = self._run_seed_cell(resolved, track=track, seed=seed)
        elif resolved.spec.objective == "sft":
            cell = self._run_sft_cell(resolved, track=track, seed=seed, directory=directory)
        else:
            cell = self._run_preference_cell(resolved, track=track, seed=seed, directory=directory)
        return self._finish(cell, directory, inputs_hash, resumed=False)

    def _run_seed_cell(self, resolved: ResolvedExperiment, *, track: str, seed: int) -> CellResult:
        adapter = self.backend.seed_adapter(track)
        return CellResult(
            experiment_id=resolved.experiment_id,
            variant_id=resolved.variant_id,
            track=track,
            seed=seed,
            trained=False,
            objective=None,
            training_view=None,
            hyperparameters=dict(resolved.hyperparameters),
            steps=0,
            final_loss=None,
            checkpoint_signature=adapter.state_signature(),
            reference_signature=None,
            compute={"sft_steps": 0, "preference_steps": 0},
            diagnostics_summary=self._diagnostics({"steps": 0}),
        )

    def _run_sft_cell(
        self, resolved: ResolvedExperiment, *, track: str, seed: int, directory: Path
    ) -> CellResult:
        rows = self.sft_rows.get(track, ())
        if not rows:
            raise ContractError(f"track {track!r} has no SFT rows; cannot train SFT")
        self._check_completion_budget(track, [row.completion for row in rows])
        batches = build_sft_batches(list(rows), self.backend.media_batch, batch_size=self._batch_size())
        config = self._trainer_config(seed=seed, batches=len(batches), preference=False)
        torch.manual_seed(seed)
        policy = self.backend.clone_trainable(track, None)
        SftTrainer(policy, config).train(batches)
        signature = policy.state_signature()
        self.backend.save(policy, directory)
        cell = CellResult(
            experiment_id=resolved.experiment_id,
            variant_id=resolved.variant_id,
            track=track,
            seed=seed,
            trained=True,
            objective="sft",
            training_view="sft",
            hyperparameters=dict(resolved.hyperparameters),
            steps=config.total_steps,
            final_loss=None,
            checkpoint_signature=signature,
            reference_signature=None,
            compute={"sft_steps": config.total_steps, "preference_steps": 0},
            diagnostics_summary=self._diagnostics({"steps": config.total_steps}),
        )
        self._release(policy)
        return cell

    def _run_preference_cell(
        self, resolved: ResolvedExperiment, *, track: str, seed: int, directory: Path
    ) -> CellResult:
        view = str(resolved.training_view)
        rows = self._shuffled(self._pairs_for_view(view, track), seed)
        if not rows:
            raise ContractError(f"track {track!r} has no pairs in view {view!r}")
        self._check_completion_budget(
            track, [text for row in rows for text in (row.chosen_text, row.rejected_text)]
        )
        batches = build_pair_batches(rows, self.backend.media_batch, batch_size=self._batch_size())
        sft_steps = 0
        warm_start = resolved.spec.policy_init == "SFT"
        if warm_start:
            sft_cell, sft_directory = self._sft_checkpoint(track, seed)
            sft_steps = sft_cell.steps
            reference = self.backend.load(track, sft_directory)
        else:
            reference = self.backend.seed_adapter(track)
        # The reference contributes precomputed log-probabilities only: it is
        # never a second live model while the policy trains.
        reference_logps = precompute_reference_logps(reference, batches)
        reference_signature = reference.state_signature()
        torch.manual_seed(seed)
        policy = self.backend.clone_trainable(track, reference if warm_start else None)
        if warm_start:
            self._release(reference)
        config = self._trainer_config(seed=seed, batches=len(batches), preference=True)
        log = DiagnosticsLog(required=REQUIRED_PREFERENCE_KEYS)
        trainer = PreferenceTrainer(
            policy,
            resolved.build_preference_objective(),
            config,
            reference_logps=reference_logps,
            log=log,
        )
        summary = trainer.train(batches)
        last = summary.get("last")
        final_loss = float(str(last["loss"])) if isinstance(last, dict) else None
        signature = policy.state_signature()
        self.backend.save(policy, directory)
        cell = CellResult(
            experiment_id=resolved.experiment_id,
            variant_id=resolved.variant_id,
            track=track,
            seed=seed,
            trained=True,
            objective=resolved.spec.objective,
            training_view=view,
            hyperparameters=dict(resolved.hyperparameters),
            steps=config.total_steps,
            final_loss=final_loss,
            checkpoint_signature=signature,
            reference_signature=reference_signature,
            compute={"sft_steps": sft_steps, "preference_steps": config.total_steps},
            diagnostics_summary=self._diagnostics(summary),
        )
        self._release(policy)
        return cell
