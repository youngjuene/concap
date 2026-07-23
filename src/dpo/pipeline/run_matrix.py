"""The nine-condition matrix runner over the tiny offline backend.

Runs the full comparison on synthetic media with real optimizer steps: SEED
untouched, SFT chosen-only SFT from SEED, DPO/IPO/CDPO/RDPO/DRDPO/WDPO
preference objectives from SEED against a frozen SEED reference, SFT_DPO DPO
from SFT against a frozen SFT reference. The runner is store-free — the CLI
publishes each cell's result as an artifact; golden tests snapshot per-cell
losses and signatures.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from dpo.contracts.study_contract import ContractError, StudyContract
from dpo.data.derive_pairs import MetadataPair, StrictPair
from dpo.data.derive_sft import SftExample
from dpo.models.base import MediaBatch
from dpo.models.tiny import TinyAdapter, encode_text, synthetic_media
from dpo.pipeline.experiments import (
    ExperimentVariant,
    ResolvedExperiment,
    expand_experiment,
    resolve_experiment,
)
from dpo.trainers.callbacks import REQUIRED_PREFERENCE_KEYS, DiagnosticsLog
from dpo.trainers.preference_trainer import (
    PreferenceTrainer,
    TrainerConfig,
    build_pair_batches,
)
from dpo.trainers.sft_trainer import SftTrainer, build_sft_batches

DEFAULT_MEDIA_DIM = 16


@dataclass(frozen=True)
class CellResult:
    experiment_id: str
    variant_id: str
    track: str
    seed: int
    trained: bool
    objective: str | None
    training_view: str | None
    hyperparameters: dict[str, object]
    steps: int
    final_loss: float | None
    checkpoint_signature: str
    reference_signature: str | None
    compute: dict[str, int]
    diagnostics_summary: dict[str, object]

    def document(self) -> dict[str, object]:
        return {
            "schema": "dpo.matrix-cell/v1",
            "experiment_id": self.experiment_id,
            "variant_id": self.variant_id,
            "track": self.track,
            "seed": self.seed,
            "trained": self.trained,
            "objective": self.objective,
            "training_view": self.training_view,
            "hyperparameters": dict(self.hyperparameters),
            "steps": self.steps,
            "final_loss": self.final_loss,
            "checkpoint_signature": self.checkpoint_signature,
            "reference_signature": self.reference_signature,
            "compute": dict(self.compute),
            "diagnostics_summary": dict(self.diagnostics_summary),
        }


def _oriented_from_metadata(rows: Sequence[MetadataPair]) -> list[StrictPair]:
    """Orient the metadata-rich view for training: majority direction, no forced ties."""
    oriented = []
    for row in rows:
        if row.preference_probability == 0.5:
            continue
        chosen_first = row.preference_probability > 0.5
        oriented.append(
            StrictPair(
                pair_id=row.pair_id,
                clip_id=row.clip_id,
                track=row.track,
                chosen_id=row.candidate_a if chosen_first else row.candidate_b,
                rejected_id=row.candidate_b if chosen_first else row.candidate_a,
                chosen_text=row.text_a if chosen_first else row.text_b,
                rejected_text=row.text_b if chosen_first else row.text_a,
                category=row.category,
                agreement=row.agreement,
                mean_strength=0.0,
                difficulty=row.difficulty,
                weight=row.weight,
            )
        )
    return oriented


@dataclass
class OfflineMatrixRunner:
    contract: StudyContract
    strict_pairs: dict[str, tuple[StrictPair, ...]]
    metadata_pairs: dict[str, tuple[MetadataPair, ...]]
    sft_rows: dict[str, tuple[SftExample, ...]]
    media_dim: int = DEFAULT_MEDIA_DIM
    _seed: dict[str, TinyAdapter] = field(default_factory=dict)
    _sft: dict[tuple[str, int], tuple[TinyAdapter, int]] = field(default_factory=dict)
    # Trained policies per (experiment, variant, track, seed), retained for
    # validation scoring, test generation, and the study export.
    policies: dict[tuple[str, str, str, int], TinyAdapter] = field(default_factory=dict)

    def seed_adapter(self, track: str) -> TinyAdapter:
        adapter = self._seed.get(track)
        if adapter is None:
            seed_model = self.contract.raw["models"]["seed"]
            implementation = str(seed_model["implementation"])
            if implementation != "dpo.models.tiny.TinyAdapter":
                raise ContractError(
                    f"the offline runner only constructs the tiny backend; models.seed names"
                    f" {implementation!r} — live backends go through the train gate"
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

    def _media_provider(self, track: str, clip_ids: Sequence[str]) -> MediaBatch:
        return synthetic_media(track, clip_ids, media_dim=self.media_dim)

    def _check_completion_budget(self, texts: Sequence[str]) -> None:
        # Tiny tokenization is bytes + EOS; the live backend must re-enforce this
        # budget with its own tokenizer when training wiring lands.
        budget = int(str(self.contract.training["max_completion_tokens"]))
        for text in texts:
            tokens = len(encode_text(text)) + 1
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

    def _shuffled(self, rows: Sequence[StrictPair], seed: int) -> list[StrictPair]:
        shuffled = list(rows)
        random.Random(seed).shuffle(shuffled)
        return shuffled

    def _train_sft(self, track: str, seed: int) -> tuple[TinyAdapter, int]:
        cached = self._sft.get((track, seed))
        if cached is not None:
            return cached
        rows = self.sft_rows.get(track, ())
        if not rows:
            raise ContractError(f"track {track!r} has no SFT rows; cannot train SFT")
        self._check_completion_budget([row.completion for row in rows])
        policy = self.seed_adapter(track).clone_trainable()
        batch_size = int(str(self.contract.training["batch_size"]))
        batches = build_sft_batches(list(rows), self._media_provider, batch_size=batch_size)
        config = self._trainer_config(seed=seed, batches=len(batches), preference=False)
        trainer = SftTrainer(policy, config)
        trainer.train(batches)
        self._sft[(track, seed)] = (policy, config.total_steps)
        return policy, config.total_steps

    def _pairs_for_view(self, view: str, track: str) -> list[StrictPair]:
        if view == "pair_strict":
            return list(self.strict_pairs.get(track, ()))
        if view == "pair_all":
            return _oriented_from_metadata(list(self.metadata_pairs.get(track, ())))
        raise ContractError(f"unknown training view {view!r}")

    def run_cell(
        self,
        experiment_id: str,
        *,
        track: str,
        seed: int,
        variant: ExperimentVariant | None = None,
    ) -> CellResult:
        resolved: ResolvedExperiment = resolve_experiment(
            self.contract, experiment_id, track=track, seed=seed, variant=variant
        )
        if resolved.spec.objective is None:
            adapter = self.seed_adapter(track)
            self.policies[(experiment_id, resolved.variant_id, track, seed)] = adapter
            return CellResult(
                experiment_id=experiment_id,
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
                diagnostics_summary={"steps": 0},
            )
        if resolved.spec.objective == "sft":
            policy, steps = self._train_sft(track, seed)
            self.policies[(experiment_id, resolved.variant_id, track, seed)] = policy
            return CellResult(
                experiment_id=experiment_id,
                variant_id=resolved.variant_id,
                track=track,
                seed=seed,
                trained=True,
                objective="sft",
                training_view="sft",
                hyperparameters=dict(resolved.hyperparameters),
                steps=steps,
                final_loss=None,
                checkpoint_signature=policy.state_signature(),
                reference_signature=None,
                compute={"sft_steps": steps, "preference_steps": 0},
                diagnostics_summary={"steps": steps},
            )
        sft_steps = 0
        if resolved.spec.policy_init == "SFT":
            base, sft_steps = self._train_sft(track, seed)
        else:
            base = self.seed_adapter(track)
        policy = base.clone_trainable()
        reference = base.clone_frozen()
        assert resolved.training_view is not None
        rows = self._shuffled(self._pairs_for_view(resolved.training_view, track), seed)
        if not rows:
            raise ContractError(f"track {track!r} has no pairs in view {resolved.training_view!r}")
        self._check_completion_budget([text for row in rows for text in (row.chosen_text, row.rejected_text)])
        batch_size = int(str(self.contract.training["batch_size"]))
        batches = build_pair_batches(rows, self._media_provider, batch_size=batch_size)
        config = self._trainer_config(seed=seed, batches=len(batches), preference=True)
        log = DiagnosticsLog(required=REQUIRED_PREFERENCE_KEYS)
        trainer = PreferenceTrainer(
            policy,
            resolved.build_preference_objective(),
            config,
            reference=reference,
            log=log,
        )
        summary = trainer.train(batches)
        last = summary.get("last")
        final_loss = float(str(last["loss"])) if isinstance(last, dict) else None
        self.policies[(experiment_id, resolved.variant_id, track, seed)] = policy
        return CellResult(
            experiment_id=experiment_id,
            variant_id=resolved.variant_id,
            track=track,
            seed=seed,
            trained=True,
            objective=resolved.spec.objective,
            training_view=resolved.training_view,
            hyperparameters=dict(resolved.hyperparameters),
            steps=config.total_steps,
            final_loss=final_loss,
            checkpoint_signature=policy.state_signature(),
            reference_signature=reference.state_signature(),
            compute={"sft_steps": sft_steps, "preference_steps": config.total_steps},
            diagnostics_summary=summary,
        )

    def run(
        self,
        *,
        experiment_ids: Sequence[str],
        tracks: Sequence[str],
        seeds: Sequence[int],
    ) -> list[CellResult]:
        results = []
        for experiment_id in experiment_ids:
            for variant in expand_experiment(self.contract, experiment_id):
                for track in tracks:
                    for seed in seeds:
                        results.append(self.run_cell(experiment_id, track=track, seed=seed, variant=variant))
        return results
