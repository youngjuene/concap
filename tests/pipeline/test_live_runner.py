"""The live runner: identical matrix semantics, checkpointed and resumable.

The load-bearing test is parity — the same cell, run through the live runner on
the tiny backend, must reproduce the offline runner's result field for field.
Everything else here is about the checkpoint contract: what makes a cell
reusable, what makes it stale, and what the policy map hands to selection.
"""

from __future__ import annotations

import copy
import hashlib
import json
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from dpo.contracts.study_contract import EXPERIMENT_IDS, ContractError, StudyContract, validate_contract
from dpo.models.base import MediaBatch
from dpo.pipeline.live_runner import (
    LiveMatrixRunner,
    TinyBackend,
    load_checkpoint_policies,
    scan_checkpoints,
)
from dpo.pipeline.run_matrix import CellResult
from tests.conftest import CANARY_CONTRACT, PreferenceWorld, build_offline_runner, derive_world_views


def _runner(
    world: PreferenceWorld,
    checkpoints: Path,
    *,
    contract: StudyContract | None = None,
    backend: object | None = None,
) -> LiveMatrixRunner:
    active = contract if contract is not None else world.contract
    derived = derive_world_views(world, active)
    return LiveMatrixRunner(
        contract=active,
        backend=backend if backend is not None else TinyBackend(contract=active),  # type: ignore[arg-type]
        checkpoint_dir=checkpoints,
        strict_pairs=derived.strict_pairs,
        metadata_pairs=derived.metadata_pairs,
        sft_rows=derived.sft_rows,
    )


def test_live_cells_match_the_offline_matrix(world: PreferenceWorld, tmp_path: Path) -> None:
    offline = build_offline_runner(world)
    live = _runner(world, tmp_path / "checkpoints")
    for experiment_id in EXPERIMENT_IDS:
        expected = offline.run_cell(experiment_id, track="visual", seed=1)
        actual = live.run_cell(experiment_id, track="visual", seed=1)
        assert actual.document() == expected.document(), experiment_id


def test_every_cell_writes_a_reusable_checkpoint(world: PreferenceWorld, tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    live = _runner(world, checkpoints)
    cell = live.run_cell("DPO", track="visual", seed=1)
    directory = live.cell_directory("DPO", cell.variant_id, "visual", 1)
    document = json.loads((directory / "cell.json").read_text(encoding="utf-8"))
    assert set(document) == {"cell", "inputs_hash"}
    assert document["cell"] == cell.document()
    assert str(document["inputs_hash"]).startswith("sha256:")
    assert (directory / "adapter_model.safetensors").is_file()
    assert live.trained_count == 1 and live.resumed_count == 0
    # SEED is a matrix cell without a checkpoint of its own: it is the seed model.
    seed_cell = live.run_cell("SEED", track="visual", seed=1)
    seed_directory = live.cell_directory("SEED", seed_cell.variant_id, "visual", 1)
    assert (seed_directory / "cell.json").is_file()
    assert not (seed_directory / "adapter_model.safetensors").exists()


def test_resumption_reuses_the_recorded_cell_without_training(world: PreferenceWorld, tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    first = _runner(world, checkpoints)
    cell = first.run_cell("IPO", track="visual", seed=1)
    directory = first.cell_directory("IPO", cell.variant_id, "visual", 1)
    # Marking the recorded document proves the rerun reads it instead of
    # recomputing an identical-looking result.
    document = json.loads((directory / "cell.json").read_text(encoding="utf-8"))
    document["cell"]["checkpoint_signature"] = "sha256:" + "ab" * 32
    (directory / "cell.json").write_text(json.dumps(document), encoding="utf-8")
    second = _runner(world, checkpoints)
    resumed = second.run_cell("IPO", track="visual", seed=1)
    assert resumed.checkpoint_signature == "sha256:" + "ab" * 32
    assert second.resumed_count == 1 and second.trained_count == 0
    assert second.was_resumed("IPO", cell.variant_id, "visual", 1)


def test_changed_training_rows_invalidate_the_checkpoint(world: PreferenceWorld, tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    first = _runner(world, checkpoints)
    cell = first.run_cell("DPO", track="visual", seed=1)
    mutated = _runner(world, checkpoints)
    rows = list(mutated.strict_pairs["visual"])
    rows[0] = replace(rows[0], chosen_text=rows[0].chosen_text + " and one more clause")
    mutated.strict_pairs = {"visual": tuple(rows)}
    again = mutated.run_cell("DPO", track="visual", seed=1)
    assert mutated.trained_count == 1 and mutated.resumed_count == 0
    assert again.checkpoint_signature != cell.checkpoint_signature
    # The stale record is overwritten, not left behind.
    directory = mutated.cell_directory("DPO", cell.variant_id, "visual", 1)
    document = json.loads((directory / "cell.json").read_text(encoding="utf-8"))
    assert document["cell"] == again.document()


def test_a_hyperparameter_change_invalidates_only_its_own_cell(
    world: PreferenceWorld, tmp_path: Path
) -> None:
    checkpoints = tmp_path / "checkpoints"
    with CANARY_CONTRACT.open("rb") as handle:
        document = tomllib.load(handle)
    _runner(world, checkpoints).run_cell("DPO", track="visual", seed=1)
    _runner(world, checkpoints).run_cell("IPO", track="visual", seed=1)
    mutated = copy.deepcopy(document)
    experiments = mutated["experiments"]
    assert isinstance(experiments, dict)
    experiments["DPO"] = {"objective": "dpo", "beta": 0.3}
    tweaked = _runner(world, checkpoints, contract=validate_contract(mutated))
    tweaked.run_cell("DPO", track="visual", seed=1)
    tweaked.run_cell("IPO", track="visual", seed=1)
    assert tweaked.was_resumed("IPO", "base", "visual", 1)
    assert not tweaked.was_resumed("DPO", "base", "visual", 1)


def test_the_inputs_hash_covers_the_published_cache_identity(world: PreferenceWorld, tmp_path: Path) -> None:
    """A change that re-keys the published artifact must re-key the checkpoint.

    Otherwise a rerun would publish a fresh matrix-cell artifact out of a stale
    checkpoint. The prompt lives in the track contract, which the cell's
    ``slice_override`` covers; the inputs hash must cover it too.
    """
    checkpoints = tmp_path / "checkpoints"
    _runner(world, checkpoints).run_cell("SFT", track="visual", seed=1)
    with CANARY_CONTRACT.open("rb") as handle:
        document = tomllib.load(handle)
    mutated = copy.deepcopy(document)
    tracks = mutated["tracks"]
    assert isinstance(tracks, dict)
    prompt = str(tracks["visual"]["prompt"]) + " Keep it factual."
    tracks["visual"]["prompt"] = prompt
    tracks["visual"]["prompt_hash"] = "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    reprompted = _runner(world, checkpoints, contract=validate_contract(mutated))
    reprompted.run_cell("SFT", track="visual", seed=1)
    assert reprompted.trained_count == 1 and reprompted.resumed_count == 0


def test_sft_dpo_warm_starts_from_the_sft_checkpoint(world: PreferenceWorld, tmp_path: Path) -> None:
    live = _runner(world, tmp_path / "checkpoints")
    sft_dpo = live.run_cell("SFT_DPO", track="visual", seed=1)
    sft = live.run_cell("SFT", track="visual", seed=1)
    assert sft_dpo.reference_signature == sft.checkpoint_signature
    assert sft_dpo.compute["sft_steps"] == sft.steps > 0
    # The warm start ran the SFT cell once; its own rerun resumes it.
    assert live.was_resumed("SFT", "base", "visual", 1) is False
    assert live.trained_count == 2


def test_policies_materialize_from_the_checkpoint_directory(world: PreferenceWorld, tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    live = _runner(world, checkpoints)
    cells = {
        experiment_id: live.run_cell(experiment_id, track="visual", seed=1)
        for experiment_id in ("SEED", "SFT", "DPO")
    }
    for experiment_id, cell in cells.items():
        adapter = live.policies[(experiment_id, cell.variant_id, "visual", 1)]
        assert adapter.state_signature() == cell.checkpoint_signature
    # A fresh process rebuilds the same map from disk alone.
    policies, recovered = load_checkpoint_policies(TinyBackend(contract=world.contract), checkpoints)
    assert len(policies) == 3
    for experiment_id, cell in cells.items():
        assert recovered[(experiment_id, cell.variant_id, "visual")].document() == cell.document()
        assert (
            policies[(experiment_id, cell.variant_id, "visual", 1)].state_signature()
            == cell.checkpoint_signature
        )


def test_scan_checkpoints_reports_every_completed_cell(world: PreferenceWorld, tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    live = _runner(world, checkpoints)
    live.run_cell("SFT", track="visual", seed=1)
    live.run_cell("WDPO", track="visual", seed=1)
    found = {cell.experiment_id for cell, _ in scan_checkpoints(checkpoints)}
    assert found == {"SFT", "WDPO"}
    assert all(isinstance(cell, CellResult) for cell, _ in scan_checkpoints(checkpoints))


class _CountingBackend(TinyBackend):
    """A tiny backend that reports the live backend's optional token count."""

    def count_completion_tokens(self, track: str, text: str) -> int:
        return 10_000

    def media_batch(self, track: str, clip_ids: list[str]) -> MediaBatch:  # pragma: no cover
        return super().media_batch(track, clip_ids)


def test_the_completion_budget_uses_the_backend_token_count(world: PreferenceWorld, tmp_path: Path) -> None:
    live = _runner(world, tmp_path / "checkpoints", backend=_CountingBackend(contract=world.contract))
    with pytest.raises(ContractError, match="max_completion_tokens"):
        live.run_cell("SFT", track="visual", seed=1)
