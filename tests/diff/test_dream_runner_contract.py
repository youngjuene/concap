"""AC-02 and the runner's caps: Dream's call contract is refused before the model is touched."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from fake_dream_model import ENDOFTEXT, IM_END, MASK, FakeDream, FakeTokenizer

from diff import dream_runner
from diff.caption_sources import SourceSet
from diff.job_plan import build_phase1_plan, dream_call_settings, freeze_plan
from diff.model_profile import DREAM_7B, QWEN_7B
from diff.model_runtime import LoadedModel, RefusedCall, chat_token_ids


def valid_call(**changes: Any) -> dict[str, Any]:
    call: dict[str, Any] = {"attention_mask": torch.ones(1, 4), **dream_call_settings(1, 1)}
    call.update(changes)
    return call


def test_the_protocol_call_is_accepted_as_is() -> None:
    call = valid_call()
    checked = dream_runner.checked_call(call)
    assert set(checked) == set(dream_runner.REQUIRED_KEYWORDS)
    assert checked["attention_mask"] is call["attention_mask"]
    assert {k: v for k, v in checked.items() if k != "attention_mask"} == dream_call_settings(1, 1)


@pytest.mark.parametrize("keyword", dream_runner.REQUIRED_KEYWORDS)
def test_every_required_keyword_is_required(keyword: str) -> None:
    call = valid_call()
    del call[keyword]
    with pytest.raises(RefusedCall, match=f"missing required keyword '{keyword}'"):
        dream_runner.checked_call(call)


@pytest.mark.parametrize(
    "keyword",
    ["threshold", "block_size", "seed", "do_sample", "generation_config", "tokenizer", "max_length"],
)
def test_unknown_keywords_are_refused_because_update_would_drop_them(keyword: str) -> None:
    with pytest.raises(RefusedCall, match="drop them silently"):
        dream_runner.checked_call(valid_call(**{keyword: 1}))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"alg": "origin"}, "torch.rand"),
        ({"alg": "random"}, "alg must be one of"),
        ({"temperature": 0.2}, "temperature must be 0"),
        ({"temperature": False}, "temperature must be 0"),
        ({"alg_temp": None}, "alg_temp must be 0"),
        ({"alg_temp": 0.1}, "alg_temp must be 0"),
        ({"top_p": 0.95}, "top_p must be None"),
        ({"top_p": 1.0}, "top_p must be None"),
        ({"top_k": 50}, "top_k must be None"),
        ({"eps": 1e-2}, "eps must be"),
        ({"mask_token_id": 151643}, "mask_token_id must be"),
        ({"output_history": False}, "output_history must be True"),
        ({"return_dict_in_generate": False}, "return_dict_in_generate must be True"),
        ({"steps": 256}, "must not exceed the canvas"),
        ({"steps": 48}, "must divide it"),
        ({"steps": 0}, "steps must be a positive integer"),
        ({"max_new_tokens": 0}, "canvas"),
        ({"generation_tokens_hook_func": "not callable"}, "must be callable"),
    ],
)
def test_refused_values(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(RefusedCall, match=message):
        dream_runner.checked_call(valid_call(**changes))


def test_read_only_hooks_are_allowed_keywords() -> None:
    call = valid_call(generation_tokens_hook_func=lambda step, x, logits: x)
    assert "generation_tokens_hook_func" in dream_runner.checked_call(call)


@pytest.mark.parametrize(("prompt", "accepted"), [(1919, True), (1920, True), (1921, False)])
def test_context_edges_2047_2048_accepted_2049_refused(prompt: int, accepted: bool) -> None:
    problems = dream_runner.context_problems([1000] * prompt, 128)
    assert (not problems) is accepted
    assert prompt + 128 in (2047, 2048, 2049)


def test_a_mask_token_in_the_prompt_is_refused() -> None:
    assert any("mask token" in p for p in dream_runner.context_problems([1000, MASK, 1001], 128))


def test_very_short_prompts_are_accepted() -> None:
    assert dream_runner.context_problems([1000], 128) == []


def loaded(model: torch.nn.Module, tokenizer: FakeTokenizer) -> LoadedModel:
    return LoadedModel(model=model, tokenizer=tokenizer, device="cpu", profile=DREAM_7B)


def frozen_job(sources: SourceSet, tokenizer: FakeTokenizer, condition: str = "dream_k1") -> Any:
    plan = build_phase1_plan(sources.sources, ("original", condition), "development")
    frozen = freeze_plan(
        plan,
        lambda job: chat_token_ids(tokenizer, job.prompt),
        {DREAM_7B.profile_id: {"fake": True}, QWEN_7B.profile_id: {"fake": True}},
    )
    return frozen.jobs[0]


def test_run_job_records_caps_counts_and_the_terminator(development_sources: SourceSet) -> None:
    tokenizer = FakeTokenizer()
    model = FakeDream("버스가 지나간다.", tokenizer=tokenizer)
    result = dream_runner.run_job(loaded(model, tokenizer), frozen_job(development_sources, tokenizer))
    assert result["status"] == "completed", result["error"]
    assert result["output"]["text"] == "버스가 지나간다."
    assert result["output"]["terminator"] == "endoftext"
    assert result["output"]["completion"] == "terminated"
    assert result["output"]["postprocess"]["tail_token_ids"] == [ENDOFTEXT] * (
        80 - len("버스가 지나간다.") - 1
    )
    assert result["caps"] == {"forward_calls": 80, "steps": 80, "canvas": 80}
    assert result["rng_unchanged"] is True
    counts = result["trace"]["summary"]["realized_per_step"]
    assert len(counts) == 80 and sum(counts) == 80
    assert counts[0] == 0 and counts[-1] == 2 and set(counts[1:-1]) == {1}
    called = model.calls[0]
    assert called["steps"] == 80 and called["max_new_tokens"] == 80 and called["top_k"] is None


def test_k4_commits_about_four_per_step(development_sources: SourceSet) -> None:
    tokenizer = FakeTokenizer()
    model = FakeDream("새가 지저귄다.", tokenizer=tokenizer)
    result = dream_runner.run_job(
        loaded(model, tokenizer), frozen_job(development_sources, tokenizer, "dream_k4")
    )
    assert result["status"] == "completed"
    counts = result["trace"]["summary"]["realized_per_step"]
    assert len(counts) == 20 and sum(counts) == 80 and max(counts) <= 8


def test_im_end_is_a_terminator_too(development_sources: SourceSet) -> None:
    tokenizer = FakeTokenizer()
    model = FakeDream("문이 닫힌다.", tokenizer=tokenizer, terminator=IM_END)
    result = dream_runner.run_job(loaded(model, tokenizer), frozen_job(development_sources, tokenizer))
    assert result["output"]["terminator"] == "im_end"


@pytest.mark.parametrize(
    ("changes", "status"),
    [
        ({"leave_mask": True}, "leftover_mask"),
        ({"extra_forward": True}, "count_mismatch"),
        ({"use_rng": True}, "rng_consumed"),
    ],
)
def test_cap_violations_are_technical_failures(
    development_sources: SourceSet, changes: dict[str, bool], status: str
) -> None:
    tokenizer = FakeTokenizer()
    model = FakeDream("바람 소리", tokenizer=tokenizer, **changes)
    result = dream_runner.run_job(loaded(model, tokenizer), frozen_job(development_sources, tokenizer))
    assert result["status"] == status


def test_a_prompt_that_differs_from_the_frozen_plan_is_refused(development_sources: SourceSet) -> None:
    tokenizer = FakeTokenizer()
    job = frozen_job(development_sources, tokenizer)
    drifted = FakeTokenizer(prefix=(1001,))
    result = dream_runner.run_job(loaded(FakeDream("x", tokenizer=drifted), drifted), job)
    assert result["status"] == "refused"


def test_realized_counts_reads_positions_leaving_the_mask() -> None:
    initial = [5, 6, MASK, MASK, MASK]
    history = [[5, 6, MASK, MASK, MASK], [5, 6, 7, MASK, MASK], [5, 6, 7, 8, 9]]
    assert dream_runner.realized_counts(initial, history, 2) == [0, 1, 2]
