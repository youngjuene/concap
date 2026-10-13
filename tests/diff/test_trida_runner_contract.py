"""AC-02 and AC-05 for Trida: the call contract is refused before the model is touched."""

from __future__ import annotations

from typing import Any

import pytest
from fake_trida_model import IM_END, MASK, PAD, FakeTokenizer, FakeTrida

from diff import trida_runner
from diff.caption_sources import SourceSet
from diff.job_plan import build_phase1_plan, freeze_plan, trida_call_settings
from diff.model_profile import QWEN_7B, TRIDA_7B
from diff.model_runtime import LoadedModel, RefusedCall, chat_token_ids


def valid_call(**changes: Any) -> dict[str, Any]:
    call = trida_call_settings(96, 0.9)
    call.update(changes)
    return call


def test_the_protocol_call_is_accepted_as_is() -> None:
    call = valid_call()
    assert trida_runner.checked_call(call) == call
    assert set(call) == set(trida_runner.REQUIRED_KEYWORDS)


@pytest.mark.parametrize("keyword", trida_runner.REQUIRED_KEYWORDS)
def test_every_required_keyword_is_required(keyword: str) -> None:
    call = valid_call()
    del call[keyword]
    with pytest.raises(RefusedCall, match=f"missing required keyword '{keyword}'"):
        trida_runner.checked_call(call)


@pytest.mark.parametrize(
    "keyword",
    ["top_p", "tokenizer", "steps", "seed", "do_sample", "stopping_criteria", "return_dict_in_generate"],
)
def test_unknown_keywords_are_refused(keyword: str) -> None:
    with pytest.raises(RefusedCall, match="passes nothing else"):
        trida_runner.checked_call(valid_call(**{keyword: 1}))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"threshold": 0.8}, "threshold must be one of"),
        ({"threshold": True}, "threshold must be one of"),
        ({"temperature": 0.2}, "temperature must be 0"),
        ({"temperature": False}, "temperature must be 0"),
        ({"block_size": 16}, "block_size must be 32"),
        ({"small_block_size": 4}, "small_block_size must be 8"),
        ({"use_block_cache": True}, "use_block_cache must be False"),
        ({"mask_id": 151666}, "mask_id must be 128012"),
        ({"stop_token": 128009}, "stop_token must be 128001"),
        ({"pad_token_id": 0}, "pad_token_id must be 128004"),
        ({"max_new_tokens": 80}, "whole number of 32-token blocks"),
        ({"max_new_tokens": 16}, "whole number of 32-token blocks"),
        ({"max_new_tokens": 96.0}, "whole number of 32-token blocks"),
    ],
)
def test_refused_values(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(RefusedCall, match=message):
        trida_runner.checked_call(valid_call(**changes))


@pytest.mark.parametrize(
    ("prompt", "accepted"), [(31, True), (32, False), (33, True), (63, True), (64, True), (65, True)]
)
def test_block_boundary_prompts_ac05(prompt: int, accepted: bool) -> None:
    # A prompt of exactly one block is not prefilled, so the first block would not see it.
    assert (not trida_runner.context_problems([1000] * prompt, 96)) is accepted


@pytest.mark.parametrize(("prompt", "accepted"), [(3999, True), (4000, False)])
def test_the_context_edge(prompt: int, accepted: bool) -> None:
    assert prompt + 96 + 1 in (4096, 4097)
    assert (not trida_runner.context_problems([1000] * prompt, 96)) is accepted


@pytest.mark.parametrize("token", [MASK, PAD])
def test_a_mask_or_pad_token_in_the_prompt_is_refused(token: int) -> None:
    assert trida_runner.context_problems([1000, token, 1001], 96)


def loaded(model: Any, tokenizer: FakeTokenizer) -> LoadedModel:
    return LoadedModel(model=model, tokenizer=tokenizer, device="cpu", profile=TRIDA_7B)


def frozen_job(sources: SourceSet, tokenizer: FakeTokenizer, condition: str = "trida_tau_090") -> Any:
    plan = build_phase1_plan(sources.sources, ("original", condition), "development")
    frozen = freeze_plan(
        plan,
        lambda job: chat_token_ids(tokenizer, job.prompt),
        {TRIDA_7B.profile_id: {"fake": True}, QWEN_7B.profile_id: {"fake": True}},
    )
    return frozen.jobs[0]


def test_run_job_records_the_output_forwards_and_rng(development_sources: SourceSet) -> None:
    tokenizer = FakeTokenizer()
    model = FakeTrida("버스가 지나간다.", tokenizer=tokenizer)
    result = trida_runner.run_job(loaded(model, tokenizer), frozen_job(development_sources, tokenizer))
    assert result["status"] == "completed", result["error"]
    assert result["output"]["text"] == "버스가 지나간다."
    assert result["output"]["terminator"] == "im_end" and result["output"]["completion"] == "terminated"
    assert result["rng_unchanged"] is True and result["caps"] == {"forward_calls": 3, "canvas": 96}
    called = model.calls[0]
    assert called["threshold"] == 0.9 and called["max_new_tokens"] == 96 and called["temperature"] == 0
    assert "top_p" not in called and "tokenizer" not in called


@pytest.mark.parametrize("condition", ["trida_tau_070", "trida_tau_100"])
def test_each_tau_condition_reaches_the_call(development_sources: SourceSet, condition: str) -> None:
    tokenizer = FakeTokenizer()
    model = FakeTrida("새가 지저귄다.", tokenizer=tokenizer)
    trida_runner.run_job(loaded(model, tokenizer), frozen_job(development_sources, tokenizer, condition))
    assert model.calls[0]["threshold"] == {"trida_tau_070": 0.7, "trida_tau_100": 1.0}[condition]


@pytest.mark.parametrize(
    ("changes", "status", "completion"),
    [
        ({"leave_mask": True}, "leftover_mask", "terminated"),
        ({"use_rng": True}, "rng_consumed", "terminated"),
        ({"stop": False}, "completed", "incomplete"),
    ],
)
def test_failures_are_recorded_never_repaired(
    development_sources: SourceSet, changes: dict[str, bool], status: str, completion: str
) -> None:
    tokenizer = FakeTokenizer()
    model = FakeTrida(
        "가" * 200 if changes.get("stop") is False else "바람 소리", tokenizer=tokenizer, **changes
    )
    result = trida_runner.run_job(loaded(model, tokenizer), frozen_job(development_sources, tokenizer))
    assert result["status"] == status and result["output"]["completion"] == completion


def test_a_prompt_that_differs_from_the_frozen_plan_is_refused(development_sources: SourceSet) -> None:
    tokenizer = FakeTokenizer()
    job = frozen_job(development_sources, tokenizer)
    drifted = FakeTokenizer(prefix=(1001,))
    result = trida_runner.run_job(loaded(FakeTrida("x", tokenizer=drifted), drifted), job)
    assert result["status"] == "refused"


def test_warm_up_uses_no_randomness() -> None:
    tokenizer = FakeTokenizer()
    warm = trida_runner.warm_up(loaded(FakeTrida("네.", tokenizer=tokenizer), tokenizer))
    assert warm["rng_unchanged"] is True


def test_the_stop_and_pad_ids_are_trida_s() -> None:
    assert (IM_END, PAD, MASK) == (128001, 128004, 128012)
