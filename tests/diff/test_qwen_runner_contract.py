"""The Qwen baseline call: greedy, repetition penalty neutralised, nothing of Dream's passed."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from fake_dream_model import IM_END, FakeQwen, FakeTokenizer

from diff import qwen_runner
from diff.caption_sources import SourceSet
from diff.job_plan import build_phase1_plan, freeze_plan, qwen_call_settings
from diff.model_profile import QWEN_7B
from diff.model_runtime import LoadedModel, RefusedCall, chat_token_ids


def valid_call(**changes: Any) -> dict[str, Any]:
    return {"attention_mask": torch.ones(1, 3), **qwen_call_settings(1), **changes}


def test_settings_match_the_contract() -> None:
    assert qwen_call_settings(1) == {
        "max_new_tokens": 128,
        "do_sample": False,
        "repetition_penalty": 1.0,
        "eos_token_id": [151645, 151643],
    }
    assert qwen_call_settings(2)["max_new_tokens"] == 192


@pytest.mark.parametrize("keyword", qwen_runner.REQUIRED_KEYWORDS)
def test_required_keywords(keyword: str) -> None:
    call = valid_call()
    del call[keyword]
    with pytest.raises(RefusedCall, match="missing required keyword"):
        qwen_runner.checked_call(call)


@pytest.mark.parametrize(
    "keyword", ["temperature", "top_p", "top_k", "steps", "alg", "mask_token_id", "generation_config"]
)
def test_sampling_and_dream_keywords_are_refused(keyword: str) -> None:
    with pytest.raises(RefusedCall, match="refused keywords"):
        qwen_runner.checked_call(valid_call(**{keyword: 0}))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"do_sample": True}, "do_sample must be False"),
        ({"repetition_penalty": 1.05}, "repetition_penalty must be 1.0"),
        ({"eos_token_id": [151643]}, "eos_token_id must be"),
    ],
)
def test_refused_values(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(RefusedCall, match=message):
        qwen_runner.checked_call(valid_call(**changes))


def test_run_job_records_the_resolved_config(development_sources: SourceSet) -> None:
    tokenizer = FakeTokenizer()
    plan = build_phase1_plan(development_sources.sources, ("qwen",), "development")
    frozen = freeze_plan(plan, lambda job: chat_token_ids(tokenizer, job.prompt), {QWEN_7B.profile_id: {}})
    model = FakeQwen("버스가 지나간다.", tokenizer=tokenizer)
    loaded = LoadedModel(model=model, tokenizer=tokenizer, device="cpu", profile=QWEN_7B)
    result = qwen_runner.run_job(loaded, frozen.jobs[0])
    assert result["status"] == "completed"
    assert result["output"]["terminator"] == "im_end" and result["output"]["text"] == "버스가 지나간다."
    assert result["resolved_config"]["repetition_penalty"] == 1.0
    assert result["resolved_config"]["do_sample"] is False
    assert set(model.calls[0]) == {
        "attention_mask",
        "max_new_tokens",
        "do_sample",
        "repetition_penalty",
        "eos_token_id",
    }
    assert IM_END in model.calls[0]["eos_token_id"]
