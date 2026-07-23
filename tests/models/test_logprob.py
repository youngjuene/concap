"""Shared log-probability implementation tests (PRD sections 11.2 and 14.3)."""

from __future__ import annotations

import math

import pytest
import torch

from dpo.models.logprob import LogProbError, completion_logprobs, masks_from_lengths


def test_prompt_tokens_do_not_contribute_to_the_loss() -> None:
    # Peaked logits at prompt positions would shift the sum by many nats if
    # they were wrongly included.
    vocab = 4
    logits = torch.zeros(1, 4, vocab)
    logits[0, 0, 1] = 50.0  # predicts position 1 (prompt) with certainty
    input_ids = torch.tensor([[0, 1, 2, 3]])
    completion_mask = torch.tensor([[0, 0, 1, 1]])
    sums = completion_logprobs(logits, input_ids, completion_mask)
    assert float(sums[0]) == pytest.approx(2 * math.log(1.0 / vocab), rel=1e-6)


def test_uniform_logits_give_log_vocab_per_completion_token() -> None:
    logits = torch.zeros(2, 5, 8)
    input_ids = torch.zeros(2, 5, dtype=torch.long)
    completion_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
    sums = completion_logprobs(logits, input_ids, completion_mask)
    assert float(sums[0]) == pytest.approx(3 * math.log(1 / 8), rel=1e-6)
    assert float(sums[1]) == pytest.approx(4 * math.log(1 / 8), rel=1e-6)


def test_length_normalization_is_a_separate_flag_not_a_default() -> None:
    logits = torch.zeros(1, 5, 8)
    input_ids = torch.zeros(1, 5, dtype=torch.long)
    completion_mask = torch.tensor([[0, 0, 1, 1, 1]])
    total = completion_logprobs(logits, input_ids, completion_mask)
    normalized = completion_logprobs(logits, input_ids, completion_mask, length_normalized=True)
    assert float(total[0]) == pytest.approx(3 * float(normalized[0]), rel=1e-6)


def test_first_position_cannot_be_supervised() -> None:
    logits = torch.zeros(1, 3, 4)
    input_ids = torch.zeros(1, 3, dtype=torch.long)
    with pytest.raises(LogProbError, match="position 0"):
        completion_logprobs(logits, input_ids, torch.tensor([[1, 1, 0]]))


def test_rows_without_completion_tokens_are_rejected() -> None:
    logits = torch.zeros(1, 3, 4)
    input_ids = torch.zeros(1, 3, dtype=torch.long)
    with pytest.raises(LogProbError, match="at least one completion token"):
        completion_logprobs(logits, input_ids, torch.zeros(1, 3, dtype=torch.long))


def test_non_finite_logits_are_refused() -> None:
    logits = torch.full((1, 3, 4), float("nan"))
    input_ids = torch.zeros(1, 3, dtype=torch.long)
    with pytest.raises(LogProbError):
        completion_logprobs(logits, input_ids, torch.tensor([[0, 1, 1]]))


def test_masks_from_lengths_shape_and_truncation_guard() -> None:
    attention, completion = masks_from_lengths(torch.tensor([2, 3]), torch.tensor([4, 5]), max_length=5)
    assert attention.tolist() == [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]
    assert completion.tolist() == [[0, 0, 1, 1, 0], [0, 0, 0, 1, 1]]
    with pytest.raises(LogProbError, match="max_length"):
        masks_from_lengths(torch.tensor([2]), torch.tensor([9]), max_length=5)
