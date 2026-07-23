"""The shared completion-only sequence log-probability implementation.

log pi(y|x) = sum_t log pi(y_t | x, y_<t), over completion tokens only:
prompt tokens and padding are masked out, chosen and rejected use identical
media and prompt preprocessing, and the default is the total sequence
log-probability — length normalization is a separately flagged ablation, never
a silent default. Log-softmax runs in float32 with explicit NaN/overflow
guards.
"""

from __future__ import annotations

import torch
from torch import Tensor


class LogProbError(ValueError):
    """Raised when a batch cannot yield a well-defined completion log-probability."""


def completion_logprobs(
    logits: Tensor,
    input_ids: Tensor,
    completion_mask: Tensor,
    *,
    length_normalized: bool = False,
) -> Tensor:
    """Sum log-probabilities of completion tokens for each row.

    ``logits`` are next-token logits aligned with ``input_ids`` ([B, L, V] and
    [B, L]); ``completion_mask`` marks completion token POSITIONS in the
    sequence (1 where ``input_ids`` holds a supervised completion token).
    Because logits at position t predict the token at t+1, position 0 can
    never be supervised.
    """
    if logits.dim() != 3 or input_ids.dim() != 2 or completion_mask.dim() != 2:
        raise LogProbError("expected logits [B, L, V], input_ids [B, L], completion_mask [B, L]")
    if logits.shape[:2] != input_ids.shape or input_ids.shape != completion_mask.shape:
        raise LogProbError("logits/input_ids/completion_mask shapes disagree")
    if input_ids.shape[1] < 2:
        raise LogProbError("sequences must contain at least one predicted token")
    if bool(completion_mask[:, 0].any()):
        raise LogProbError("position 0 cannot be a completion token; nothing predicts it")
    mask = completion_mask[:, 1:].to(dtype=torch.float32)
    per_row_tokens = mask.sum(dim=1)
    if bool((per_row_tokens < 1).any()):
        raise LogProbError("every row must supervise at least one completion token")
    log_probs = torch.log_softmax(logits[:, :-1].to(torch.float32), dim=-1)
    if not bool(torch.isfinite(log_probs).all()):
        raise LogProbError("log-softmax produced non-finite values; refusing to train on them")
    targets = input_ids[:, 1:].unsqueeze(-1)
    token_logps = log_probs.gather(-1, targets).squeeze(-1)
    sums = (token_logps * mask).sum(dim=1)
    if not bool(torch.isfinite(sums).all()):
        raise LogProbError("sequence log-probabilities are non-finite")
    if length_normalized:
        return sums / per_row_tokens
    return sums


def masks_from_lengths(
    prompt_lengths: Tensor, total_lengths: Tensor, max_length: int
) -> tuple[Tensor, Tensor]:
    """Build attention and completion masks from per-row prompt/total lengths."""
    if prompt_lengths.shape != total_lengths.shape or prompt_lengths.dim() != 1:
        raise LogProbError("length tensors must be matching 1-D [batch]")
    if bool((prompt_lengths < 1).any()):
        raise LogProbError("every row needs at least one prompt token")
    if bool((total_lengths <= prompt_lengths).any()):
        raise LogProbError("every row needs at least one completion token after the prompt")
    if bool((total_lengths > max_length).any()):
        raise LogProbError("a row exceeds max_length; truncation would drop supervised tokens")
    positions = torch.arange(max_length, device=prompt_lengths.device).unsqueeze(0)
    attention = (positions < total_lengths.unsqueeze(1)).to(torch.long)
    completion = ((positions >= prompt_lengths.unsqueeze(1)) & (positions < total_lengths.unsqueeze(1))).to(
        torch.long
    )
    return attention, completion
