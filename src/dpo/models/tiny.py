"""A deterministic tiny caption model for golden tests and the offline canary.

Byte-level vocabulary, learned positions, media conditioning by mean-pooled
feature projection, and causal mixing via cumulative means — small enough for
CPU golden tests, real enough to exercise every contract: prompt masking,
shared media for chosen/rejected, reference freezing, checkpoint round-trips,
and deterministic greedy generation.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import Tensor, nn

from dpo.core.identity import semantic_hash
from dpo.models.base import CompletionBatch, MediaBatch, ModalityIsolationError
from dpo.models.logprob import completion_logprobs, masks_from_lengths

BOS = 256
EOS = 257
VOCAB_SIZE = 258
MAX_LENGTH = 512


def encode_text(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode_tokens(tokens: Sequence[int]) -> str:
    # Untrained tiny models emit arbitrary byte sequences; replacement decoding
    # keeps generation total so compliance metrics can report on the output
    # honestly instead of the pipeline crashing on invalid UTF-8.
    return bytes(token for token in tokens if 0 <= token <= 255).decode("utf-8", errors="replace")


class TinyCaptionModel(nn.Module):
    """Causal byte LM conditioned on pooled media features."""

    def __init__(self, *, media_dim: int, hidden: int = 48, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.media_dim = media_dim
        self.hidden = hidden
        self.token_embedding = nn.Embedding(VOCAB_SIZE, hidden)
        self.position_embedding = nn.Embedding(MAX_LENGTH, hidden)
        self.media_projection = nn.Linear(media_dim, hidden)
        self.mixer = nn.Linear(2 * hidden, hidden)
        self.head = nn.Linear(hidden, VOCAB_SIZE)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(torch.empty_like(parameter).normal_(0.0, 0.05, generator=generator))

    def forward(self, input_ids: Tensor, media_vector: Tensor) -> Tensor:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be [B, L]")
        batch, length = input_ids.shape
        if length > MAX_LENGTH:
            raise ValueError(f"sequence length {length} exceeds MAX_LENGTH={MAX_LENGTH}")
        positions = torch.arange(length, device=input_ids.device)
        hidden = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions).unsqueeze(0)
            + self.media_projection(media_vector).unsqueeze(1)
        )
        counts = torch.arange(1, length + 1, device=input_ids.device, dtype=hidden.dtype)
        causal_context = hidden.cumsum(dim=1) / counts.view(1, length, 1)
        mixed = torch.tanh(self.mixer(torch.cat([hidden, causal_context], dim=-1)))
        logits: Tensor = self.head(mixed)
        return logits


def _pool_media(media: MediaBatch) -> Tensor:
    tensors = []
    for key in sorted(media.features):
        tensor = media.features[key]
        flattened = tensor.reshape(tensor.shape[0], -1, tensor.shape[-1]).to(torch.float32)
        tensors.append(flattened.mean(dim=1))
    return torch.cat(tensors, dim=-1)


class TinyAdapter:
    """ModelAdapter over TinyCaptionModel for one track."""

    def __init__(self, *, track: str, prompt: str, media_dim: int, hidden: int = 48, seed: int = 0) -> None:
        self._track = track
        self.prompt = prompt
        self.model = TinyCaptionModel(media_dim=media_dim, hidden=hidden, seed=seed)

    @property
    def track(self) -> str:
        return self._track

    @contextmanager
    def module_mode(self, *, training: bool) -> Iterator[None]:
        """Honoured for protocol parity; the tiny model has no mode-dependent layers.

        No dropout and no gradient checkpointing, so train and eval compute the
        same thing. The flag is still applied so a fixture cannot pass here and
        then behave differently on the real backend.
        """
        previous = bool(self.model.training)
        self.model.train(training)
        try:
            yield
        finally:
            self.model.train(previous)

    def _require_track(self, media: MediaBatch) -> None:
        if media.track != self._track:
            raise ModalityIsolationError(f"{self._track} adapter received a {media.track} media batch")

    def _sequences(self, completions: CompletionBatch) -> tuple[Tensor, Tensor, Tensor]:
        prompt_tokens = [BOS, *encode_text(self.prompt)]
        rows = []
        prompt_lengths = []
        total_lengths = []
        for text in completions.texts:
            completion_tokens = [*encode_text(text), EOS]
            rows.append(prompt_tokens + completion_tokens)
            prompt_lengths.append(len(prompt_tokens))
            total_lengths.append(len(prompt_tokens) + len(completion_tokens))
        max_length = max(total_lengths)
        input_ids = torch.zeros((len(rows), max_length), dtype=torch.long)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        attention, completion = masks_from_lengths(
            torch.tensor(prompt_lengths), torch.tensor(total_lengths), max_length
        )
        del attention
        return input_ids, completion, torch.tensor(total_lengths)

    def completion_logps(self, media: MediaBatch, completions: CompletionBatch) -> Tensor:
        self._require_track(media)
        if media.batch_size != len(completions.texts):
            raise ModalityIsolationError("media and completion batch sizes disagree")
        input_ids, completion_mask, _ = self._sequences(completions)
        logits = self.model(input_ids, _pool_media(media))
        return completion_logprobs(logits, input_ids, completion_mask)

    def generate(
        self,
        media: MediaBatch,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int,
    ) -> list[str]:
        self._require_track(media)
        del top_p
        generator = torch.Generator().manual_seed(seed)
        media_vector = _pool_media(media)
        outputs = []
        prompt_tokens = [BOS, *encode_text(self.prompt)]
        with torch.no_grad():
            for index in range(media.batch_size):
                tokens = list(prompt_tokens)
                produced: list[int] = []
                for _ in range(max_new_tokens):
                    input_ids = torch.tensor([tokens], dtype=torch.long)
                    logits = self.model(input_ids, media_vector[index : index + 1])[0, -1]
                    if temperature <= 0:
                        next_token = int(torch.argmax(logits))
                    else:
                        probabilities = torch.softmax(logits / temperature, dim=-1)
                        next_token = int(torch.multinomial(probabilities, 1, generator=generator))
                    if next_token == EOS:
                        break
                    if next_token == BOS:
                        continue
                    produced.append(next_token)
                    tokens.append(next_token)
                    if len(tokens) >= MAX_LENGTH - 1:
                        break
                outputs.append(decode_tokens(produced))
        return outputs

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return iter(self.model.parameters())

    def state_signature(self) -> str:
        payload = {
            name: [float(value) for value in tensor.detach().reshape(-1)]
            for name, tensor in sorted(self.model.state_dict().items())
        }
        return semantic_hash(payload)

    def save(self, directory: str | Path) -> None:
        from safetensors.torch import save_file

        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        save_file(dict(self.model.state_dict()), str(destination / "adapter_model.safetensors"))
        (destination / "tiny_adapter.json").write_text(
            json.dumps(
                {
                    "schema": "dpo.tiny-adapter/v1",
                    "track": self._track,
                    "prompt": self.prompt,
                    "media_dim": self.model.media_dim,
                    "hidden": self.model.hidden,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> TinyAdapter:
        from safetensors.torch import load_file

        source = Path(directory)
        config = json.loads((source / "tiny_adapter.json").read_text(encoding="utf-8"))
        if config.get("schema") != "dpo.tiny-adapter/v1":
            raise ValueError("tiny adapter manifest schema is unknown")
        adapter = cls(
            track=str(config["track"]),
            prompt=str(config["prompt"]),
            media_dim=int(config["media_dim"]),
            hidden=int(config["hidden"]),
        )
        adapter.model.load_state_dict(load_file(str(source / "adapter_model.safetensors")))
        return adapter

    def clone_frozen(self) -> TinyAdapter:
        """An exact frozen copy for use as the preference reference."""
        clone = self.clone_trainable()
        for parameter in clone.model.parameters():
            parameter.requires_grad_(False)
        clone.model.eval()
        return clone

    def clone_trainable(self) -> TinyAdapter:
        """An exact trainable copy for use as a policy initialization."""
        clone = TinyAdapter(
            track=self._track,
            prompt=self.prompt,
            media_dim=self.model.media_dim,
            hidden=self.model.hidden,
        )
        clone.model.load_state_dict(
            {name: tensor.clone() for name, tensor in self.model.state_dict().items()}
        )
        for parameter in clone.model.parameters():
            parameter.requires_grad_(True)
        return clone


def synthetic_media(track: str, clip_ids: Sequence[str], *, media_dim: int) -> MediaBatch:
    """Deterministic per-clip synthetic media features for tests and the canary."""
    rows = []
    for clip_id in clip_ids:
        digest = semantic_hash({"track": track, "clip": clip_id}).removeprefix("sha256:")
        values = [
            (int(digest[index * 2 : index * 2 + 2], 16) / 255.0) * 2.0 - 1.0 for index in range(media_dim)
        ]
        rows.append(values)
    tensor = torch.tensor(rows, dtype=torch.float32)
    if track == "visual":
        frames = tensor.unsqueeze(1).repeat(1, 4, 1)
        from dpo.models.visual_media import build_visual_media

        return build_visual_media(list(clip_ids), frames)
    from dpo.models.audio_media import build_audio_media

    return build_audio_media(list(clip_ids), tensor.unsqueeze(1).repeat(1, 6, 1))
