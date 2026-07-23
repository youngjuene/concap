"""Fail-closed pinned boundary for external evidence providers.

Every provider call is frozen as two content-addressed artifacts: a raw bundle
binding the exact request, adapter/model/prompt/schema/lock pins, invocation
policy, response bytes, and response hash; and a parsed child that must equal
parsing that exact frozen response. Authority comes only from the code-owned
artifact type — no manifest attribute can grant a capability. Requests are
track-isolated in both directions: an audio request may carry no visual key,
hash, or vocabulary, and a visual request may carry no audio key, hash, or
vocabulary.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from dpo.core.artifacts import ArtifactManifest, ArtifactStore, ParentEdge, RequestSpec
from dpo.core.identity import canonical_bytes, semantic_hash, sha256_bytes
from dpo.core.textsafety import UntrustedTextError, validate_untrusted_value
from dpo.evidence.audio_evidence import AudioEvidence, parse_audio_evidence
from dpo.evidence.records import EvidenceError
from dpo.evidence.visual_evidence import VisualEvidence, parse_visual_evidence

ALLOWED_CAPABILITIES = frozenset({"visual_evidence", "audio_evidence"})
CAPABILITY_TRACKS = {"visual_evidence": "visual", "audio_evidence": "audio"}
TRACK_CAPABILITIES = {track: capability for capability, track in CAPABILITY_TRACKS.items()}

FORBIDDEN_IN_AUDIO_REQUESTS = re.compile(
    r"visual|image|frame|video|vision|picture|screen|watch", re.IGNORECASE
)
FORBIDDEN_IN_VISUAL_REQUESTS = re.compile(r"\baudio\b|sound|hear|speech|music|acoustic|listen", re.IGNORECASE)
HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
REVISION_RE = re.compile(r"(?:[0-9a-f]{40}|sha256:[0-9a-f]{64})\Z")
CREDENTIAL_RE = re.compile(r"[A-Z][A-Z0-9_]*\Z")
RAW_PROVIDER_BUNDLE_SCHEMA = "dpo.provider-raw-bundle/v1"
PROVIDER_DECODING_FIELDS = frozenset({"temperature", "top_p", "max_tokens"})
PROVIDER_RETRY_FIELDS = frozenset({"max_attempts", "initial_backoff_ms", "multiplier"})
NO_RETRY_POLICY: dict[str, object] = {
    "max_attempts": 1,
    "initial_backoff_ms": 0,
    "multiplier": 1.0,
}

PROVIDER_RAW_ARTIFACT_TYPES: dict[str, str] = {
    capability: f"dpo.provider-{capability.replace('_', '-')}-raw/v1"
    for capability in sorted(ALLOWED_CAPABILITIES)
}
PROVIDER_PARSED_ARTIFACT_TYPES: dict[str, str] = {
    capability: f"dpo.provider-{capability.replace('_', '-')}-parsed/v1"
    for capability in sorted(ALLOWED_CAPABILITIES)
}
PROVIDER_ARTIFACT_CAPABILITIES = {
    **{artifact_type: capability for capability, artifact_type in PROVIDER_RAW_ARTIFACT_TYPES.items()},
    **{artifact_type: capability for capability, artifact_type in PROVIDER_PARSED_ARTIFACT_TYPES.items()},
}

CAPABILITY_RESPONSE_SCHEMAS: dict[str, dict[str, object]] = {
    "visual_evidence": {
        "schema": "dpo.visual-evidence-response/v1",
        "fields": ["language", "clip_id", "frame_timestamps_ms", "scene_boundaries_ms", "items"],
        "language": "en",
    },
    "audio_evidence": {
        "schema": "dpo.audio-evidence-response/v1",
        "fields": [
            "language",
            "clip_id",
            "speech_present",
            "music_present",
            "low_activity_spans_ms",
            "items",
        ],
        "language": "en",
    },
}
CAPABILITY_REQUEST_SCHEMAS: dict[str, str] = {
    "visual_evidence": "dpo.visual-evidence-request/v1",
    "audio_evidence": "dpo.audio-evidence-request/v1",
}


def capability_schema_hash(capability: str) -> str:
    try:
        schema = CAPABILITY_RESPONSE_SCHEMAS[capability]
    except KeyError as exc:
        raise EvidenceError(f"unknown provider capability {capability!r}") from exc
    return semantic_hash({"capability": capability, "response_schema": schema})


def _contains_forbidden_tokens(value: object, pattern: re.Pattern[str]) -> bool:
    if isinstance(value, Mapping):
        return any(
            pattern.search(str(key)) is not None or _contains_forbidden_tokens(item, pattern)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_forbidden_tokens(item, pattern) for item in value)
    return isinstance(value, str) and pattern.search(value) is not None


def _check_track_isolation(document: Mapping[str, object], *, track: str, context: str) -> None:
    if track == "audio" and _contains_forbidden_tokens(dict(document), FORBIDDEN_IN_AUDIO_REQUESTS):
        raise EvidenceError(f"{context}: audio-track document has visual leakage")
    if track == "visual":
        # The request schema string itself legitimately contains the word
        # "visual"; only the payload-bearing fields are scanned.
        body = {key: value for key, value in document.items() if key != "schema"}
        if _contains_forbidden_tokens(body, FORBIDDEN_IN_VISUAL_REQUESTS):
            raise EvidenceError(f"{context}: visual-track document has audio leakage")


def _validate_parsed_evidence(
    parsed: Mapping[str, object], *, capability: str, clip_id: str, provider: str
) -> None:
    record: VisualEvidence | AudioEvidence
    if capability == "visual_evidence":
        record = parse_visual_evidence(parsed, provider=provider)
    else:
        record = parse_audio_evidence(parsed, provider=provider)
    if record.clip_id != clip_id:
        raise EvidenceError("provider evidence clip does not match the request clip")
    _check_track_isolation(dict(parsed), track=CAPABILITY_TRACKS[capability], context="parsed evidence")
    try:
        validate_untrusted_value(dict(parsed), field=f"provider.{capability}")
    except UntrustedTextError as exc:
        raise EvidenceError("provider parsed evidence contains prompt-injection/control text") from exc


@dataclass(frozen=True)
class AdapterIdentity:
    capability: str
    implementation: str
    model: str
    revision: str
    prompt_hash: str
    schema_hash: str
    lock_hash: str
    credential_env: str

    def __post_init__(self) -> None:
        if self.capability not in ALLOWED_CAPABILITIES:
            raise EvidenceError(f"forbidden capability {self.capability!r} for a provider adapter")
        for name, value in {"implementation": self.implementation, "model": self.model}.items():
            if not value.strip() or "${" in value:
                raise EvidenceError(f"adapter {name} pin must be resolved")
        if not REVISION_RE.fullmatch(self.revision):
            raise EvidenceError("adapter revision pin must be immutable")
        for name, value in {
            "prompt_hash": self.prompt_hash,
            "schema_hash": self.schema_hash,
            "lock_hash": self.lock_hash,
        }.items():
            if not HASH_RE.fullmatch(value):
                raise EvidenceError(f"adapter {name} pin must be a sha256 hash")
        if self.schema_hash != capability_schema_hash(self.capability):
            raise EvidenceError("adapter schema pin does not match its code-owned capability schema")
        if not CREDENTIAL_RE.fullmatch(self.credential_env):
            raise EvidenceError("adapter credential environment name is invalid")

    def document(self) -> dict[str, str]:
        return {
            "capability": self.capability,
            "implementation": self.implementation,
            "model": self.model,
            "revision": self.revision,
            "prompt_hash": self.prompt_hash,
            "schema_hash": self.schema_hash,
            "lock_hash": self.lock_hash,
            "credential_env": self.credential_env,
        }


def _validate_decoding(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != PROVIDER_DECODING_FIELDS:
        raise EvidenceError("provider decoding policy must have its exact canonical fields")
    temperature = value["temperature"]
    top_p = value["top_p"]
    max_tokens = value["max_tokens"]
    if (
        type(temperature) not in {int, float}
        or not math.isfinite(float(temperature))
        or float(temperature) < 0
        or type(top_p) not in {int, float}
        or not math.isfinite(float(top_p))
        or not 0 < float(top_p) <= 1
        or type(max_tokens) is not int
        or int(max_tokens) < 1
    ):
        raise EvidenceError("provider decoding policy values are invalid")
    return dict(value)


def _validate_retry_policy(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != PROVIDER_RETRY_FIELDS:
        raise EvidenceError("provider retry policy must have its exact canonical fields")
    attempts = value["max_attempts"]
    backoff = value["initial_backoff_ms"]
    multiplier = value["multiplier"]
    if (
        type(attempts) is not int
        or int(attempts) < 1
        or type(backoff) is not int
        or int(backoff) < 0
        or type(multiplier) not in {int, float}
        or not math.isfinite(float(multiplier))
        or float(multiplier) < 1
    ):
        raise EvidenceError("provider retry policy values are invalid")
    return dict(value)


def _adapter_identity(value: object) -> AdapterIdentity:
    if not isinstance(value, Mapping) or set(value) != set(AdapterIdentity.__dataclass_fields__):
        raise EvidenceError("provider adapter identity is not exact")
    if not all(isinstance(item, str) for item in value.values()):
        raise EvidenceError("provider adapter identity fields must be strings")
    try:
        return AdapterIdentity(**{str(key): str(item) for key, item in value.items()})
    except (TypeError, EvidenceError) as exc:
        raise EvidenceError("provider adapter identity is not exact") from exc


def build_evidence_request(
    *, capability: str, clip_id: str, media_hash: str, prompt: str
) -> dict[str, object]:
    """Build the exact single-modality evidence request for one clip derivative."""
    if capability not in ALLOWED_CAPABILITIES:
        raise EvidenceError(f"unknown provider capability {capability!r}")
    if not clip_id.strip():
        raise EvidenceError("evidence request requires a clip id")
    if not HASH_RE.fullmatch(media_hash):
        raise EvidenceError("evidence request media pin must be a sha256 hash")
    if not prompt.strip():
        raise EvidenceError("evidence request instruction must be non-empty")
    track = CAPABILITY_TRACKS[capability]
    media_key = "visual" if track == "visual" else "audio"
    document: dict[str, object] = {
        "schema": CAPABILITY_REQUEST_SCHEMAS[capability],
        "clip_id": clip_id,
        media_key: {"content_hash": media_hash},
        "instruction": prompt,
    }
    _check_track_isolation(document, track=track, context="evidence request")
    return document


def _validate_request_document(
    request_document: object,
    *,
    capability: str,
    clip_id: object,
    prompt_hash: str,
) -> dict[str, object]:
    if not isinstance(request_document, Mapping):
        raise EvidenceError("provider raw artifact is missing its exact request")
    if not isinstance(clip_id, str) or not clip_id.strip():
        raise EvidenceError("provider request clip is invalid")
    document = dict(request_document)
    track = CAPABILITY_TRACKS[capability]
    media_key = "visual" if track == "visual" else "audio"
    expected_fields = {"schema", "clip_id", media_key, "instruction"}
    instruction = document.get("instruction")
    if (
        set(document) != expected_fields
        or document.get("schema") != CAPABILITY_REQUEST_SCHEMAS[capability]
        or document.get("clip_id") != clip_id
        or not isinstance(instruction, str)
        or sha256_bytes(instruction.encode()) != prompt_hash
    ):
        raise EvidenceError("provider request schema/prompt/clip does not match its declared pins")
    media = document.get(media_key)
    if (
        not isinstance(media, Mapping)
        or set(media) != {"content_hash"}
        or not HASH_RE.fullmatch(str(media.get("content_hash", "")))
    ):
        raise EvidenceError("provider request media pin is not exact")
    _check_track_isolation(document, track=track, context="provider request")
    return document


def _parse_raw_bundle(payload: bytes) -> dict[str, object]:
    try:
        document = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise EvidenceError("provider raw payload is not valid canonical JSON") from exc
    if (
        not isinstance(document, dict)
        or set(document)
        != {
            "schema",
            "request_id",
            "capability",
            "request",
            "adapter",
            "decoding",
            "retry_policy",
            "response",
            "response_hash",
        }
        or document.get("schema") != RAW_PROVIDER_BUNDLE_SCHEMA
        or canonical_bytes(document) != payload
    ):
        raise EvidenceError("provider raw payload does not match its strict canonical schema")
    return document


def validate_provider_artifact(
    manifest: ArtifactManifest,
    payload: bytes,
    *,
    expected_capability: str | None = None,
    raw_manifest: ArtifactManifest | None = None,
    raw_payload: bytes | None = None,
) -> str:
    """Validate provider authority from its code-owned type, semantic request, and bytes."""
    capability = PROVIDER_ARTIFACT_CAPABILITIES.get(manifest.artifact_type)
    if capability is None:
        raise EvidenceError(f"artifact type {manifest.artifact_type!r} is not a provider authority")
    if expected_capability is not None and capability != expected_capability:
        raise EvidenceError("provider artifact has the wrong code-owned capability type")
    track = CAPABILITY_TRACKS[capability]
    request = manifest.semantic.get("request")
    attributes = manifest.semantic.get("attributes")
    if not isinstance(request, Mapping) or not isinstance(attributes, Mapping):
        raise EvidenceError("provider artifact request/attributes are invalid")
    model_pins = request.get("model_pins")
    prompt_pins = request.get("prompt_pins")
    parameters = request.get("parameters")
    identity = _adapter_identity(model_pins)
    if not isinstance(prompt_pins, Mapping) or not isinstance(parameters, Mapping):
        raise EvidenceError("provider adapter/model/prompt/schema pins are incomplete")
    if (
        identity.capability != capability
        or request.get("lock_id") != identity.lock_hash
        or dict(prompt_pins) != {"prompt_hash": identity.prompt_hash}
    ):
        raise EvidenceError("provider capability is not bound to exact immutable request pins")

    common_fields = {"capability", "response_schema_hash", "clip_id", "track"}
    if manifest.artifact_type == PROVIDER_RAW_ARTIFACT_TYPES[capability]:
        if set(parameters) != common_fields | {"request", "decoding", "retry_policy"}:
            raise EvidenceError("provider raw semantic request has unexpected or missing fields")
        decoding = _validate_decoding(parameters.get("decoding"))
        retry_policy = _validate_retry_policy(parameters.get("retry_policy"))
        request_document = _validate_request_document(
            parameters.get("request"),
            capability=capability,
            clip_id=parameters.get("clip_id"),
            prompt_hash=identity.prompt_hash,
        )
        bundle = _parse_raw_bundle(payload)
        response = bundle.get("response")
        response_hash = sha256_bytes(response.encode()) if isinstance(response, str) else None
        if (
            parameters.get("capability") != capability
            or parameters.get("track") != track
            or parameters.get("response_schema_hash") != identity.schema_hash
            or dict(attributes) != {"track": track, "raw_response_hash": response_hash}
            or bundle.get("request_id") != manifest.request_id
            or bundle.get("capability") != capability
            or bundle.get("request") != request_document
            or bundle.get("adapter") != identity.document()
            or bundle.get("decoding") != decoding
            or bundle.get("retry_policy") != retry_policy
            or not isinstance(response, str)
            or bundle.get("response_hash") != response_hash
        ):
            raise EvidenceError("provider raw payload disagrees with its exact semantic request pins")
        try:
            parsed_response = json.loads(response)
        except json.JSONDecodeError as exc:
            raise EvidenceError("provider response cannot be parsed into the pinned schema") from exc
        if not isinstance(parsed_response, dict):
            raise EvidenceError("parsed provider response must be an object")
        _validate_parsed_evidence(
            parsed_response,
            capability=capability,
            clip_id=str(parameters.get("clip_id")),
            provider=identity.model,
        )
        return capability

    if set(parameters) != common_fields | {
        "parser_version",
        "parser_schema_hash",
        "raw_artifact_id",
        "raw_response_hash",
    }:
        raise EvidenceError("provider parsed semantic request has unexpected or missing fields")
    parser_version = parameters.get("parser_version")
    raw_artifact_id = parameters.get("raw_artifact_id")
    if (
        not isinstance(parser_version, str)
        or not parser_version.strip()
        or "${" in parser_version
        or parameters.get("parser_schema_hash") != identity.schema_hash
        or not isinstance(raw_artifact_id, str)
        or not HASH_RE.fullmatch(raw_artifact_id)
        or len(manifest.parents) != 1
        or manifest.parents[0] != ParentEdge(str(raw_artifact_id), "raw-provider-response")
        or raw_manifest is None
        or raw_payload is None
        or raw_manifest.artifact_id != raw_artifact_id
    ):
        raise EvidenceError("provider parsed artifact is not bound to one exact raw response/parser")
    raw_capability = validate_provider_artifact(raw_manifest, raw_payload, expected_capability=capability)
    if raw_capability != capability:
        raise EvidenceError("parsed provider capability differs from its raw parent")
    raw_request = raw_manifest.semantic["request"]
    raw_parameters = raw_request["parameters"]
    raw_attributes = raw_manifest.semantic["attributes"]
    raw_bundle = _parse_raw_bundle(raw_payload)
    try:
        parsed_payload = json.loads(payload)
        parsed_response = json.loads(str(raw_bundle["response"]))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise EvidenceError("parsed provider payload is not valid JSON") from exc
    if (
        not isinstance(parsed_payload, dict)
        or canonical_bytes(parsed_payload) != payload
        or parsed_payload != parsed_response
    ):
        raise EvidenceError("parsed provider payload does not equal its exact raw response")
    raw_response_hash = raw_attributes.get("raw_response_hash")
    if (
        {key: request.get(key) for key in ("contract_id", "model_pins", "prompt_pins", "seed", "lock_id")}
        != {
            key: raw_request.get(key)
            for key in ("contract_id", "model_pins", "prompt_pins", "seed", "lock_id")
        }
        or {key: parameters.get(key) for key in common_fields}
        != {key: raw_parameters.get(key) for key in common_fields}
        or parameters.get("raw_response_hash") != raw_response_hash
        or dict(attributes)
        != {
            "track": track,
            "parser_schema_hash": identity.schema_hash,
            "parser_version": parser_version,
            "raw_response_hash": raw_response_hash,
        }
    ):
        raise EvidenceError("parsed provider artifact pins differ from its exact raw parent")
    _validate_parsed_evidence(
        parsed_payload,
        capability=capability,
        clip_id=str(parameters.get("clip_id")),
        provider=identity.model,
    )
    return capability


def validate_provider_artifact_from_store(
    store: ArtifactStore,
    manifest: ArtifactManifest,
    *,
    expected_capability: str | None = None,
) -> str:
    """Load and validate a provider artifact plus its exact raw parent from ``store``."""
    payload = store.read_payload(manifest.artifact_id)
    if manifest.artifact_type in PROVIDER_PARSED_ARTIFACT_TYPES.values():
        if len(manifest.parents) != 1:
            raise EvidenceError("provider parsed artifact must have exactly one raw parent")
        raw_manifest = store.verify(manifest.parents[0].artifact_id)
        raw_payload = store.read_payload(raw_manifest.artifact_id)
        return validate_provider_artifact(
            manifest,
            payload,
            expected_capability=expected_capability,
            raw_manifest=raw_manifest,
            raw_payload=raw_payload,
        )
    return validate_provider_artifact(manifest, payload, expected_capability=expected_capability)


class ProviderAdapter(Protocol):
    identity: AdapterIdentity

    def invoke(self, request: Mapping[str, object]) -> str: ...


class FixtureAdapter:
    def __init__(self, identity: AdapterIdentity, response: str) -> None:
        self.identity = identity
        self.response = response
        self.calls = 0

    def invoke(self, request: Mapping[str, object]) -> str:
        del request
        self.calls += 1
        return self.response


@dataclass(frozen=True)
class EvidenceRun:
    raw_artifact_id: str
    parsed_artifact_id: str
    parsed: Mapping[str, object]


class PinnedProviderRunner:
    """Provider-neutral content-addressed raw/parsed execution boundary."""

    def __init__(self, store: ArtifactStore, *, contract_id: str) -> None:
        self.store = store
        self.contract_id = contract_id

    def run(
        self,
        adapter: ProviderAdapter,
        *,
        request_document: Mapping[str, object],
        clip_id: str,
        seed: int,
        parser_version: str,
        decoding: Mapping[str, object],
        retry_policy: Mapping[str, object],
        role_exposure: set[str] | None = None,
        parents: tuple[ParentEdge, ...] = (),
    ) -> EvidenceRun:
        capability = adapter.identity.capability
        track = CAPABILITY_TRACKS[capability]
        validated_request = _validate_request_document(
            request_document,
            capability=capability,
            clip_id=clip_id,
            prompt_hash=adapter.identity.prompt_hash,
        )
        decoding_document = _validate_decoding(decoding)
        retry_document = _validate_retry_policy(retry_policy)
        common_parameters = {
            "capability": capability,
            "response_schema_hash": adapter.identity.schema_hash,
            "clip_id": clip_id,
            "track": track,
        }
        raw_request = RequestSpec(
            artifact_type=PROVIDER_RAW_ARTIFACT_TYPES[capability],
            parents=parents,
            contract_id=self.contract_id,
            model_pins=adapter.identity.document(),
            prompt_pins={"prompt_hash": adapter.identity.prompt_hash},
            parameters={
                **common_parameters,
                "request": validated_request,
                "decoding": decoding_document,
                "retry_policy": retry_document,
            },
            seed=seed,
            lock_id=adapter.identity.lock_hash,
        )
        existing = self.store.find_by_request_id(raw_request.request_id)
        if existing is None:
            response = adapter.invoke(validated_request)
            if not isinstance(response, str):
                raise EvidenceError("provider response boundary requires an exact UTF-8 JSON string")
            response_hash = sha256_bytes(response.encode())
            raw_payload_bytes = canonical_bytes(
                {
                    "schema": RAW_PROVIDER_BUNDLE_SCHEMA,
                    "request_id": raw_request.request_id,
                    "capability": capability,
                    "request": validated_request,
                    "adapter": adapter.identity.document(),
                    "decoding": decoding_document,
                    "retry_policy": retry_document,
                    "response": response,
                    "response_hash": response_hash,
                }
            )
            raw = self.store.publish(
                raw_request,
                raw_payload_bytes,
                row_count=1,
                clips={clip_id},
                role_exposure=role_exposure,
                attributes={"track": track, "raw_response_hash": response_hash},
            )
        else:
            raw = self.store.verify(existing)
        raw_payload = self.store.read_payload(raw.artifact_id)
        validate_provider_artifact(raw, raw_payload, expected_capability=capability)
        try:
            raw_document = json.loads(raw_payload)
            response_value = raw_document["response"]
            if not isinstance(response_value, str):
                raise TypeError
            parsed = json.loads(response_value)
        except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError) as exc:
            raise EvidenceError("provider response cannot be parsed into the pinned schema") from exc
        if not isinstance(parsed, dict):
            raise EvidenceError("parsed provider response must be an object")
        _validate_parsed_evidence(
            parsed, capability=capability, clip_id=clip_id, provider=adapter.identity.model
        )
        parsed_request = RequestSpec(
            artifact_type=PROVIDER_PARSED_ARTIFACT_TYPES[capability],
            parents=(ParentEdge(raw.artifact_id, "raw-provider-response"),),
            contract_id=self.contract_id,
            model_pins=adapter.identity.document(),
            prompt_pins={"prompt_hash": adapter.identity.prompt_hash},
            parameters={
                **common_parameters,
                "parser_version": parser_version,
                "parser_schema_hash": adapter.identity.schema_hash,
                "raw_artifact_id": raw.artifact_id,
                "raw_response_hash": raw_document["response_hash"],
            },
            seed=seed,
            lock_id=adapter.identity.lock_hash,
        )
        parsed_artifact = self.store.publish(
            parsed_request,
            canonical_bytes(parsed),
            row_count=1,
            clips={clip_id},
            role_exposure=role_exposure,
            attributes={
                "track": track,
                "parser_schema_hash": adapter.identity.schema_hash,
                "parser_version": parser_version,
                "raw_response_hash": raw_document["response_hash"],
            },
        )
        validate_provider_artifact(
            parsed_artifact,
            self.store.read_payload(parsed_artifact.artifact_id),
            expected_capability=capability,
            raw_manifest=raw,
            raw_payload=raw_payload,
        )
        return EvidenceRun(raw.artifact_id, parsed_artifact.artifact_id, parsed)


class EvidenceRunner:
    """Track-aware wrapper that binds contract prompts to evidence adapters."""

    def __init__(
        self,
        store: ArtifactStore,
        *,
        contract_id: str,
        prompt_contracts: Mapping[str, str],
    ) -> None:
        if set(prompt_contracts) != set(TRACK_CAPABILITIES):
            raise EvidenceError("evidence runner requires exact prompt text for both tracks")
        self.store = store
        self.contract_id = contract_id
        self.prompt_contracts = dict(prompt_contracts)
        self.provider_runner = PinnedProviderRunner(store, contract_id=contract_id)

    def run(
        self,
        adapter: ProviderAdapter,
        *,
        clip_id: str,
        track: str,
        media_hash: str,
        seed: int,
        parser_version: str,
        decoding: Mapping[str, object],
        retry_policy: Mapping[str, object],
        role_exposure: set[str] | None = None,
    ) -> EvidenceRun:
        if track not in TRACK_CAPABILITIES:
            raise EvidenceError(f"unknown track {track!r}")
        if adapter.identity.capability != TRACK_CAPABILITIES[track]:
            raise EvidenceError("adapter capability does not match the requested track")
        instruction = self.prompt_contracts[track]
        if sha256_bytes(instruction.encode()) != adapter.identity.prompt_hash:
            raise EvidenceError("exact provider instruction does not match the adapter/contract pins")
        request_document = build_evidence_request(
            capability=adapter.identity.capability,
            clip_id=clip_id,
            media_hash=media_hash,
            prompt=instruction,
        )
        return self.provider_runner.run(
            adapter,
            request_document=request_document,
            clip_id=clip_id,
            seed=seed,
            parser_version=parser_version,
            decoding=decoding,
            retry_policy=retry_policy,
            role_exposure=role_exposure,
        )
