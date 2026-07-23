"""Screening for untrusted evidence and caption text.

Evidence captions, provider responses, and candidate captions are untrusted
model/provider output. These expressions are deliberately code-owned rather than
contract-configurable: a study contract must not be able to weaken the boundary
that keeps evidence from becoming a second instruction channel. The patterns
cover the control delimiters used by the chat templates we support and direct
instruction-override/exfiltration/deletion language.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping


class UntrustedTextError(ValueError):
    """Raised before untrusted text can reach a prompt, dataset, or artifact."""


_ROLE_OR_CONTROL_DELIMITER_RE = re.compile(
    r"(?:"
    r"<\|(?:system|assistant|user|developer|im_start|im_end|start_of_turn|end_of_turn)[^>]*\|>"
    r"|</?(?:start_of_turn|end_of_turn|start_of_image|end_of_image)>"
    r"|</?(?:system|assistant|developer|user)>"
    r"|\[/?INST\]|<<\s*/?SYS\s*>>|###\s*(?:system|assistant|developer|user)"
    r"|(?:^|[\r\n])\s*(?:system|assistant|developer|user|model)\s*:"
    r")",
    re.IGNORECASE,
)
_INJECTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        r"\b(?:ignore|disregard|forget|override|bypass|replace)\b.{0,48}"
        r"\b(?:previous|prior|above|system|developer|instructions?|prompts?|rules?)\b",
        r"\b(?:previous|prior|above|system|developer)\b.{0,48}"
        r"\b(?:instructions?|prompts?|rules?)\b.{0,32}"
        r"\b(?:ignore|disregard|forget|override|bypass|replace)\b",
        r"\bdo\s+not\s+(?:follow|obey|respect)\b.{0,32}"
        r"\b(?:previous|prior|above|system|developer|instructions?|prompts?|rules?)\b",
        r"\btreat\b.{0,48}\b(?:following|this)\b.{0,32}\b(?:new|replacement)\b"
        r".{0,24}\b(?:system|developer)\b.{0,24}\b(?:message|prompt|instruction)\b",
        r"\bfrom\s+now\s+on\b.{0,48}\b(?:answer|respond|output|write|say)\b",
        r"\byour\s+new\s+(?:task|instruction|role)\b.{0,48}"
        r"\b(?:answer|respond|output|write|say)\b",
        r"\b(?:answer|respond|output|write|say)\s+only\b",
        r"\b(?:reveal|show|print|expose|leak|output|repeat)\b.{0,48}"
        r"\b(?:system\s+prompt|developer\s+message|secrets?|api\s*keys?|passwords?|credentials?|tokens?)\b",
        r"\b(?:delete|erase|remove|wipe|destroy|drop)\b.{0,48}"
        r"\b(?:state|files?|databases?|artifacts?|records?|history|memory)\b",
    )
)


def validate_untrusted_text(text: str, *, field: str) -> str:
    """Reject text that attempts to become chat control or an instruction.

    Text is scanned after Unicode compatibility normalization, but retained
    verbatim apart from outer whitespace when safe. Suspicious text is rejected,
    never rewritten, because sanitizing a caption would create untracked
    scientific data and can leave obfuscated control text behind.
    """
    normalized = unicodedata.normalize("NFKC", text).strip()
    if not normalized:
        raise UntrustedTextError(f"{field}: text must be non-empty")
    if any(
        (unicodedata.category(character) == "Cc" and character not in "\t\n\r")
        or unicodedata.category(character) == "Cf"
        for character in normalized
    ):
        raise UntrustedTextError(f"{field}: text contains a control/format character")
    if _ROLE_OR_CONTROL_DELIMITER_RE.search(normalized):
        raise UntrustedTextError(f"{field}: text contains a role/control delimiter")
    if any(pattern.search(normalized) for pattern in _INJECTION_PATTERNS):
        raise UntrustedTextError(f"{field}: text contains prompt-injection/control text")
    return text.strip()


def validate_untrusted_value(value: object, *, field: str) -> None:
    """Recursively validate every string leaf in a parsed untrusted value."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str):
                validate_untrusted_text(key, field=f"{field}.<key>")
            validate_untrusted_value(item, field=f"{field}[value]")
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            validate_untrusted_value(item, field=f"{field}[{index}]")
    elif isinstance(value, str):
        validate_untrusted_text(value, field=field)
