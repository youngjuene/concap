"""Output checks: fail, flag and measure — never repair (plan §4.6).

A failed check excludes the output and records why. Nothing is retried for
quality, re-prompted, truncated or shortened. The only transformation between
the model's tokens and the checked text is the cut at the first terminator
(``<|endoftext|>`` or ``<|im_end|>``, whichever comes first, recorded with the
trimmed tail) and stripping surrounding whitespace; both strings are kept.

*Fail* rules are technical and format rules a person need not judge. *Flag*
rules are content judgements; they only route an output to double review.
*Measures* are the §5.3 text measures. Fidelity and length are outcomes of the
condition and are recorded as such, never as covariates.

Lexical matching is deliberately simple and documented, because its counts
feed G2: a closed-vocabulary quality counts only in one of its two vocabulary
forms as whole words; an evidence source or relation term counts when a word
sequence starts with it (so particles such as "버스가" still match). Everything
is versioned by ``CHECKS_VERSION``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from diff.caption_sources import QUALITIES, CaptionSource
from diff.study_protocol import GATES, MAX_CHARS, MAX_LINES

CHECKS_VERSION: Final = 2
_PUNCTUATION: Final = ".,!?;:…·'\"“”‘’()[]{}「」『』<>~-—"
_HANGUL_RANGES: Final = (
    (0xAC00, 0xD7A3),
    (0x1100, 0x11FF),
    (0x3130, 0x318F),
    (0xA960, 0xA97F),
    (0xD7B0, 0xD7FF),
)
_JONGSEONG_MIEUM: Final = 16


@dataclass(frozen=True)
class GeneratedOutput:
    """The checked text and everything the cut removed, kept for the ``postprocess`` event."""

    raw_text: str
    text: str
    terminator: str | None
    terminator_index: int | None
    tail_token_ids: tuple[int, ...]
    tail_text: str
    leftover_mask_count: int
    special_ids_in_text: tuple[int, ...]
    new_token_count: int

    @property
    def completion(self) -> str:
        return "terminated" if self.terminator is not None else "incomplete"

    def record(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "text": self.text,
            "completion": self.completion,
            "terminator": self.terminator,
            "terminator_index": self.terminator_index,
            "postprocess": {"tail_token_ids": list(self.tail_token_ids), "tail_text": self.tail_text},
            "leftover_mask_count": self.leftover_mask_count,
            "special_ids_in_text": list(self.special_ids_in_text),
            "new_token_count": self.new_token_count,
        }


def output_from_record(record: Mapping[str, Any]) -> GeneratedOutput:
    return GeneratedOutput(
        raw_text=str(record["raw_text"]),
        text=str(record["text"]),
        terminator=record["terminator"],
        terminator_index=record["terminator_index"],
        tail_token_ids=tuple(int(t) for t in record["postprocess"]["tail_token_ids"]),
        tail_text=str(record["postprocess"]["tail_text"]),
        leftover_mask_count=int(record["leftover_mask_count"]),
        special_ids_in_text=tuple(int(t) for t in record["special_ids_in_text"]),
        new_token_count=int(record["new_token_count"]),
    )


def extract_output(
    sequence: Sequence[int],
    prompt_tokens: int,
    decode: Callable[[list[int]], str],
    special_ids: frozenset[int],
    terminators: Mapping[int, str],
    *,
    mask_id: int | None = None,
) -> GeneratedOutput:
    """Cut at the first terminator, decoding without skipping specials; never shorten.

    The new region is everything after the prompt (for Dream, exactly the
    canvas). No terminator inside it means ``incomplete``. Mask IDs left
    anywhere in it are counted; the runner treats any as a technical failure.
    """
    new = [int(token) for token in sequence[prompt_tokens:]]
    index = next((i for i, token in enumerate(new) if token in terminators), None)
    text_ids = new[:index] if index is not None else new
    tail = new[index + 1 :] if index is not None else []
    raw = str(decode(text_ids))
    return GeneratedOutput(
        raw_text=raw,
        text=raw.strip(),
        terminator=terminators[new[index]] if index is not None else None,
        terminator_index=index,
        tail_token_ids=tuple(tail),
        tail_text=str(decode(tail)) if tail else "",
        leftover_mask_count=new.count(mask_id) if mask_id is not None else 0,
        special_ids_in_text=tuple(token for token in text_ids if token in special_ids),
        new_token_count=len(new),
    )


@dataclass(frozen=True)
class Finding:
    code: str
    detail: str

    def record(self) -> dict[str, str]:
        return {"code": self.code, "detail": self.detail}


@dataclass(frozen=True)
class CheckResult:
    profile: str
    failures: tuple[Finding, ...]
    flags: tuple[Finding, ...]
    measures: Mapping[str, Any] = field(default_factory=dict)

    @property
    def excluded(self) -> bool:
        return bool(self.failures)

    def record(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "version": CHECKS_VERSION,
            "excluded": self.excluded,
            "failures": [finding.record() for finding in self.failures],
            "flags": [finding.record() for finding in self.flags],
            "measures": dict(self.measures),
        }


def is_hangul(character: str) -> bool:
    point = ord(character)
    return any(low <= point <= high for low, high in _HANGUL_RANGES)


def non_hangul_runs(text: str) -> list[str]:
    """Maximal runs of letters that are not Hangul (digits and punctuation are not letters)."""
    runs, current = [], ""
    for character in text:
        if character.isalpha() and not is_hangul(character):
            current += character
        elif current:
            runs.append(current)
            current = ""
    if current:
        runs.append(current)
    return runs


def sentence_count(text: str) -> int:
    """Sentences split at . ! ? 。 followed by whitespace; a dot inside 3.5 is no break."""
    parts = re.split(r"(?<=[.!?。])\s+", text.strip())
    return sum(1 for part in parts if part.strip())


def ending_class(text: str) -> tuple[str, str]:
    """CV-10's closed ending classes (v1): the final form, and the final punctuation.

    Forms: ``da`` (declarative 다, ㄴ다, 는다), ``yo`` (polite 요), ``nominal_m``
    (a final syllable closed by ㅁ: 음, 함, 림…), ``noun`` (any other Hangul
    syllable), ``other`` (no Hangul ending).
    """
    stripped = text.rstrip()
    punctuation = (
        "period"
        if stripped.endswith(".")
        else "none"
        if not stripped or stripped[-1].isalnum() or is_hangul(stripped[-1])
        else "other"
    )
    core = stripped.rstrip(_PUNCTUATION + " ")
    if not core or not is_hangul(core[-1]):
        return ("other", punctuation)
    last = core[-1]
    if last == "다":
        return ("da", punctuation)
    if last == "요":
        return ("yo", punctuation)
    point = ord(last) - 0xAC00
    if 0 <= point <= 0xD7A3 - 0xAC00 and point % 28 == _JONGSEONG_MIEUM:
        return ("nominal_m", punctuation)
    return ("noun", punctuation)


def words(text: str) -> list[str]:
    return [word.strip(_PUNCTUATION) for word in text.split() if word.strip(_PUNCTUATION)]


def phrase_hits(text: str, phrases: Iterable[str], *, prefix_last: bool) -> list[str]:
    """Phrases found as whole word sequences; the last word may carry a particle when allowed."""
    tokens, hits = words(text), []
    for phrase in phrases:
        target = phrase.split()
        if not target:
            continue
        for start in range(len(tokens) - len(target) + 1):
            window = tokens[start : start + len(target)]
            head_matches = window[:-1] == target[:-1]
            last_matches = window[-1].startswith(target[-1]) if prefix_last else window[-1] == target[-1]
            if head_matches and last_matches:
                hits.append(phrase)
                break
    return hits


def quality_keys_in(text: str) -> list[str]:
    """Closed-vocabulary qualities used, by key, matching only their two vocabulary forms."""
    return [
        key
        for key, (final, connective, _) in QUALITIES.items()
        if phrase_hits(text, (final, connective), prefix_last=False)
    ]


def repeated_word_ngrams(text: str, n: int = 3) -> list[str]:
    """Word n-grams that occur more than once — a degeneration signal G0 reports."""
    tokens = words(text)
    seen: dict[tuple[str, ...], int] = {}
    for start in range(len(tokens) - n + 1):
        gram = tuple(tokens[start : start + n])
        seen[gram] = seen.get(gram, 0) + 1
    return [" ".join(gram) for gram, count in seen.items() if count > 1]


def consecutive_repeats(text: str) -> int:
    """How many words immediately repeat the word before them."""
    tokens = words(text)
    return sum(1 for before, after in zip(tokens, tokens[1:], strict=False) if before == after)


def check_output(
    source: CaptionSource,
    output: GeneratedOutput,
    *,
    template_phase: int | None = None,
    cell: Mapping[str, Any] | None = None,
    lexicon: Iterable[str] = (),
) -> CheckResult:
    """Every rule for one output against its source.

    ``template_phase`` is the prompt template the job used (G0 renders phase-2
    cell prompts over development-clip cues, so it can differ from the
    source's phase); ``cell`` carries the phase-2 ``quality_limit`` and
    ``context``. ``lexicon`` is the sound-source names of every source in the
    same corpus; a name from it that this cue's evidence lacks is flagged as a
    possibly added sound.
    """
    phase = source.phase if template_phase is None else template_phase
    text = output.text
    failures: list[Finding] = []
    flags: list[Finding] = []
    terms = source.evidence_terms()
    if output.terminator is None:
        failures.append(Finding("incomplete", f"no terminator inside {output.new_token_count} new positions"))
    if output.leftover_mask_count:
        failures.append(Finding("leftover_mask", f"{output.leftover_mask_count} mask tokens left"))
    if output.special_ids_in_text:
        failures.append(Finding("special_token_in_text", str(list(output.special_ids_in_text))))
    if not text:
        failures.append(Finding("empty_text", "nothing before the terminator"))
    else:
        if len(text) > MAX_CHARS[phase]:
            failures.append(Finding("length_over_limit", f"{len(text)} > {MAX_CHARS[phase]} characters"))
        if phase in MAX_LINES and len(text.split("\n")) > MAX_LINES[phase]:
            failures.append(Finding("too_many_lines", f"more than {MAX_LINES[phase]} lines"))
        if sentence_count(text) > 1:
            failures.append(Finding("multiple_sentences", f"{sentence_count(text)} sentences"))
        if "\ufffd" in text:
            # Byte-level BPE can emit part of a multi-byte character; decoding then shows
            # U+FFFD (seen in the Dream smoke run, where a diffusion step committed a
            # continuation byte first). That is not Korean text, so it fails.
            failures.append(Finding("undecodable_bytes", f"{text.count(chr(0xFFFD))} replacement characters"))
        allowed_letters = " ".join([source.text, *terms["sources"], *terms["relations"]])
        foreign = [run for run in non_hangul_runs(text) if run not in allowed_letters]
        if foreign:
            failures.append(Finding("non_korean_script", ", ".join(foreign)))
        if phase == 1 and ending_class(text) != ending_class(source.text):
            failures.append(
                Finding(
                    "sentence_ending_mismatch", f"{ending_class(text)} != source {ending_class(source.text)}"
                )
            )
    used_qualities = quality_keys_in(text)
    sources_used = phrase_hits(text, terms["sources"], prefix_last=True)
    relations_used = phrase_hits(text, terms["relations"], prefix_last=True)
    absent = sorted(set(phrase_hits(text, set(lexicon) - set(terms["sources"]), prefix_last=True)))
    if absent:
        flags.append(Finding("sound_term_absent_from_evidence", ", ".join(absent)))
    allowed_qualities = set(terms["qualities"]) | (set(quality_keys_in(source.text)) if phase == 1 else set())
    outside = [key for key in used_qualities if key not in allowed_qualities]
    if outside:
        flags.append(Finding("quality_outside_evidence", ", ".join(outside)))
    if text and text == source.text:
        flags.append(Finding("identical_to_source", "the output repeats the source text"))
    repeated = repeated_word_ngrams(text)
    if repeated or consecutive_repeats(text):
        flags.append(Finding("repetition", ", ".join(repeated) or "consecutive repeated words"))
    if phase == 2:
        if cell is None:
            raise ValueError("Phase-2 checks need the cell's quality_limit and context")
        limit, context = int(cell["quality_limit"]), str(cell["context"])
        if len(used_qualities) > limit:
            flags.append(Finding("quality_count_over_limit", f"{len(used_qualities)} > {limit}"))
        if context == "broad" and sources_used:
            flags.append(Finding("source_name_at_broad", ", ".join(sources_used)))
        if context != "relation" and relations_used:
            flags.append(Finding("relation_below_relation_step", ", ".join(relations_used)))
    measures = {
        "chars": len(text),
        "new_tokens": output.new_token_count,
        "completion": output.completion,
        "terminator": output.terminator,
        "quality_terms": used_qualities,
        "quality_term_count": len(used_qualities),
        "specific_sources": sources_used,
        "specific_source_count": len(sources_used),
        "relation_terms": relations_used,
        "relation_count": len(relations_used),
        "exact_copy_of_source": bool(text) and text == source.text,
        "repeated_word_trigrams": repeated,
        "consecutive_repeats": consecutive_repeats(text),
        "ending_class": list(ending_class(text)) if text else None,
    }
    return CheckResult(f"p{phase}-v{CHECKS_VERSION}", tuple(failures), tuple(flags), measures)


def script_passes(result: CheckResult) -> bool:
    """G0's script check: no foreign-script letters, and output that can be shown as a caption.

    The failing codes are protocol data (``GATES["G0"]["script_check_failures"]``).
    """
    codes = {finding.code for finding in result.failures}
    return not codes & set(GATES["G0"]["script_check_failures"])
