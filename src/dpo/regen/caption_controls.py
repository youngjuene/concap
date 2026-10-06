"""Bound the low-context vocabulary; retain the model's raw output separately.

The model selects audible categories and acoustic qualities. Rendering those
selections through a closed vocabulary prevents scene/location text escaping a
low-context request. This is an output contract, not an audio-grounding judge.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from dpo.regen.captions import in_language
from dpo.regen.config import SOUND_FAMILIES

CONTROL_SCHEMA = "category-acoustics/v1"
CATEGORIES_KO = {
    "human": "사람 소리",
    "animal": "동물 소리",
    "things": "교통·기계 소리",
    "music": "음악 소리",
    "natural": "자연 소리",
    "other": "소리",
}
CATEGORIES_EN = {
    "human": "human sounds",
    "animal": "animal sounds",
    "things": "traffic and machinery sounds",
    "music": "music sounds",
    "natural": "nature sounds",
    "other": "sounds",
}
# Final adjective, connective adjective, English adjective. No spatial terms.
QUALITIES = {
    "soft": ("작은", "작고", "soft"),
    "loud": ("큰", "크고", "loud"),
    "low_pitched": ("음이 낮은", "음이 낮고", "low-pitched"),
    "high_pitched": ("음이 높은", "음이 높고", "high-pitched"),
    "steady": ("일정한", "일정하고", "steady"),
    "intermittent": ("간헐적인", "간헐적이고", "intermittent"),
    "rhythmic": ("규칙적인", "규칙적이고", "rhythmic"),
    "irregular": ("불규칙적인", "불규칙적이고", "irregular"),
    "rough": ("거친", "거칠고", "rough"),
    "smooth": ("부드러운", "부드럽고", "smooth"),
    "sharp": ("날카로운", "날카롭고", "sharp"),
    "dull": ("둔탁한", "둔탁하고", "dull"),
    "metallic": ("금속성인", "금속성이고", "metallic"),
    "muffled": ("먹먹한", "먹먹하고", "muffled"),
    "rising": ("점점 커지는", "점점 커지고", "growing louder"),
    "fading": ("점점 작아지는", "점점 작아지고", "fading"),
}
SILENCE = {"ko": "뚜렷하게 들리는 소리가 없다.", "en": "No clearly audible sounds."}
NO_AUDIBLE_EVENT = {
    "no sounds are audible in the excerpt",
    "no sounds are audible in the current excerpt",
    "no sounds are audible",
    "no audible sounds",
    "no clearly audible sounds",
}
CATEGORY_ALIASES = {label: key for key, label in SOUND_FAMILIES.items()}
CATEGORY_ALIASES.update({"nature": "natural", "traffic": "things"})


def output_instruction(language: str, axes: Mapping[str, float]) -> str:
    if axes["context"] < 0.34:
        limit = round(axes["texture"] * 4)
        return (
            "Return one JSON object with exactly two keys: categories and qualities. "
            f"categories is an array of 1–3 unique keys from {list(SOUND_FAMILIES)} or other or none. "
            "Listen to the attached audio and select the categories of the sounds you hear. "
            "Use other for an audible sound whose source cannot be identified. "
            "Use none alone only for silence, never merely because a sound is difficult to identify. "
            "Never infer an audible event from the visible scene. "
            f"qualities is an array of 0–{limit} unique keys from {list(QUALITIES)}. "
            "Use an empty qualities array for silence or when no quality is supported. "
            "Do not add source names, locations, prose, or other keys. "
            "The interface renders these selections in the study language. "
            "JSON 키와 열거값은 위 영문 그대로 사용하세요."
        )
    language_rule = (
        "최종 문장은 반드시 한국어로만 쓰세요. 영어로 답하지 마세요. "
        if language == "ko"
        else "Respond only in English. "
    )
    return "One readable sentence, at most 160 characters. " + language_rule


def render_result(raw: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    language = spec["language"]
    if language not in SILENCE:
        raise ValueError("Unsupported caption language")
    text = " ".join(raw.split())
    if text.casefold().rstrip(".") in NO_AUDIBLE_EVENT:
        return {"text": SILENCE[language], "normalization": "localized_no_audible_event"}
    axes = spec.get("axes", {"texture": 0.5, "context": 0.5})
    if axes["context"] < 0.34:
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw.strip(), flags=re.DOTALL)
        try:
            value = json.loads(fenced.group(1) if fenced else raw)
        except (ValueError, TypeError) as exc:
            raise ValueError("Low context requires a structured category/acoustics result") from exc
        if not isinstance(value, dict) or set(value) != {"categories", "qualities"}:
            raise ValueError("Invalid structured caption fields")
        categories, qualities = value["categories"], value["qualities"]
        if isinstance(categories, list) and all(isinstance(key, str) for key in categories):
            categories = [CATEGORY_ALIASES.get(key, key) for key in categories]
        if (
            not isinstance(categories, list)
            or not 1 <= len(categories) <= 3
            or any(not isinstance(key, str) for key in categories)
            or not set(categories) <= {*SOUND_FAMILIES, "other", "none"}
            or len(set(categories)) != len(categories)
            or ("none" in categories and categories != ["none"])
        ):
            raise ValueError("Invalid structured sound categories")
        if (
            not isinstance(qualities, list)
            or any(not isinstance(key, str) for key in qualities)
            or not set(qualities) <= QUALITIES.keys()
            or len(set(qualities)) != len(qualities)
            or (categories == ["none"] and qualities)
        ):
            raise ValueError("Invalid structured acoustic qualities")
        limit = round(axes["texture"] * 4)
        selected = qualities[:limit]
        if categories == ["none"]:
            text = SILENCE[language]
        elif language == "ko":
            adjectives = " ".join(
                QUALITIES[key][0 if index == len(selected) - 1 else 1] for index, key in enumerate(selected)
            )
            sources = "와 ".join(CATEGORIES_KO[key] for key in categories)
            text = f"{adjectives + ' ' if adjectives else ''}{sources}가 들린다."
        else:
            adjectives = ", ".join(QUALITIES[key][2] for key in selected)
            sources = " and ".join(CATEGORIES_EN[key] for key in categories)
            text = f"{adjectives + ' ' if adjectives else ''}{sources} are audible."
            text = text[0].upper() + text[1:]
        result: dict[str, Any] = {
            "text": text,
            "rendering": {
                "schema": CONTROL_SCHEMA,
                "categories": categories,
                "qualities": selected,
                "omitted_qualities": qualities[limit:],
                "raw_categories": value["categories"],
            },
        }
    else:
        result = {"text": text}
    if not text or len(text) > 160 or not in_language(text, language):
        raise ValueError("Generated caption failed language/length validation")
    # Reject the observed purely visual descriptions. This lexical check is a
    # minimum sound-caption contract, not a claim of perceptual grounding.
    auditory = (
        r"소리|들리|들린|들려|음악|대화|울리|울린|발걸음|웅웅|덜컹"
        if language == "ko"
        else r"sound|audib|hear|footstep|engine|rumbl|honk|chirp|bark|buzz|rustl|rattl|"
        r"music|speech|talk|convers|thunder|rain|wind|whistl|tap|clap|splash|traffic|silence"
    )
    if not re.search(auditory, text, flags=re.IGNORECASE):
        raise ValueError("Generated caption has no auditory description")
    return result


def repair_instruction(spec: Mapping[str, Any]) -> str:
    """Retry the same audio once; never treat failed model prose as instructions."""
    return (
        str(spec["instruction"])
        + "\nThe previous response failed the caption output contract. Re-listen to the attached audio. "
        "Describe the sound, not a list of visible objects. Do not claim a visible object makes a sound "
        "unless the audio supports it. "
        + output_instruction(spec["language"], spec.get("axes", {"texture": 0.5, "context": 0.5}))
    )
