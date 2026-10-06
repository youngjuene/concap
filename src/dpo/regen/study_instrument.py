"""Rendered v2 study instrument snapshots for response provenance."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from importlib.resources import files
from pathlib import Path
from typing import Any

from dpo.regen.config import (
    DEFAULT_SCALE_ANCHORS,
    DEFAULT_SCALE_ANCHORS_BY_LANGUAGE,
    ConfigError,
    Configuration,
    Scale,
    load_configuration,
)
from dpo.regen.items import ItemSet, load_items
from dpo.regen.study_schema import OVERALL_ITEMS, PREFERENCES, VIDEO_ITEMS, digest

INSTRUMENT_SCHEMA = "dpo.caption-study-instrument/v2"
INSTRUMENT_PROVENANCE = "pilot-placeholder-long-hierarchy-v2"
LOCALIZED_PILOT_SUFFIX = "localized-scale-v1"
AGREEMENT_POINTS = 5


def viewing_translations() -> dict[str, str]:
    """Korean participant-facing copy used by the standalone viewing UI."""
    return dict(json.loads(files("dpo.regen").joinpath("study-ko.json").read_text(encoding="utf-8")))


def _label(text: str, language: str, translations: Mapping[str, str]) -> str:
    return translations.get(text, text) if language == "ko" else text


def rendered_items(
    items: Sequence[Mapping[str, Any]],
    language: str,
    translations: Mapping[str, str] | None = None,
    *,
    group: str | None = None,
    group_title: str | None = None,
) -> list[dict[str, Any]]:
    """Items exactly as the participant sees them, with stable stored values.

    Rating and choice values remain the existing English/numeric response
    contract. Localized strings are labels beside those values, so a Korean
    response can later be interpreted from the exact wording on screen without
    changing the submitted answer values.
    """
    translations = viewing_translations() if translations is None else translations
    rendered = []
    for item in items:
        entry = {key: item[key] for key in ("id", "type") if key in item}
        entry["text"] = _label(str(item["text"]), language, translations)
        if group:
            entry["group"] = group
        if group_title:
            entry["group_title"] = _label(group_title, language, translations)
        if item.get("optional"):
            entry["optional"] = True
        if item.get("na"):
            entry["na"] = True
            entry["na_label"] = _label("Not applicable", language, translations)
        if item["type"] == "rating":
            entry["points"] = int(item.get("points", AGREEMENT_POINTS))
            entry["anchors"] = [
                _label(str(item.get("low", "Strongly disagree")), language, translations),
                _label(str(item.get("high", "Strongly agree")), language, translations),
            ]
        elif item["type"] == "choice":
            options = [str(option) for option in item["options"]]
            entry["options"] = options
            entry["option_labels"] = [_label(option, language, translations) for option in options]
        rendered.append(entry)
    return rendered


def rendered_prss_items(
    item_set: ItemSet, scale: Scale, language: str, translations: Mapping[str, str] | None = None
) -> list[dict[str, Any]]:
    translations = viewing_translations() if translations is None else translations
    anchors = [_label(anchor, language, translations) for anchor in scale.anchors_for(language)]
    prss = next(block for block in item_set.page_blocks("survey") if block.id == "prss")
    return [
        {
            "id": item.id,
            "type": "rating",
            "text": item.wording(language),
            "points": scale.points,
            "anchors": anchors,
            "group": prss.id,
            "group_title": prss.heading(language),
            "block": prss.id,
            "block_title": prss.heading(language),
            **({"reverse": True} if item.reverse else {}),
        }
        for item in prss.items
    ]


def instrument_snapshot(
    language: str,
    translations: Mapping[str, str] | None = None,
    item_set: ItemSet | None = None,
    scale: Scale | None = None,
) -> dict[str, Any]:
    """Versioned pilot snapshot ready to stamp onto a v2 response."""
    item_set = load_items() if item_set is None else item_set
    scale = Scale() if scale is None else scale
    prss = rendered_prss_items(item_set, scale, language, translations)
    video = rendered_items(VIDEO_ITEMS, language, translations, group="video", group_title="This video")
    overall = (
        rendered_items(
            OVERALL_ITEMS,
            language,
            translations,
            group="caption_controls",
            group_title="Captions and controls",
        )
        + prss
    )
    document = {
        "schema": INSTRUMENT_SCHEMA,
        "provenance": INSTRUMENT_PROVENANCE,
        "language": language,
        "flow": "long-video-surveys/v2",
        "prss": {
            "items_digest": item_set.digest,
            "items_provenance": item_set.provenance,
            "scale": scale.record(),
            "block": "prss",
        },
        "items": {
            "preferences": rendered_items(PREFERENCES, language, translations),
            "video": video,
            "overall": overall,
            "final": overall,
        },
    }
    return {**document, "hash": digest(document)}


def localized_scale() -> Scale:
    """The explicit opt-in bilingual pilot scale."""
    return Scale(
        anchors=DEFAULT_SCALE_ANCHORS,
        anchors_by_language=dict(DEFAULT_SCALE_ANCHORS_BY_LANGUAGE),
    )


def localized_pilot_artifact(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    """A fresh config artifact that opts into localized scale anchors.

    The input artifact is left untouched. The returned artifact gets a new
    study id suffix, so serving it naturally uses a new config hash, cookie
    namespace and output directory choice rather than mixing with a live run.
    """
    source = load_configuration(raw_config)
    if source.scale.anchors != DEFAULT_SCALE_ANCHORS:
        raise ConfigError("Custom scale wording requires its own reviewed localized anchors")
    study_id = (
        source.study_id
        if source.study_id.endswith(f"-{LOCALIZED_PILOT_SUFFIX}")
        else f"{source.study_id}-{LOCALIZED_PILOT_SUFFIX}"
    )
    return Configuration(
        study_id=study_id,
        corpus_id=source.corpus_id,
        calibration=replace(source.calibration, scale=replace(localized_scale(), points=source.scale.points)),
    ).artifact()


def prepare_pilot(legacy: Path, output: Path) -> dict[str, Any]:
    """Prepare a separate pilot document without overwriting an existing study."""
    from dpo.regen.document import load_regen_document, validate_regen_document

    document = load_regen_document(legacy)
    document["config"] = localized_pilot_artifact(document["config"])
    suffix = f"-{LOCALIZED_PILOT_SUFFIX}"
    if not document["session_id"].endswith(suffix):
        document["session_id"] += suffix
    validate_regen_document(document)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    return {
        "output": str(output),
        "session_id": document["session_id"],
        "config_hash": load_configuration(document["config"]).hash,
        "fresh_output_required": True,
        "scale_provenance": "draft-translation",
        "questionnaire_wording_changed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a separate bilingual pilot calibration document")
    parser.add_argument("--legacy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = prepare_pilot(args.legacy, args.output)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
