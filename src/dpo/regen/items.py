"""§9.2: the survey items, held as data and referenced twice.

The items are a JSON document outside the code. Two things follow from that
which are worth saying, because both are requirements rather than conveniences.

*The ART block is one definition, not two copies.* §3 asks the four ART
sub-factors after the first viewing and §8 asks them again after the second,
and the comparison between the two is the study's main measure. Two literals
that happen to match today would drift the first time someone edits one, and
the drift would look like an effect. So a page names the blocks it asks by id,
and both pages name the same ``art`` block; :func:`page_blocks` resolves the
names against one table.

*A study can change its items without changing this build.* The wording of a
validated instrument is the researcher's, and the default file shipped beside
this module is explicitly not one — it is a placeholder set in the right shape,
so the instrument can be run end to end before the real items exist. A document
says which it is loading, the provenance travels into every response record,
and serving placeholders says so on the operator's console. An analysis can
then never mistake a pilot's placeholder responses for instrument responses.

The response format is not here. §9.3 requires it to be identical across the
survey pages, so it lives once in :class:`dpo.regen.config.Scale` and the items
carry no scale of their own to disagree with it.

Section numbers cite ``docs/v3-regen/spec-behavior.md``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

ITEMS_SCHEMA = "dpo.caption-regen-items/v1"
DEFAULT_ITEMS = "items/default.json"
ID_RE = re.compile(r"[a-z0-9_]+\Z")
PLACEHOLDER = "placeholder"
AUTHORED = "authored"
PROVENANCES = (PLACEHOLDER, AUTHORED)

# The blocks each survey page asks, in order. §3 asks ART alone; §8 asks ART
# again — the same block, so the two are comparable item by item — then the
# caption measures and the PRSS.
PAGE_BLOCKS: Mapping[str, tuple[str, ...]] = {
    "art": ("art",),
    "survey": ("art", "caption", "prss"),
}


class ItemsError(ValueError):
    """The items document is not one a survey page could be built from."""


@dataclass(frozen=True)
class Item:
    id: str
    text: str
    reverse: bool = False

    def record(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "reverse": self.reverse}


@dataclass(frozen=True)
class Block:
    """One named group of items, asked together under one heading."""

    id: str
    title: str
    items: tuple[Item, ...]

    def record(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "items": [item.record() for item in self.items]}


@dataclass(frozen=True)
class ItemSet:
    """Every block a study asks, and where the wording came from."""

    provenance: str
    blocks: Mapping[str, Block]

    @property
    def is_placeholder(self) -> bool:
        return self.provenance == PLACEHOLDER

    @property
    def digest(self) -> str:
        """Identifies the wording. Stamped onto every response record.

        Responses are only comparable across participants if they answered the
        same items; a study that edits one item mid-run gets a different digest
        from that point on, and the analysis can see the seam instead of
        averaging across it.
        """
        payload = json.dumps(
            {
                "provenance": self.provenance,
                "blocks": {k: v.record() for k, v in sorted(self.blocks.items())},
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def page_blocks(self, page: str) -> tuple[Block, ...]:
        """The blocks one page asks, resolved against the one table (§9.2)."""
        try:
            names = PAGE_BLOCKS[page]
        except KeyError:
            raise ItemsError(f"no page named {page!r}; pages are {sorted(PAGE_BLOCKS)}") from None
        missing = [name for name in names if name not in self.blocks]
        if missing:
            raise ItemsError(f"page {page!r} asks blocks the items document does not define: {missing}")
        return tuple(self.blocks[name] for name in names)

    def item_ids(self, page: str) -> tuple[str, ...]:
        """Every item id the page must have an answer for, in order."""
        return tuple(item.id for block in self.page_blocks(page) for item in block.items)


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ItemsError(f"{path}: must be a non-empty string")
    return value


def _identifier(value: object, path: str) -> str:
    text = _string(value, path)
    if not ID_RE.fullmatch(text):
        raise ItemsError(f"{path}: must match [a-z0-9_]+")
    return text


def parse_items(raw: Mapping[str, Any]) -> ItemSet:
    """Validate an items document, naming the path that fails."""
    if raw.get("schema") != ITEMS_SCHEMA:
        raise ItemsError(f"not a {ITEMS_SCHEMA} document")
    provenance = raw.get("provenance")
    if provenance not in PROVENANCES:
        raise ItemsError(
            f"provenance: must be one of {list(PROVENANCES)}, so a pilot cannot pass for a study"
        )
    blocks_raw = raw.get("blocks")
    if not isinstance(blocks_raw, Sequence) or isinstance(blocks_raw, str) or not blocks_raw:
        raise ItemsError("blocks: must be a non-empty list")
    blocks: dict[str, Block] = {}
    seen: set[str] = set()
    for position, entry in enumerate(blocks_raw):
        path = f"blocks[{position}]"
        if not isinstance(entry, Mapping):
            raise ItemsError(f"{path}: must be an object")
        block_id = _identifier(entry.get("id"), f"{path}.id")
        if block_id in blocks:
            raise ItemsError(f"{path}.id: {block_id!r} is defined twice")
        items_raw = entry.get("items")
        if not isinstance(items_raw, Sequence) or isinstance(items_raw, str) or not items_raw:
            raise ItemsError(f"{path}.items: must be a non-empty list")
        items = []
        for index, item_raw in enumerate(items_raw):
            item_path = f"{path}.items[{index}]"
            if not isinstance(item_raw, Mapping):
                raise ItemsError(f"{item_path}: must be an object")
            item_id = _identifier(item_raw.get("id"), f"{item_path}.id")
            # Ids are keys in every response record, so they are unique across
            # the whole document and not only within a block.
            if item_id in seen:
                raise ItemsError(f"{item_path}.id: {item_id!r} is used by another item")
            seen.add(item_id)
            reverse = item_raw.get("reverse", False)
            if not isinstance(reverse, bool):
                raise ItemsError(f"{item_path}.reverse: must be true or false")
            items.append(
                Item(id=item_id, text=_string(item_raw.get("text"), f"{item_path}.text"), reverse=reverse)
            )
        blocks[block_id] = Block(
            id=block_id, title=_string(entry.get("title"), f"{path}.title"), items=tuple(items)
        )
    missing = sorted({name for names in PAGE_BLOCKS.values() for name in names} - set(blocks))
    if missing:
        raise ItemsError(f"blocks: the survey pages need blocks that are not defined: {missing}")
    return ItemSet(provenance=str(provenance), blocks=blocks)


def load_items(path: Path | None = None) -> ItemSet:
    """The study's items, or the placeholder set shipped with this package."""
    if path is None:
        text = files("dpo.regen").joinpath(DEFAULT_ITEMS).read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ItemsError(f"{path or DEFAULT_ITEMS}: not JSON: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ItemsError(f"{path or DEFAULT_ITEMS}: must be an object")
    return parse_items(raw)
