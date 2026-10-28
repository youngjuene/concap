"""Draft data/phase2/scene-keys.csv: each ko_final sound's AudioSet label and the audio mask it uses.

One Claude call per cue window labels the participant's Korean sounds (ko_final) and the English
phrases the 2026-10-22 audio masks were prompted with (en_afnext) together, so the two lists are
labelled alike. Each label must be an exact AudioSet ontology name. Researcher's D2 (2026-10-28):
a sound uses the mask of a phrase in the same window with the same label ("label"); otherwise it has
no mask ("none") and is never placed on screen.

Every call is billed; answers are cached per window under --raw, so a rerun calls only for missing ones.

Usage (repository env, ANTHROPIC_API_KEY set):
  .venv/bin/python scripts/diff_phase2_keys.py --out data/phase2/scene-keys.csv \
      --raw artifacts/diff/phase2-scene/keys-draft
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diff.model_profile import CLAUDE_WRITER  # noqa: E402
from diff.phase1_scene import PLAUSIBLE, ontology_groups  # noqa: E402


def split_phrases(text: str) -> list[str]:
    """sa2va_captions.split_phrases: the phrases the audio masks were prompted with."""
    out: list[str] = []
    for part in re.split(r"[,\n]", str(text)):
        phrase = part.strip().rstrip(".").strip()
        if phrase and phrase not in out:
            out.append(phrase)
    return out


ONTOLOGY = Path("/mnt/hdd/research/2026/Sa2VA/avmask/data/ontology.json")
JOBS = Path("/mnt/hdd/research/2026/Sa2VA/avmask/runs/phase2/jobs.csv")
APPROVED = ROOT / "artifacts/diff/phase1/setup-draft.json"

INSTRUCTIONS = """You label sounds in captions of urban street videos with AudioSet ontology classes.

For every item below give the one AudioSet class name that best names the sound's source, copied
exactly from the list of allowed names. Label the Korean and the English items the same way: two
items naming the same sound must get the same class. Prefer the classes the study already uses
(listed first) when they fit. Use a specific class only when no listed one fits the sound.

Classes the study already uses:
{approved}

Allowed names (the ontology classes whose AudioSet groups the study can place on screen or off it;
when none fits, use "Miscellaneous sources"):
{names}

Items, as JSON:
{items}

Answer with JSON only: {{"labels": {{"<item id>": "<class name>", ...}}}}, one entry per item."""


def split(text: str) -> list[str]:
    return [part.strip().rstrip(".").strip() for part in re.split(r"[,\n]", text) if part.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", type=Path, default=ROOT / "data/phase2/captions.csv")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True, help="one JSON per window: request and answer")
    args = ap.parse_args()

    import anthropic

    entries = json.loads(ONTOLOGY.read_text(encoding="utf-8"))
    groups_of = ontology_groups(ONTOLOGY)
    # D3 (researcher, 2026-10-28): phase 1's PLAUSIBLE map unchanged, so only classes it covers.
    names = sorted(e["name"] for e in entries if all(g in PLAUSIBLE for g in groups_of(e["name"])))
    setup = json.loads(APPROVED.read_text(encoding="utf-8"))
    approved = sorted({o["audioset_label"] for c in setup["clips"] for o in c["options"]["A1s"]})
    masked = {
        (r["video"], int(r["window"]), r["phrase"]): r["label_dir"] for r in csv.DictReader(JOBS.open())
    }

    client = anthropic.Anthropic(max_retries=2)
    args.raw.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(args.captions.open(encoding="utf-8-sig")))
    out = []
    for row in rows:
        video, window = row["video"], int(row["window"])
        sounds, phrases = split(row["ko_final"]), split_phrases(row["en_afnext"])
        items = {f"k{i}": s for i, s in enumerate(sounds)} | {f"e{i}": p for i, p in enumerate(phrases)}
        raw_path = args.raw / f"{video}-w{window:02d}.json"
        if raw_path.is_file():
            labels = json.loads(raw_path.read_text(encoding="utf-8"))["labels"]
        else:
            prompt = INSTRUCTIONS.format(
                approved="\n".join(approved),
                names="\n".join(names),
                items=json.dumps(items, ensure_ascii=False),
            )
            labels = {}
            for _ in range(2):
                message = client.messages.create(
                    model=CLAUDE_WRITER.model, max_tokens=8000, messages=[{"role": "user", "content": prompt}]
                )
                text = "".join(block.text for block in message.content if block.type == "text")
                match = re.search(r"\{.*\}", text, re.S)
                try:
                    labels = json.loads(match.group(0))["labels"] if match else {}
                except json.JSONDecodeError:
                    labels = {}
                if set(labels) == set(items) and all(v in set(names) for v in labels.values()):
                    break
            raw_path.write_text(
                json.dumps(
                    {"items": items, "labels": labels, "model": message.model, "message_id": message.id},
                    ensure_ascii=False,
                    indent=1,
                ),
                encoding="utf-8",
            )
        for i, sound in enumerate(sounds):
            label = labels.get(f"k{i}", "")
            groups = groups_of(label) if label in names else ()
            same = [p for j, p in enumerate(phrases) if labels.get(f"e{j}") == label]
            chosen, how = (same, "label") if same else ([], "none")
            # Of several candidates, the phrase whose words the sound shares most stands.
            phrase = max(chosen, key=lambda p: len(set(p.split()) & set(sound.split())), default="")
            phrase = chosen[0] if chosen and not phrase else phrase
            unmapped = [g for g in groups if g not in PLAUSIBLE]
            out.append(
                {
                    "video": video,
                    "window": window,
                    "sound_index": i,
                    "sound": sound,
                    "audioset_label": label,
                    "groups": "; ".join(groups),
                    "mask_match": how,
                    "mask_phrase": phrase,
                    "mask_dir": masked.get((video, window, phrase), "") if phrase else "",
                    "candidates": " | ".join(
                        f"{p} [{labels.get(f'e{j}', '')}]" for j, p in enumerate(phrases)
                    ),
                    "unmapped_groups": "; ".join(unmapped),
                    "approved_by": "",
                }
            )
        print(
            f"{video} w{window:02d}: "
            + ", ".join(f"{r['audioset_label']}/{r['mask_match']}" for r in out[-len(sounds) :])
        )
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    counts = {how: sum(r["mask_match"] == how for r in out) for how in ("label", "none")}
    print(f"{len(out)} sounds: {counts}; unmapped groups in {sum(bool(r['unmapped_groups']) for r in out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
