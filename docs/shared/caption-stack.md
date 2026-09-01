# The caption stack — what both instruments share

`dpo.caption` is the only code the two participant instruments have in common.
It exists so that `dpo.session` (v1) and `dpo.console` (v2) cannot drift apart
on the expensive, model-shaped half of the problem, and so that archiving
either one is a directory deletion rather than an untangling.

```text
src/dpo/caption/     writer.py, media.py     ← shared
src/dpo/session/     v1, docs/v1-session/    ← imports dpo.caption
src/dpo/console/     v2, docs/v2-console/    ← imports dpo.caption
```

The dependency runs one way and a test asserts it
(`tests/console/test_detachment.py`): nothing in `dpo.caption` may import
either instrument, and neither instrument may import the other.

## The seam is `CaptionRequest`

Each instrument reaches a caption its own way. v1 builds a request from a
skeleton ordering (`dpo.session.writer.build_request`); v2 builds one from a
regime and a grain (`dpo.console.requests.RequestBuilder`). Everything upstream
of that — what a control surface is, what a source's weights mean, what the
participant touched — is the instrument's own business, and the two disagree
about all of it.

From the request onward the machinery is identical:

| Piece | What it does |
|---|---|
| `TemplateWriter` | deterministic string assembly, so an instrument can be built, tested and demonstrated without a GPU |
| `GemmaWriter` | the study's Gemma 4 E4B, decoding frozen at temperature 0 and seed 0, with the character budget enforced after the fact |
| `CachedWriter` | identical settings return the identical caption; one lock, one JSON file under `--out` |
| `CAPTION_MAX_CHARS` | 96 — the two reserved lines of the caption box at the measure both instruments typeset it at |
| `media.py` | finding a clip's video, cutting a shot's audio, taking a still |

## What each instrument still owns

`GemmaWriter` takes an `instruction=` callable, because the two say different
things to the model. v1 tells it about levels and role heads and, on a control
clip, asks for a visual description from a still. v2 tells it about grain, and
about which sources are out of frame so it writes them as arriving from outside
it. Neither instruction lives in the shared module; each is a constant in its
own package where a reader can inspect exactly what the model was told.

The template renderers are likewise separate — `TemplateWriter` here for v1's
levels, `ConsoleTemplateWriter` in `dpo.console.requests` for v2's grains —
because they render different distinctions.

## When the loser is archived

Delete the package, its tests, and its documentation directory:

```bash
# if v2 is not adopted
rm -rf src/dpo/console tests/console docs/v2-console src/dpo/cli/console.py
# then drop the `register_console(commands)` line from src/dpo/cli/__init__.py
```

Nothing in `dpo.caption` refers to either instrument except in prose, so the
shared stack needs no edit. The one thing to check afterwards is
`GemmaWriter`'s `instruction=` parameter: with a single caller it can collapse
back into a module constant, but it costs nothing to leave.
