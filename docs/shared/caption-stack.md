# The caption stack — what both instruments share

`dpo.caption` is the only code the two participant instruments have in common.
It exists so that `dpo.session` (v1) and `dpo.console` (v2) cannot drift apart
on the expensive, model-shaped half of the problem, and so that archiving
either one is a directory deletion rather than an untangling.

```text
src/dpo/caption/     writer.py, media.py, background.py, ontology.py   ← shared
tests/caption/       its tests, and the seam test                      ← survives either archival
src/dpo/session/     v1, docs/v1-session/    ← imports dpo.caption
src/dpo/console/     v2, docs/v2-console/    ← imports dpo.caption
```

The dependency runs one way and a test asserts it
(`tests/caption/test_detachment.py`): nothing in `dpo.caption` may import
either instrument, and neither instrument may import the other. The same test
watches the test tree, because the seam leaked there first: an instrument's
tests may not import the other's package, and every module of the shared
package has a test file under `tests/caption`, so no archival can take the
last tests of code that survives it.

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

```bash
# if v1 is not adopted
rm -rf src/dpo/session tests/session docs/v1-session src/dpo/cli/session.py
# then drop the `session` block and its imports from src/dpo/cli/__init__.py,
# and the session-demo target from the Makefile
```

`tests/caption/` stays in both directions: it holds the tests of the shared
package and the seam test itself, so `make check` is green after either
deletion without recovering a test from history. Ruff and mypy fail closed on
anything the edit to `src/dpo/cli/__init__.py` misses.

Nothing in `dpo.caption` refers to either instrument except in prose, so the
shared stack needs no edit to keep working. Two things are worth doing
afterwards. `GemmaWriter`'s `instruction=` parameter can collapse back into a
module constant with a single caller, though it costs nothing to leave. And if
v1 is the one archived, the shared writer is left holding v1's own text — the
level rules, the template's rendering of heads and scene rows, the
control-clip instruction — which is then dead code inside the surviving
package and should be deleted with it.
