"""``python -m diff`` — the diff pipeline's command line (plan §10, Stage 1).

Run from the repository root as ``uv run python -m diff …``. Each handler
imports what it needs, so ``--help`` never imports torch or transformers.
There is deliberately no console script: a ``diff`` executable in the venv
would shadow the system ``diff``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any, *, force: bool = False) -> None:
    if path.exists() and not force:
        raise SystemExit(f"{path} exists; pass --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sources(path: Path) -> Any:
    from diff.caption_sources import SourceError, parse_sources

    try:
        return parse_sources(_read_json(path))
    except SourceError as exc:
        raise SystemExit("invalid sources:\n  " + "\n  ".join(exc.problems)) from exc


def sources_validate(args: argparse.Namespace) -> int:
    from diff.caption_sources import bundle_problems

    source_set = _sources(args.sources)
    units: dict[tuple[int, str, str], int] = {}
    for source in source_set.sources:
        unit = (source.phase, source.unit_key, source.material)
        units[unit] = units.get(unit, 0) + 1
    for (phase, key, material), count in sorted(units.items()):
        print(f"phase {phase} {key} ({material}): {count} cues")
    ineligible = [problem for source in source_set.sources for problem in bundle_problems(source)]
    print(f"{len(source_set.sources)} sources; {len(ineligible)} bundle-eligibility problems")
    for problem in ineligible[: args.show]:
        print(f"  {problem}")
    return 0


def sources_import(args: argparse.Namespace) -> int:
    from diff.caption_sources import import_regen_document, import_viewing_manifest

    document = _read_json(args.input)
    importer = import_viewing_manifest if args.kind == "viewing" else import_regen_document
    imported = importer(document, args.material)
    _write_json(args.out, imported.record(), force=args.force)
    print(f"wrote {len(imported.sources)} draft {args.material} sources to {args.out} (unreviewed)")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python -m diff", description=__doc__.split("\n\n")[0])
    commands = root.add_subparsers(dest="command", required=True)

    sources = commands.add_parser("sources", help="validate or import caption sources").add_subparsers(
        dest="action", required=True
    )
    validate = sources.add_parser("validate", help="validate a diff.caption-sources/v1 file")
    validate.add_argument("--sources", type=Path, required=True)
    validate.add_argument("--show", type=int, default=20, help="eligibility problems to print")
    validate.set_defaults(handler=sources_validate)
    for kind, help_text in (
        ("viewing", "regen viewing manifest → phase-2 drafts"),
        ("regen", "regen document → phase-1 drafts"),
    ):
        imported = sources.add_parser(f"import-{kind}", help=help_text)
        imported.add_argument("--input", type=Path, required=True)
        imported.add_argument("--out", type=Path, required=True)
        imported.add_argument("--material", choices=("development", "participant"), default="participant")
        imported.add_argument("--force", action="store_true")
        imported.set_defaults(handler=sources_import, kind=kind)

    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
