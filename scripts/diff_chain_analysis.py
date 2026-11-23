"""Fit the chain's tests from a research export, or write a simulated export to check them on.

    uv run python scripts/diff_chain_analysis.py fit --export EXPORT_DIR [--out REPORT.json]
    uv run python scripts/diff_chain_analysis.py simulate --out DIR [--participants 48] [--follow 0.3] \\
        [--gamma-context 0.3] [--gamma-texture 0.3] [--recovery 0.65] [--seed N]

The tests and their methods are ``diff.chain_analysis``'s (``docs/v4-diff/chain-spec.md`` §5): H1 and R
confirmatory, H2, H3 and D registered estimates. ``simulate`` writes only the four tables ``fit``
reads, with known effects, so the analysis can be run before any session.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diff.chain_analysis import fit, simulate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fitting = commands.add_parser("fit", help="every test of the chain from one export")
    fitting.add_argument("--export", type=Path, required=True)
    fitting.add_argument("--out", type=Path)
    making = commands.add_parser("simulate", help="a simulated export with known effects")
    making.add_argument("--out", type=Path, required=True)
    making.add_argument("--participants", type=int, default=48)
    making.add_argument("--follow", type=float, default=0.3)
    making.add_argument("--gamma-context", type=float, default=0.3)
    making.add_argument("--gamma-texture", type=float, default=0.3)
    making.add_argument("--recovery", type=float, default=0.65)
    making.add_argument("--seed", type=int, default=20261123)
    args = parser.parse_args()
    if args.command == "simulate":
        if args.out.exists() and any(args.out.iterdir()):
            raise SystemExit(f"{args.out} is not empty")
        made = simulate(
            args.out,
            participants=args.participants,
            follow=args.follow,
            gamma_context=args.gamma_context,
            gamma_texture=args.gamma_texture,
            recovery=args.recovery,
            seed=args.seed,
        )
        print(json.dumps(made, indent=1))
        return 0
    report = json.dumps(fit(args.export), ensure_ascii=False, indent=1)
    if args.out:
        args.out.write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
