"""Issue the participants' personal links for the Vercel study, and list them.

    uv run --with "psycopg[binary]==3.3.6" --with psycopg-pool python deploy/vercel/links.py P1 P2 P3
    uv run --with "psycopg[binary]==3.3.6" --with psycopg-pool python deploy/vercel/links.py --range 1 24
    uv run --with "psycopg[binary]==3.3.6" --with psycopg-pool python deploy/vercel/links.py

A link is ``<base>/?code=<access code>&p=<token>``. The token is random and stands for one participant
ID (``diff.session_store``'s ``links`` table, in the purpose's database): the participant never
types an ID, and changing the token opens nobody else's session. Asking again for an ID gives the
same link, so the command is safe to repeat; with no ID it lists every link issued.

The table printed (participant, link) is the researcher's to keep and to send from: it is the
only place a link is tied to an ID outside the database.
"""

from __future__ import annotations

import argparse
import sys

from pull import REPO, VERCEL, database_url

sys.path.insert(0, str(REPO / "src"))

BASE = "https://concap-exp.vercel.app"


def main() -> int:
    from pg_store import DATABASES, PgSessionStore, on_database

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("participants", nargs="*", help="participant IDs (P1, P2, …)")
    parser.add_argument("--range", nargs=2, type=int, metavar=("FIRST", "LAST"), help="P<first> to P<last>")
    parser.add_argument("--purpose", choices=tuple(DATABASES), default="participant")
    parser.add_argument("--base", default=BASE, help="the study's address")
    args = parser.parse_args()
    code = (VERCEL / "access-code").read_text().strip()
    wanted = list(args.participants)
    if args.range:
        wanted += [f"P{number}" for number in range(args.range[0], args.range[1] + 1)]
    store = PgSessionStore(on_database(database_url(), DATABASES[args.purpose]), max_size=1)
    for participant in wanted:
        store.link_issue(participant)
    listed = [pair for pair in store.links() if not wanted or pair[0] in wanted]
    print("participant,link")
    for participant, token in listed:
        print(f"{participant},{args.base}/?code={code}&p={token}")
    print(f"{len(listed)} links ({args.purpose})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
