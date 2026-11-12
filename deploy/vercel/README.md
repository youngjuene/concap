# diff on Vercel

The participant app (`python -m diff live serve`) as a Vercel Function, with every caption
written by Claude through the API. No GPU and no lab machine sits between the participant and
the study: the app runs in Singapore (`sin1`) beside its database, the clips come from Vercel
Blob.

**Both phases, smoke only (2026-10-27).** The app serves `purpose: smoke` until the
researcher's participant gates are lifted.

**Any instance serves any request (2026-11-12).** Vercel offers no session affinity, and until
this date the live watch kept its session in one instance's memory: a request that reached
another instance put the participant's video back at its start. The watch's state is now in
the store (`diff.session_store`, the `watch_*` tables): the sliders and playback reports per
video, every caption written, and a claim per caption being written. A second instance carries
on with the same video and captions (`server_joined` in the log) and writes nothing twice. A
video starts again only when the participant's browser lost it (a reload), logged as
`video_restarted`; `pull.py` counts both. The writer pool grows with the sessions an instance
meets, so `--sessions` is only how many writers an instance starts with.

## What is where

| Thing | Where |
|---|---|
| Study link | `https://concap-exp.vercel.app/?code=<code>`, code in `artifacts/diff/vercel/access-code` |
| Project | `youngjuenes-projects/concap-diff` (Hobby), not connected to GitHub |
| Staged site | `artifacts/diff/vercel/site/` (built by `stage.py`, linked by its `.vercel/`) |
| Sessions | Neon Postgres `concap-sessions` (`sin1`), `DATABASE_URL`: database `neondb` for smoke, `participant` for participants |
| Videos | Blob store `concap-diff-media` (`icn1`, public), named by content hash, as web copies (`artifacts/diff/vercel/web/`): phase 1's remuxed for fast start, phase 2's at 720p |
| Secrets | `ANTHROPIC_API_KEY`, `DIFF_SECRET` (keys the media tokens), production and preview |

## Deploy

From the repository root, after committing what should be served:

```sh
uv run python deploy/vercel/stage.py --out artifacts/diff/vercel/site --upload
cd artifacts/diff/vercel/site && vercel deploy --prod
```

- `stage.py` copies `diff` and the `dpo` modules it imports **as committed at HEAD**, checks the
  setup against the full repository, copies the stills and masks the server reads, and with
  `--upload` puts any video not yet on Blob there. It refuses an uncommitted `deploy/vercel/`
  unless `--dirty` is passed.
- It also copies `data/phase2/phase2-scene.json` when it exists (gate 5: each cue's scene part, numbers
  and labels only, no masks), and `deploy.json` names it; without it every phase-2 cue is acousmatic.
- Phase 2's viewing videos go up as web copies: 720p, video capped at 2 Mbps, frame rate and
  timing kept, sound copied untouched (`WEB_COPY` in `stage.py`). The originals are 1080p60 at
  5.8 Mbps, 665 MB a participant. Hobby's Blob allows 10 GB of downloads a month, and over it
  every clip stops for 30 days. With the copies a participant costs about 355 MB
  (phase 1's clips 115 MB, phase 2's 240 MB), roughly 28 a month. Blob's usage is in the
  project's Observability tab.
- Phase 1's clips go up remuxed with their index (`moov`) first (`FAST_START`): the same packets,
  so a browser can start them without fetching the file's end first.
- `--new-code` makes a new six-character code (letters and digits, case ignored); after the
  deploy, every old link shows the closed page.
- `--closed` stages a site that shows the closed page to everyone, code or not: deploy it to
  close the study, and deploy again without it to reopen (same code, so old links work again).
- `vercel deploy` without `--prod` makes a preview, behind Vercel's login
  (`vercel curl <path> --deployment <url>`).

## Personal links

A participant deploy takes no typed IDs (`entry: link` in `deploy.json`; `stage.py --entry` to
override). Each participant gets their own link, which carries the access code and a random
token that stands for their ID:

```sh
uv run --with "psycopg[binary]==3.3.6" --with psycopg-pool python deploy/vercel/links.py --range 1 24
uv run --with "psycopg[binary]==3.3.6" --with psycopg-pool python deploy/vercel/links.py   # list them
```

It prints `participant,link` rows: keep that table, it is the only place a link is tied to an ID
outside the database. Asking again gives the same links. Opening a link starts the session,
opening it again resumes it, and after the end it shows the completion page. A new access code
(`--new-code`) changes every link's `code`, so issue the list again after one. Smoke deploys
keep the typed entry (`/#enter`), so a walker needs no link.

## Smoke and participant sessions

They go to two databases of the one Neon project, as the lab keeps two stores:
- `--purpose smoke` (the default) → `neondb`;
- `--purpose participant` → `participant` (created 2026-10-28).

So a test P1 never blocks a real P1, and a participant pull holds no test session. `deploy.json`
names the database, and `app.py` swaps it into `DATABASE_URL`. `stage.py --purpose participant`
refuses while the lab's participant gates stand (`_participant_blockers`: the provisional prompts,
the tuner text, the binding model, unreviewed starting captions), as `live serve` does.

Once production serves participants, test on a preview, which sits behind Vercel's login:

```sh
uv run python deploy/vercel/stage.py --out artifacts/diff/vercel/site   # smoke
cd artifacts/diff/vercel/site && vercel deploy                        # a preview, not --prod
```

Then re-stage `--purpose participant`, so the site directory again matches production.

## Bring the data to the lab

```sh
uv run --with "psycopg[binary]==3.3.6" python deploy/vercel/pull.py   # --purpose smoke for tests
uv run python -m diff export --store artifacts/diff/vercel/pulls/<time>/sessions.sqlite \
  --log-dir artifacts/diff/vercel/pulls/<time>/logs \
  --phase1-setup artifacts/diff/phase1/setup-draft.json --out artifacts/diff/export-<date>
```

`pull.py` reads Neon (one consistent snapshot; the study keeps running) into a new directory: a
lab-format `sessions.sqlite` and the live logs as the files the lab server writes. It prints a health line
(sessions finished, and any live watch that restarted its video) and the export command to run
next. It finds the database through `artifacts/diff/vercel/.env.production`
(`vercel env pull ../.env.production --environment production` in the site; secrets come back
as placeholders, the database URL does not).

## A participant withdraws

The consent sheet's data-flow page (`docs/v4-diff/consent-data-flow.md`) promises to delete a
withdrawn participant's records everywhere:

```sh
uv run --with "psycopg[binary]==3.3.6" python deploy/vercel/forget.py P12         # counts only
uv run --with "psycopg[binary]==3.3.6" python deploy/vercel/forget.py P12 --yes   # deletes
```

It deletes the sessions, receipts, events, live logs, shared watch record and personal link from
Neon in one transaction. It then lists
every pull and export on this machine that still holds the participant. Delete those, then pull
and export again.

## How it differs from the lab server

`app.py` builds the same app as `live serve`, with:
- an access gate: only a browser that opened the study link gets past the closed page;
- videos answered by a 307 to their Blob copy (a function's response is at most 4.5 MB);
- one `DIFF_SECRET` for every instance, so a media token issued by one opens on another;
- the lab's session store on Postgres (`pg_store.py`), shared by every instance;
- a writer pool that grows instead of refusing, and personal links instead of typed IDs;
- the live logs as rows of a `logs` table, one per line, since `/tmp` dies with the instance.
