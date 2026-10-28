# diff on Vercel

The participant app (`python -m diff live serve`) as a Vercel Function, with every caption
written by Claude through the API. No GPU and no lab machine sits between the participant and
the study: the app runs in Singapore (`sin1`) beside its database, the clips come from Vercel
Blob.

**Both phases, smoke only (2026-10-27).** The app serves `purpose: smoke` until the
researcher's participant gates are lifted.

**One caveat for phase 2.** The live watch keeps its session in the instance's memory (its
writer, its schedule, its caption tokens); only its progress is in the store. A watch request
that reaches another instance rebuilds the session at the start of the video it was in, so the
participant's video restarts, and a `video-ended` from the old video is refused there. Vercel
offers no session affinity. On the 2026-10-27 walk (P909, three 5-minute videos at real speed)
every request stayed on one instance: one `session_started`, no `session_resumed`. A
`session_resumed` with `video_restarted: true` in the logs is how it would show.

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

## How it differs from the lab server

`app.py` builds the same app as `live serve`, with:
- an access gate: only a browser that opened the study link gets past the closed page;
- videos answered by a 307 to their Blob copy (a function's response is at most 4.5 MB);
- one `DIFF_SECRET` for every instance, so a media token issued by one opens on another;
- the lab's session store on Postgres (`pg_store.py`), shared by every instance;
- the live logs as rows of a `logs` table, one per line, since `/tmp` dies with the instance.
