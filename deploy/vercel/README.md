# diff on Vercel

The participant app (`python -m diff live serve`) as a Vercel Function, with every caption
written by Claude through the API. No GPU and no lab machine sits between the participant and
the study: the app runs in Singapore (`sin1`) beside its database, the clips come from Vercel
Blob.

**Phase 1 only, smoke only (2026-10-27).** Phase 2's videos are not uploaded yet, and the app
serves `purpose: smoke` until the researcher's participant gates are lifted.

## What is where

| Thing | Where |
|---|---|
| Study link | `https://concap-exp.vercel.app/?code=<code>`, code in `artifacts/diff/vercel/access-code` |
| Project | `youngjuenes-projects/concap-diff` (Hobby), not connected to GitHub |
| Staged site | `artifacts/diff/vercel/site/` (built by `stage.py`, linked by its `.vercel/`) |
| Sessions | Neon Postgres `concap-sessions` (`sin1`), `DATABASE_URL` |
| Videos | Blob store `concap-diff-media` (`icn1`, public), named by content hash |
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
- `--new-code` makes a new six-character code (letters and digits, case ignored); after the
  deploy, every old link shows the closed page.
- `--closed` stages a site that shows the closed page to everyone, code or not: deploy it to
  close the study, and deploy again without it to reopen (same code, so old links work again).
- `vercel deploy` without `--prod` makes a preview, behind Vercel's login
  (`vercel curl <path> --deployment <url>`).

## Bring the data to the lab

```sh
uv run --with "psycopg[binary]==3.3.6" python deploy/vercel/pull.py
uv run python -m diff export --store artifacts/diff/vercel/pulls/<time>/sessions.sqlite \
  --log-dir artifacts/diff/vercel/pulls/<time>/logs \
  --phase1-setup artifacts/diff/phase1/setup-draft.json --out artifacts/diff/export-<date>
```

`pull.py` reads Neon (one consistent snapshot; the study keeps running) into a new directory: a
lab-format `sessions.sqlite` and the live logs as the files the lab server writes. It prints the
export command to run next. It finds the database through `artifacts/diff/vercel/.env.production`
(`vercel env pull ../.env.production --environment production` in the site; secrets come back
as placeholders, the database URL does not).

## How it differs from the lab server

`app.py` builds the same app as `live serve`, with:
- an access gate: only a browser that opened the study link gets past the closed page;
- videos answered by a 307 to their Blob copy (a function's response is at most 4.5 MB);
- one `DIFF_SECRET` for every instance, so a media token issued by one opens on another;
- the lab's session store on Postgres (`pg_store.py`), shared by every instance;
- the live logs as rows of a `logs` table, one per line, since `/tmp` dies with the instance.
