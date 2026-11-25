# deploy/

The participant app (`diff`) is published by Vercel; the lab machine is its
fallback. Nothing in this directory publishes `dpo regen`: its Tailscale
Funnel door (`deploy/edge`, `docker-compose.yml`, `make open`/`close`/
`status`/`code`) was removed in 2026-11.

| What | Where |
|---|---|
| The app on Vercel: staging, deploy, store, pulling the lab export | [`vercel/README.md`](vercel/README.md) |
| The same app on this machine, and which place answers now | `deploy/study.py`; `make local`, `make local-stop`, `make where` |

## Research dashboard

`make dashboard` serves the read-only researcher console at
`http://127.0.0.1:8781/`. It reads the current Phase 1/2 responses and interaction
records independently of the participant server. It is local-only: it listens on
loopback and is not part of any participant-facing deployment. See
[`DASHBOARD.md`](../src/dpo/regen/DASHBOARD.md) for views, exports and data definitions.
