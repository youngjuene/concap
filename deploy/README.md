# The public edge of `dpo regen`

`https://concap.<your tailnet>.ts.net` → Tailscale's public Funnel ingress →
encrypted to the Tailscale node in one container here, which ends the TLS
connection → the instrument on `127.0.0.1:8779`. There is no login for
participants: the switch is the access control, and the link carries an
access code so a leaked or stale link enrols nobody. Open, and anyone with
the current link is a participant; closed, and the link stops answering.
Nothing on this machine listens publicly, no port is opened, and nothing of
the website's is involved. The reasoning is in `docs/v3-regen/runbook.md`,
"Running it for someone off campus"; this file is the operator's card.

## Four commands, from the repo root

| Do | Command |
|---|---|
| Open: instrument serving, Funnel on, link printed | `make open` |
| Close: link stops answering (instrument kept warm, reopening is instant) | `make close` |
| See what is up, what is down, and why | `make status` |
| New access code: old links stop enrolling at once, new link printed | `make code` |

The link looks like `https://concap.<tailnet>.ts.net/?code=word-word-word`.
Without the right code the page says the study is not open. The code lives
in `data/live/access-code`, is read by the instrument on every enrolment,
and `make code` replaces it without restarting anything.

While open, the instrument runs in its public profile: the log download on
the last screen is offered only on this machine (the files under `--out` are
the record anyway), and requests are rate-limited by address, twenty a
second with a burst of sixty, five new enrolments a minute.

Behind them is one script, `deploy/edge`, which also takes
`edge close --gpu` (ends the instrument too, freeing the GPU) and passes
flags to the instrument when it has to start: `deploy/edge open
--items data/live/items.json`. Its terminal is `tmux attach -t regen`
(detach with `Ctrl-b d`). The kiosk browser on this machine uses
`http://127.0.0.1:8779/`.

## The first `make open`

It stops once or twice for things only you can click and prints the link
each time. For the sign-in it waits up to fifteen minutes for you to finish,
then carries on by itself; for anything else it says "then run: make open".
In order:

1. **Sign in.** The machine has to join your Tailscale account. Open the
   printed `login.tailscale.com` link, sign in (Google works), approve the
   machine named `concap`. No account yet: the same link lets you create
   one, free.
2. **Approve publishing**, if Tailscale asks. Publishing to the internet
   needs HTTPS certificates on and the Funnel permission; a new account
   usually has both, and if not, `make open` prints the exact link that
   turns each on.

3. **Wait, once.** A brand-new name takes a few minutes to reach public
   DNS and to get its certificate. `make open` waits up to twelve minutes,
   then prints the link. If it gives up, the link is still published;
   `make status` a few minutes later shows it answering.

After that, every `make open` just prints the link, which never changes.

## Reading `make status`

| Line | Meaning | Do |
|---|---|---|
| Instrument NOT listening | the study server is down | `make open` |
| Container not running | the Tailscale node is down | `make open` |
| Node not signed in | the machine left your account | `make open`, click the link |
| Funnel off | closed | nothing, or `make open` |
| Probe 200 | open, over the internet, from here | nothing |
| Probe 502 | Funnel on, instrument not answering | `make open` |
| Probe 000 | Funnel on, link not answering at all | wait a minute (certificate being fetched), then `make status` again |

## After a reboot

The node's container restarts on its own and keeps its identity (the
`state` volume). Whether Funnel is on is remembered too. The instrument does
not restart on its own, and `make open` starts it, so a reboot changes
nothing about the procedure.

## What can go wrong, and where it shows

- **Your account.** If the machine is removed from the account, or the
  account's Funnel permission is withdrawn, the link stops. `make status`
  says "not signed in" or `make open` prints the approval link.
- **Tailscale's service.** Funnel traffic rides Tailscale's ingress. An
  outage there is a probe of 000 with everything here fine. Nothing to fix
  on this side.
- **Bandwidth.** Tailscale caps Funnel bandwidth and does not publish the
  number. A slow link means a longer "Preparing the clip…" before Start, not
  a different stimulus: the page fetches the whole clip before it enables
  Start and plays it from memory.
- **The certificate.** Tailscale fetches a Let's Encrypt certificate for
  the node's name at the first HTTPS request; `make open` waits for it.

## Files

| File | What |
|---|---|
| `edge` | the command behind the four make targets |
| `docker-compose.yml` | the Tailscale node: pinned image, userspace networking, host network, state volume |
