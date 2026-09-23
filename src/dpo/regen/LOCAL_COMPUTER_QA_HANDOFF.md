# Local Computer Use handoff — 2026-09-22

The source is on SSH host `tx01`, workspace
`/mnt/hdd/research/2026/concap/src/dpo/regen`.

An isolated current-source real-media/Gemma QA server is running on the host at
`http://127.0.0.1:18789`. Its output is
`/tmp/regen-qualification-20260922/{out,viewing-out}`. It uses GPU 1 and the
same base model/backend/contract as the existing live service. The QA server
has no playback-completion seeding routes. `/qa/source` exposes its media/model
configuration for verification; the mounted study provides its own QA export.

The public/live service on 8779 has not been restarted. A read-only version
probe found its Python backend lacks `/api/playback` (404) while its served
`regen.js` matches the current workspace. Qualify the isolated instance to test
the current coherent frontend/backend version.

## Task for a local Codex session with Computer Use

Use the local Computer Use/browser capability to perform actual participant
journeys against the SSH-hosted QA instance. Discover the existing SSH host alias
for `tx01` from the local machine's connection configuration. Establish a local
forward equivalent to:

```sh
ssh -N -L 18789:127.0.0.1:18789 <existing-SSH-host-alias>
```

If local port 18789 is occupied, choose another local port and keep the remote
destination 127.0.0.1:18789. Open the resulting localhost URL in the local browser.
Confirm the server and source metadata before starting. Use fresh isolated
browser sessions for EN and KO; record which local OS/browser and connection
route were actually used.

1. Play the two short clips normally, each with original and updated captions.
   Submit the actual ART, visual points, five sound reports, updated-caption
   questions and chapter PRSS through the displayed controls.
2. Verify keyboard and pointer interaction, fullscreen exit/resume, one reload
   during playback, draft restoration and failed-upload recovery. Do not edit
   media time, state, coverage or generation outputs.
3. Complete all three real long videos at normal playback speed. Exercise each
   slider independently, pad, presets, reset and an A-to-B-to-A reversal. Check
   that selected levels change immediately but applied-caption labels change
   only with displayed captions. Complete each video's four questions and the
   final chapter questionnaire through the page.
4. Capture screenshots, visible errors and the final receipt. Reconcile the QA
   exports against viewing counts, played coverage, surveys, event scope,
   acknowledged revisions, generated/fallback exposures and frozen hashes.
5. Observe GPU 1 and the isolated worker's job results via SSH. Report cold and
   warm generation separately, fallback reasons/rate, control-to-visible-caption
   delay and resource use. A successful model call does not by itself establish
   full E2E, caption quality or multi-participant capacity.
6. Obtain independent architecture review from a separate supported reviewer
   session/role with source access. Browser QA cannot award architecture approval.
   Use `PROTOCOL_IMPLEMENTATION.md`, the approved plan and qualification evidence.
   Follow that session's runtime/role prerequisites; the present remote thread's
   OMX preflight returned `unsupported_documented_leader_proof` and its architect
   role previously used an unsupported model.

Keep QA data separate from participant data. Do not restart the live 8779 service,
change study instruments or modify the configured latency ceiling during QA.
Report a failure if the configured cold-start budget is missed rather than
raising it to make a test pass. Release the isolated GPU worker when testing is
finished; keep the actual test evidence and report paths.

## Evidence already gathered in the remote session

- Outside the sandbox, both RTX 3090s are accessible. GPU 1 was nearly empty
  before QA; the model-backed smoke occupied approximately 15.6 GiB afterward.
- A real remote-host Chromium smoke watched the first short clip, submitted its
  surveys/observations and invoked Gemma. Cold generation returned model text
  but took 21,904 ms, exceeding the unchanged 20,000 ms phase-1 ceiling, so the
  interface correctly served the recorded fallback. This is a cold-budget miss,
  not a passing latency qualification.
- Cold artifacts: `/tmp/regen-qualification-20260922/browser-smoke/`.
- A second real-browser participant used the other short clip: warm Gemma
  generation completed in 906 ms without fallback. This used a different clip
  and prompt, rather than replaying the cold request from cache.
- Warm artifacts: `/tmp/regen-qualification-20260922/browser-warm/`.
- Those browser runs originate on `tx01`; they do not count as local Computer Use
  or a physical local-to-SSH browser journey.
