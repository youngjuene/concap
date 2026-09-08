/* The participant's side of the six steps.

   Three things this file is careful about, all of them requirements rather
   than preferences. Section numbers cite docs/v3-regen/spec-behavior.md;
   findings cite docs/v3-regen/interface-audit.md.

   The step comes from the server. Every screen is rendered from what
   /api/step/<step> returns, and every submit re-reads the step the server
   reports back. The page never decides where it is, so a reload, a second tab
   and a back button all land on the same screen the log says the session is on
   (§9.1). A 409 means the server disagrees, and the page follows the server.

   Nothing measured is ever shown. §4's points get a count and nothing else:
   the page never learns which object a point hit, because matching happens
   once on submit (§4) and telling the participant would turn the task into
   hunting for a mask.

   Play and select are separate controls (§5). They sit at opposite ends of the
   lane and neither handler touches the other's state, so a participant cannot
   select a lane by trying to hear it.

   And one thing it is careful about now that it was not: every control here is
   reachable by keyboard and says what it is. §4 gated the rest of the session
   behind divs driven by pointer events, so a participant who could not use a
   trackpad could not reach §5 either. */

const $ = (id) => document.getElementById(id);

const state = {
  participant: null,
  step: null,
  strings: null,
  // Every language's chrome, keyed by tag, and `strings` is the one being
  // read. Both are held because §9.3 lets the participant switch until the
  // first clip plays, and a switch should redraw rather than re-fetch.
  chrome: null,
  scale: null,
  minimumPoints: 3,
  ceilingMs: 20000,
  cueSlots: 4,
  steps: [],
  entered: null,
  points: [],
  frame: 0,
  frames: [],
  language: "en",
  languages: [],
  languageLocked: false,
  // The second viewing's clip, fetched during §6's wait. A promise rather than
  // a blob: the viewing may open before the fetch has finished, and awaiting
  // the one in flight is right where starting a second one would not be.
  warming: null,
};

const SCREENS = [
  "screen-start",
  "screen-survey",
  "screen-visual",
  "screen-auditory",
  "screen-waiting",
  "screen-done",
  "screen-closed",
  "screen-error",
  "screen-viewing",
];

function show(id) {
  for (const screen of SCREENS) $(screen).hidden = screen !== id;
  $("shell").hidden = id === "screen-viewing";
}

function clear(node) {
  node.replaceChildren();
  return node;
}

function fill(template, values) {
  return Object.entries(values).reduce(
    (text, [key, value]) => text.replaceAll(`{${key}}`, String(value)),
    template
  );
}

function syncChrome() {
  document.title = state.strings.app_title;
  $("rail").setAttribute("aria-label", state.strings.rail.label);
}

/* Where the rail's marker sits. The wait is not one of §9.1's six steps but a
   participant standing on it is nearly through the fifth, and telling them so
   is the cheapest answer there is to §6's drop-off risk. */
function railAt() {
  if (state.step === "regenerating") return state.steps.indexOf("view_regenerated");
  if (state.step === "done") return state.steps.length;
  return state.steps.indexOf(state.step);
}

/* The step marker above every heading: which of six this is, then the study's
   name. The heading itself is the step's own name. */
function head(prefix, heading) {
  const at = Math.min(Math.max(railAt(), 0), state.steps.length - 1);
  $(`${prefix}-eyebrow`).textContent = fill(state.strings.eyebrow, {
    n: at + 1,
    total: state.steps.length,
    study: state.strings.app_title,
  });
  if (heading !== undefined) $(`${prefix}-heading`).textContent = heading;
}

function stepName(index) {
  return state.strings.steps[index] || "";
}

/* The study is not taking new participants: no code in the link, a stale
   one, or the operator has closed it. Not an error, so not the error screen:
   nothing has gone wrong and there is no researcher in the room to tell. */
function closed() {
  const copy = state.strings.closed;
  $("closed-eyebrow").textContent = state.strings.app_title;
  $("closed-heading").textContent = copy.heading;
  $("closed-body").textContent = copy.body;
  show("screen-closed");
}

function fail(message, cause) {
  const strings = state.strings?.error;
  if (!strings) {
    $("error-text").textContent = message || "Something went wrong.";
    return show("screen-error");
  }
  head("error", strings.heading);
  $("error-text").textContent = strings.body;
  $("error-kept").textContent = strings.kept;
  // The researcher is standing behind the participant, not reading a server
  // log, so this screen is the whole diagnostic surface. All three of these
  // were already in memory when fail() threw away the message and printed one
  // generic sentence instead.
  const labels = strings.labels;
  const width = Math.max(...Object.values(labels).map((label) => label.length));
  const pad = (label) => label.padEnd(width, " ");
  $("error-diagnosis").textContent = [
    `${pad(labels.session)}  ${state.participant || "—"}`,
    `${pad(labels.step)}  ${state.step || "—"}`,
    `${pad(labels.at)}  ${new Date().toISOString()}`,
    `${pad(labels.cause)}  ${cause || message || strings.unknown}`,
  ].join("\n");
  // progress.py keeps the state, so a dropped packet during /api/survey ends a
  // session the server would happily have resumed. Re-reading the step the
  // server reports is safe by construction: it cannot move the session on.
  const retry = $("error-retry");
  retry.textContent = state.strings.actions.retry;
  retry.disabled = false;
  retry.onclick = async () => {
    retry.disabled = true;
    try {
      const next = await api(`/api/state?participant=${state.participant}`);
      if (next) await render(next.step);
    } catch (error) {
      retry.disabled = false;
      fail(error.message, error.message);
    }
  };
  show("screen-error");
}

async function api(path, payload) {
  const options = payload
    ? { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(payload) }
    : {};
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    // The server is the authority on the step; a 409 carries the real one.
    if (response.status === 409 && body.step) return render(body.step);
    throw Object.assign(new Error(body.error || `${path} failed with ${response.status}`), {
      status: response.status,
    });
  }
  return body;
}

/* The event stream (§9.5's third file). Batched and best-effort: a dropped
   batch must never block a step, so failures are swallowed here rather than
   surfaced. Anything a measure depends on goes through its own route. */
let pending = [];
let flushing = null;

function note(type, fields) {
  pending.push({ type, at: new Date().toISOString(), ...fields });
  if (!flushing) flushing = setTimeout(flush, 1200);
}

async function flush() {
  flushing = null;
  const events = pending;
  pending = [];
  if (!events.length || !state.participant) return;
  try {
    await fetch("/api/events", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ participant: state.participant, events }),
    });
  } catch {
    /* best effort */
  }
}

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "hidden") flush();
});

/* The language toggle, top right. Live until the first clip has played and a
   plain label after that: §8 compares against §3, so a session read half in
   one language and half in the other has moved something the study measures.
   The server enforces it; this only stops the participant asking.

   The locked toggle keeps both buttons on screen. Deleting the alternate one
   made a lock look like a bug, and the reason was set as a title — needing a
   hover, never appearing on touch, announced inconsistently — so in a
   supervised session it became a question for the researcher mid-run. */
function drawLanguages() {
  const host = $("languages");
  if (state.languages.length < 2) {
    host.hidden = true;
    return;
  }
  const copy = state.strings.languages;
  host.hidden = false;
  host.setAttribute("aria-label", copy.label);
  const pair = clear(host.querySelector(".pair"));
  for (const tag of state.languages) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = copy.names[tag] || tag;
    button.lang = tag;
    const at = tag === state.language;
    if (at) {
      button.classList.add("at");
      button.setAttribute("aria-current", "true");
    }
    button.disabled = state.languageLocked;
    if (!state.languageLocked) button.onclick = () => choose(tag);
    pair.append(button);
  }
  const locked = host.querySelector(".locked");
  locked.textContent = copy.locked;
  locked.hidden = !state.languageLocked;
}

async function choose(tag) {
  if (tag === state.language || state.languageLocked) return;
  const result = await api("/api/language", { participant: state.participant, language: tag });
  if (!result || !result.language) return;
  state.language = result.language;
  state.strings = state.chrome[state.language] || state.chrome.en;
  state.languageLocked = Boolean(result.language_locked);
  // The page's lang drives the Korean font stack and line-heights in
  // identity.css, so the document has to carry it rather than the strings
  // alone (K1, K2).
  document.documentElement.lang = state.language;
  syncChrome();
  drawLanguages();
  // Re-render where they are, so the captions and the copy on screen change
  // with the choice rather than at the next step.
  render(state.step);
}

/* §9.1 — the rail. Six named steps, not six empty spans: the names have been
   in copy.py since the first commit and were used only for their count, so
   there was no way — visual or otherwise — to learn which step this was. */
function drawRail() {
  const list = clear($("rail").querySelector("ol"));
  const at = railAt();
  state.steps.forEach((_, index) => {
    const entry = document.createElement("li");
    if (at >= 0 && index < at) entry.className = "done";
    if (index === at) {
      entry.className = "at";
      entry.setAttribute("aria-current", "step");
    }
    const number = document.createElement("span");
    number.className = "n";
    number.textContent = String(index + 1).padStart(2, "0");
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = stepName(index);
    const bar = document.createElement("span");
    bar.className = "bar";
    entry.append(number, name, bar);
    list.append(entry);
  });
}

/* The clip is the stimulus, and a clip that arrives over the network while it
   plays is a different stimulus on a slow link than on a fast one: the browser
   starts on a partial buffer and halts when it runs dry, and onended fires
   the same either way. So the whole file is fetched before Start is enabled
   and played from memory, which makes playback independent of the connection
   the participant happens to be on. A fetch that fails falls back to
   streaming rather than stranding the participant, and says so in the log. */
async function prefetchClip(source, step) {
  const began = performance.now();
  try {
    const response = await fetch(source);
    if (!response.ok) throw new Error(`${response.status}`);
    const blob = await response.blob();
    const ms = Math.round(performance.now() - began);
    note("viewing.prefetch", { step, prefetched: true, bytes: blob.size, ms });
    return { url: URL.createObjectURL(blob), prefetched: true, bytes: blob.size };
  } catch (error) {
    note("viewing.prefetch", { step, prefetched: false, error: error.message });
    return { url: source, prefetched: false, bytes: null };
  }
}

/* §2 and §7 — the viewing. One start button, then fullscreen and no controls. */
async function renderViewing(step) {
  const detail = await api(`/api/step/${step}?participant=${state.participant}`);
  if (!detail) return;
  const strings = state.strings.view;
  const index = step === "view_prepared" ? 0 : 4;
  head("start", stepName(index));
  // The second viewing plays the *other* segment (§1). This screen was
  // otherwise identical to the first viewing's, so nothing on it said so.
  const different = $("start-different");
  different.hidden = step !== "view_regenerated";
  different.textContent = different.hidden ? "" : strings.different;
  $("start-headphones").textContent = strings.headphones;
  $("start-ready").textContent = strings.ready;
  const button = $("start-button");
  // Start is enabled only once the clip is local. The button says so rather
  // than sitting dead: on campus the wait is unnoticeable, off campus it can
  // be seconds, and a participant who reads the instruction line meanwhile
  // is doing what the screen is for.
  button.textContent = strings.preparing;
  button.disabled = true;
  show("screen-start");
  // Warmed during §6 if this is the viewing that follows it, and claimed here
  // so a second viewing can never be handed a URL the first one revoked.
  const warmed = state.warming && state.warming.segment === detail.segment ? state.warming : null;
  state.warming = null;
  const clip = warmed ? await warmed.clip : await prefetchClip(`/media/video/${detail.segment}`, step);
  button.textContent = state.strings.actions.start;
  button.disabled = false;

  button.onclick = async () => {
    button.disabled = true;
    const video = $("video");
    const stage = $("screen-viewing");
    const band = $("cue");
    const cues = detail.captions;
    video.src = clip.url;
    band.textContent = "";
    $("viewing-interrupted").hidden = true;
    show("screen-viewing");
    try {
      await stage.requestFullscreen();
    } catch {
      /* A browser that refuses fullscreen still plays; the clip is the
         stimulus and losing it to a permissions prompt would be worse. */
    }

    let shown = -1;
    const showCue = (at) => {
      if (at === shown) return;
      shown = at;
      band.textContent = at === -1 ? "" : cues[at].text;
      if (at !== -1) note("caption.shown", { step, index: at, text: cues[at].text });
    };

    /* The same cues as a real TextTrack (§2·2). It is hidden rather than
       showing — the band is the caption the study manipulates, and two of them
       on screen would be two stimuli — but a hidden track still fires
       cuechange, which takes cue timing off the ~4Hz timeupdate rate that
       could land a cue up to 250ms after its start_ms and put the same slack
       into caption.shown. timeupdate stays as the backstop until the track has
       proved it parsed. */
    let native = false;
    const track = $("cues");
    const vtt = [
      "WEBVTT",
      "",
      ...cues.flatMap((cue, at) => [
        String(at + 1),
        `${stamp(cue.start_ms)} --> ${stamp(cue.end_ms)}`,
        cue.text,
        "",
      ]),
    ].join("\n");
    if (track.src.startsWith("blob:")) URL.revokeObjectURL(track.src);
    track.src = URL.createObjectURL(new Blob([vtt], { type: "text/vtt" }));
    track.track.mode = "hidden";
    track.onload = () => {
      const list = track.track.cues;
      if (!list || !list.length) return;
      native = true;
      track.track.oncuechange = () => {
        const active = track.track.activeCues;
        if (!active || !active.length) return showCue(-1);
        showCue(Number(active[0].id) - 1);
      };
    };

    video.ontimeupdate = () => {
      if (native) return;
      const ms = video.currentTime * 1000;
      showCue(cues.findIndex((cue) => ms >= cue.start_ms && ms < cue.end_ms));
    };

    const startedAt = new Date().toISOString();
    let ending = false;
    let interruptions = 0;

    /* A halt for want of data is the other way a viewing stops being the
       stimulus, and it leaves no trace of its own: the clip resumes, onended
       fires, the timestamps are a little further apart. `waiting` is the
       browser saying playback has stopped on an empty buffer; `stalled` that
       the fetch behind a streamed source has gone quiet. Neither counts
       before the first frame, which is loading rather than stalling, nor
       while the clip is paused for an interruption, nor across a seek —
       the instrument never seeks, but a seek fires `waiting` too, and a
       count that means "halted mid-play" should not be movable by one. */
    let stalls = 0;
    let playing = false;
    video.onplaying = () => {
      playing = true;
    };
    const onStall = (event) => {
      if (!playing || ending || video.ended || video.paused || video.seeking) return;
      if (event.type === "waiting") stalls += 1;
      note("viewing.stall", {
        step,
        kind: event.type,
        at_ms: Math.round(video.currentTime * 1000),
        count: stalls,
      });
    };
    video.onwaiting = onStall;
    video.onstalled = onStall;

    /* Fullscreen was requested and never watched. Escape is the browser's own
       shortcut and the first thing a nervous participant tries; the clip used
       to keep playing in a 720px column with the band pinned to the viewport,
       then onended fired and /api/viewing recorded a clean viewing. The log
       could not tell that session from a correct one. */
    const onFullscreen = () => {
      if (document.fullscreenElement || ending || video.ended) return;
      interruptions += 1;
      video.pause();
      $("interrupted-heading").textContent = strings.interrupted_heading;
      $("interrupted-body").textContent = strings.interrupted_body;
      const resume = $("interrupted-resume");
      resume.textContent = state.strings.actions.resume;
      resume.onclick = async () => {
        $("viewing-interrupted").hidden = true;
        try {
          await stage.requestFullscreen();
        } catch {
          /* as above: the clip matters more than the chrome */
        }
        video.play().catch(() => {});
      };
      $("viewing-interrupted").hidden = false;
      resume.focus();
      note("viewing.interrupted", { step, at_ms: Math.round(video.currentTime * 1000), count: interruptions });
    };
    document.addEventListener("fullscreenchange", onFullscreen);

    video.onended = async () => {
      ending = true;
      document.removeEventListener("fullscreenchange", onFullscreen);
      track.track.oncuechange = null;
      video.onplaying = video.onwaiting = video.onstalled = null;
      const endedAt = new Date().toISOString();
      if (document.fullscreenElement) await document.exitFullscreen().catch(() => {});
      // Whether the clip was watched as the stimulus it is meant to be. It
      // rides the event stream rather than the viewing payload so the route's
      // contract is unchanged, and flushes before the step turns over.
      note("viewing.integrity", {
        step,
        interruptions,
        stalls,
        prefetched: clip.prefetched,
        bytes: clip.bytes,
      });
      await flush();
      const result = await api("/api/viewing", {
        participant: state.participant,
        step,
        started_at: startedAt,
        ended_at: endedAt,
      });
      // A clip has played; the language is the session's now.
      state.languageLocked = true;
      if (clip.prefetched) URL.revokeObjectURL(clip.url);
      if (result) render(result.step);
    };
    try {
      await video.play();
    } catch (error) {
      document.removeEventListener("fullscreenchange", onFullscreen);
      fail(`the clip would not start: ${error.message}`, `${detail.segment}: ${error.message}`);
    }
  };
}

function stamp(ms) {
  const total = Math.max(0, ms) / 1000;
  const minutes = String(Math.floor(total / 60)).padStart(2, "0");
  const seconds = String(Math.floor(total % 60)).padStart(2, "0");
  return `${minutes}:${seconds}.${String(Math.round(ms % 1000)).padStart(3, "0")}`;
}

/* §3 and §8 — the surveys, built from the blocks the server sends. */
async function renderSurvey(page) {
  const detail = await api(`/api/step/${page}?participant=${state.participant}`);
  if (!detail) return;
  state.entered = new Date().toISOString();
  const answers = new Map();
  const container = clear($("survey-blocks"));
  const copy = state.strings.survey;
  const total = detail.blocks.reduce((sum, block) => sum + block.items.length, 0);
  head("survey", stepName(railAt()));
  $("survey-instruction").textContent =
    detail.blocks.length > 1
      ? fill(copy.instruction_all, { blocks: detail.blocks.length, count: total })
      : fill(copy.instruction, { count: total });
  document.documentElement.style.setProperty("--points", String(detail.scale.points));

  const rows = new Map();
  const counters = new Map();
  const submit = $("survey-submit");
  const first = $("survey-first");
  submit.textContent = state.strings.actions.submit;
  first.textContent = copy.first;

  const update = () => {
    const left = total - answers.size;
    submit.disabled = left > 0;
    $("survey-remaining").textContent = left
      ? fill(copy.remaining, { count: left, total })
      : fill(copy.complete, { total });
    // The count is a control now: it says how many are left and goes to the
    // first of them, instead of leaving the participant to re-scan 22 rows.
    const open = [...rows.entries()].find(([id]) => !answers.has(id));
    first.hidden = !open;
    if (open) first.onclick = () => {
      open[1].scrollIntoView({ block: "center", behavior: "smooth" });
      const input = open[1].querySelector("input");
      if (input) input.focus();
    };
    for (const [id, block] of counters) {
      const done = block.ids.filter((item) => answers.has(item)).length;
      block.node.textContent = fill(copy.block_done, { done, total: block.ids.length });
      void id;
    }
  };

  for (const block of detail.blocks) {
    const section = document.createElement("section");
    section.className = "block";
    // Stated once per block and kept on screen. A block of eight items is
    // about 900px tall, so items 4 to 8 used to be answered with no visible
    // scale meaning at all — the numbers 1 to 7 and nothing else.
    const cap = document.createElement("div");
    cap.className = "cap";
    const titles = document.createElement("div");
    titles.className = "titles";
    const heading = document.createElement("h2");
    heading.textContent = block.title;
    const done = document.createElement("span");
    done.className = "of";
    titles.append(heading, done);
    counters.set(block.id, { node: done, ids: block.items.map((item) => item.id) });

    /* One grid, shared by the anchors and every scale beneath them. The
       anchors were a full-width flex row above a 304px left-aligned scale, so
       "Very much" sat 368px to the right of point 7, above nothing at all. The
       verbal anchors are what turn seven numbers into an interval scale, and a
       participant who reads the high one as belonging to the whitespace may
       anchor differently on §3 than on §8 — the exact comparison the study is
       built on. §9.3 gives two anchors and only two; they go at the ends and
       position does the rest. */
    const anchors = document.createElement("div");
    anchors.className = "anchors";
    const [low, high] = detail.scale.anchors;
    const lowSpan = document.createElement("span");
    lowSpan.className = "low";
    lowSpan.textContent = low;
    const highSpan = document.createElement("span");
    highSpan.className = "high";
    highSpan.textContent = high;
    anchors.append(lowSpan, highSpan);
    cap.append(titles, anchors);
    section.append(cap);

    for (const item of block.items) {
      const row = document.createElement("div");
      row.className = "item";
      row.id = `row-${item.id}`;
      const text = document.createElement("p");
      text.textContent = item.text;
      const group = document.createElement("div");
      group.className = "choices";
      group.setAttribute("role", "radiogroup");
      group.setAttribute("aria-label", item.text);
      for (let value = 1; value <= detail.scale.points; value += 1) {
        const label = document.createElement("label");
        const input = document.createElement("input");
        input.type = "radio";
        input.name = item.id;
        input.value = String(value);
        // The scale's meaning is at the ends of the row, so a cell needs to
        // carry it too or a screen reader hears seven bare numbers.
        input.setAttribute("aria-label", value === 1 ? `${value} — ${low}` : value === detail.scale.points ? `${value} — ${high}` : String(value));
        input.onchange = () => {
          answers.set(item.id, value);
          row.classList.add("answered");
          note("survey.answered", { page, item: item.id, value });
          update();
        };
        const badge = document.createElement("span");
        badge.textContent = String(value);
        label.append(input, badge);
        group.append(label);
      }
      row.append(text, group);
      rows.set(item.id, row);
      section.append(row);
    }
    container.append(section);
  }
  update();
  show("screen-survey");

  submit.onclick = async () => {
    submit.disabled = true;
    try {
      const result = await api("/api/survey", {
        participant: state.participant,
        page,
        responses: Object.fromEntries(answers),
        entered_at: state.entered,
        submitted_at: new Date().toISOString(),
      });
      if (result) render(result.step);
      else submit.disabled = false;
    } catch (error) {
      submit.disabled = false;
      if (error.status) return fail(error.message, error.message);
      $("survey-remaining").textContent = state.strings.network.retry;
    }
  };
}

/* §4 — marks on one moment at a time.

   The picker is text and the picture is a picture. What was there before was a
   scroller in which the marked frame was full-size and its neighbours were
   blurred, which asked the blur to say "do not mark here" and "click me to
   navigate" at once — and put two mechanisms and the participant in contention
   for one scroll position. Nothing below owns a scroll position.

   Coordinates are still normalised to the image, so a resize between marking
   and submitting does not move what was meant, and each mark still carries the
   frame it belongs to: /api/visual and points.py see exactly what they saw. */
async function renderVisual() {
  const detail = await api(`/api/step/visual?participant=${state.participant}`);
  if (!detail) return;
  state.points = [];
  state.frames = detail.frames;
  state.minimumPoints = detail.minimum;
  state.frame = 0;
  const copy = state.strings.visual;
  const plate = $("plate");
  const picker = $("moments");
  picker.setAttribute("aria-label", copy.moments_label);
  picker.style.setProperty("--moments", String(detail.frames.length));
  $("visual-clear").textContent = state.strings.actions.clear;
  $("visual-undo").textContent = state.strings.actions.undo;
  $("visual-next").textContent = state.strings.actions.next;
  $("moment-back").setAttribute("aria-label", copy.back);
  $("moment-on").setAttribute("aria-label", copy.on);

  const image = document.createElement("img");
  image.alt = "";
  image.draggable = false;
  // The crosshair is the keyboard's cursor. It appears when the plate takes
  // focus from the keyboard and never when a pointer is doing the work.
  const cross = document.createElement("i");
  cross.className = "crosshair";
  cross.hidden = true;
  let aim = { x: 0.5, y: 0.5 };

  const seconds = (index) => (detail.frames[index].at_ms / 1000).toFixed(1);

  /* Every moment is fetched when §4 opens, not when it is first chosen. The
     strip is five pictures and the screen's whole job is comparing one against
     another, so fetching on the click charged a wait to every first visit —
     and charged it again on the way back, because the stills were served with
     no cache headers and revalidation returned the whole file. They are held
     in a list so the browser keeps them for the life of the screen. The one on
     screen is asked for first; the other four are explicitly the lower
     priority, so the picture being marked on is never behind the ones that are
     not. */
  const strip = detail.frames.map((frame, index) => {
    const picture = new Image();
    picture.fetchPriority = index === 0 ? "high" : "low";
    picture.src = `/media/frame/${detail.segment}/${frame.index}`;
    return picture;
  });

  const buttons = detail.frames.map((frame, index) => {
    const button = document.createElement("button");
    button.type = "button";
    const at = document.createElement("span");
    at.textContent = fill(copy.moment, { seconds: seconds(index) });
    const tally = document.createElement("span");
    tally.className = "tally";
    button.append(at, tally);
    button.onclick = () => select(index);
    picker.append(button);
    return { button, tally, frame };
  });

  function select(index) {
    if (index < 0 || index >= buttons.length) return;
    const changed = index !== state.frame;
    state.frame = index;
    image.src = strip[index].src;
    $("visual-heading").textContent = fill(copy.heading, { seconds: seconds(index) });
    plate.setAttribute("aria-label", fill(copy.plate, { seconds: seconds(index) }));
    if (changed) note("frame.selected", { frame: index, at_ms: detail.frames[index].at_ms });
    paint();
  }

  const at = (event) => {
    const box = image.getBoundingClientRect();
    return {
      x: Math.min(1, Math.max(0, (event.clientX - box.left) / box.width)),
      y: Math.min(1, Math.max(0, (event.clientY - box.top) / box.height)),
    };
  };

  function place(x, y) {
    const point = { frame: state.frame, x, y };
    state.points.push(point);
    note("point.placed", { index: state.points.length - 1, ...point });
    paint();
  }

  plate.onclick = (event) => {
    // Only the picture takes a mark. A click that started on an existing mark
    // or on its remove control is that mark's business, not a new point.
    if (event.target !== plate && event.target !== image) return;
    const box = image.getBoundingClientRect();
    if (
      event.clientX < box.left ||
      event.clientX > box.right ||
      event.clientY < box.top ||
      event.clientY > box.bottom
    ) {
      return;
    }
    const spot = at(event);
    place(spot.x, spot.y);
  };

  /* The keyboard's path through §4. Arrow keys move a crosshair over the
     picture and Enter places a mark where it stands; the picker is real
     buttons, so choosing a moment needs nothing extra. */
  plate.onkeydown = (event) => {
    const nudge = event.shiftKey ? 0.01 : 0.04;
    const moves = {
      ArrowLeft: [-nudge, 0],
      ArrowRight: [nudge, 0],
      ArrowUp: [0, -nudge],
      ArrowDown: [0, nudge],
    };
    if (moves[event.key]) {
      event.preventDefault();
      cross.hidden = false;
      aim = {
        x: Math.min(1, Math.max(0, aim.x + moves[event.key][0])),
        y: Math.min(1, Math.max(0, aim.y + moves[event.key][1])),
      };
      cross.style.left = `${aim.x * 100}%`;
      cross.style.top = `${aim.y * 100}%`;
      return;
    }
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      cross.hidden = false;
      place(aim.x, aim.y);
    }
  };

  function paint() {
    clear(plate).append(image, cross);
    cross.style.left = `${aim.x * 100}%`;
    cross.style.top = `${aim.y * 100}%`;
    state.points.forEach((point, index) => {
      if (point.frame !== state.frame) return;
      const dot = document.createElement("div");
      dot.className = "point";
      dot.style.left = `${point.x * 100}%`;
      dot.style.top = `${point.y * 100}%`;
      dot.textContent = String(index + 1);
      dot.onpointerdown = (event) => drag(event, index, dot);
      // Removing a mark is its own control. It used to be a press that
      // happened not to move, decided on pointerup against a drag, on a 22px
      // target with no undo — so a one-pixel tremor turned a nudge into a
      // deletion and the only way to inspect a mark was to risk losing it. A
      // deleted mark is a reported perception withdrawn.
      const drop = document.createElement("button");
      drop.type = "button";
      drop.className = "drop";
      drop.textContent = "✕";
      drop.setAttribute("aria-label", `${state.strings.actions.undo} ${index + 1}`);
      // Which shoulder the ✕ sits on: the plate clips, so a mark on the
      // skyline or against the right edge still has a reachable one.
      dot.classList.toggle("at-top", point.y < 0.09);
      dot.classList.toggle("at-right", point.x > 0.94);
      drop.onpointerdown = (event) => event.stopPropagation();
      drop.onclick = (event) => {
        event.stopPropagation();
        const [removed] = state.points.splice(index, 1);
        note("point.removed", { index, ...removed });
        paint();
      };
      dot.append(drop);
      plate.append(dot);
    });

    buttons.forEach(({ tally }, index) => {
      const here = state.points.filter((point) => point.frame === index).length;
      tally.textContent = here ? fill(copy.tally, { count: here }) : copy.tally_none;
      buttons[index].button.classList.toggle("at", index === state.frame);
      buttons[index].button.setAttribute("aria-pressed", String(index === state.frame));
    });

    const count = state.points.length;
    const moments = new Set(state.points.map((point) => point.frame)).size;
    $("visual-count").textContent = count
      ? fill(copy.placed, { count, moments, total: detail.frames.length })
      : copy.placed_none;
    const short = count < state.minimumPoints;
    $("visual-next").disabled = short;
    $("visual-undo").disabled = count === 0;
    $("visual-clear").disabled = count === 0;
    // On its own reserved line. Appending the floor to the instruction and
    // stripping it once met cost the paragraph a sentence — and possibly a
    // line of height — the instant the first mark landed, shifting the picture
    // up under the cursor that had just placed it.
    $("visual-hint").textContent = short
      ? state.minimumPoints === 1
        ? copy.minimum_one
        : fill(copy.minimum, { minimum: state.minimumPoints })
      : copy.drag;
  }

  function drag(event, index, dot) {
    event.preventDefault();
    event.stopPropagation();
    dot.setPointerCapture(event.pointerId);
    dot.classList.add("dragging");
    let moved = false;
    const move = (moveEvent) => {
      moved = true;
      const next = at(moveEvent);
      state.points[index] = { ...state.points[index], ...next };
      dot.style.left = `${next.x * 100}%`;
      dot.style.top = `${next.y * 100}%`;
    };
    const up = () => {
      dot.removeEventListener("pointermove", move);
      dot.classList.remove("dragging");
      if (moved) note("point.moved", { index, ...state.points[index] });
      paint();
    };
    dot.addEventListener("pointermove", move);
    dot.addEventListener("pointerup", up, { once: true });
    dot.addEventListener("pointercancel", up, { once: true });
  }

  $("moment-back").onclick = () => select(state.frame - 1);
  $("moment-on").onclick = () => select(state.frame + 1);

  $("visual-undo").onclick = () => {
    const removed = state.points.pop();
    if (!removed) return;
    note("point.removed", { index: state.points.length, ...removed });
    if (removed.frame !== state.frame) select(removed.frame);
    else paint();
  };

  $("visual-clear").onclick = () => {
    note("points.cleared", { count: state.points.length });
    state.points = [];
    paint();
  };

  $("visual-next").onclick = async () => {
    $("visual-next").disabled = true;
    try {
      const result = await api("/api/visual", { participant: state.participant, points: state.points });
      if (result) render(result.step);
      else paint();
    } catch (error) {
      $("visual-next").disabled = state.points.length < state.minimumPoints;
      if (error.status) return fail(error.message, error.message);
      $("visual-count").textContent = state.strings.network.retry;
    }
  };

  head("visual", "");
  $("visual-instruction").textContent = `${copy.instruction} ${copy.keyboard}`;
  select(0);
  show("screen-visual");
}

/* §5 — heard or did not hear, on each of five fixed sound families.

   It used to be a selection among the sources a clip happened to carry, with a
   waveform lane and a playback control for each. Two things were wrong with
   that. A participant could only ever report a source that was there, so the
   screen could not tell "I heard it" from "I was given the chance to say so",
   and a family nobody put in the clip was unreportable rather than a false
   alarm. And the lanes needed separated stem audio, which does not exist for
   this corpus — every lane read "No sound".

   Five families, asked of everyone, every time. Both answers are explicit:
   a blank is a participant who did not answer, which is not the same as one
   who did not hear, and §6 must not be conditioned on the difference between
   a denial and a shrug. */
async function renderAuditory() {
  const detail = await api(`/api/step/auditory?participant=${state.participant}`);
  if (!detail) return;
  state.segment = detail.segment;
  const strings = state.strings.auditory;
  const answers = new Map();
  head("auditory", stepName(railAt()));
  $("auditory-instruction").textContent = strings.instruction;
  $("lanes-legend").textContent = strings.legend;
  const submit = $("auditory-submit");
  submit.textContent = state.strings.actions.submit;

  const container = $("lanes");
  for (const node of [...container.querySelectorAll(".family")]) node.remove();

  const count = () => {
    const done = answers.size;
    $("auditory-note").textContent = done
      ? fill(strings.answered, { count: done, total: detail.families.length })
      : strings.answered_none;
    submit.disabled = done < detail.families.length;
  };

  for (const family of detail.families) {
    const row = document.createElement("div");
    row.className = "family";
    /* Each row is its own question, and has to say so. The two radios are
       named "Heard" and "Didn't hear" and nothing else, so without this a
       screen reader reads ten controls with five identical pairs of labels and
       never says which family any of them is about — the same defect the lanes
       this replaced were explicitly fixed for. The group carries the family
       name, and the examples ride along as its description. */
    row.setAttribute("role", "radiogroup");
    row.setAttribute("aria-labelledby", `family-${family}-name`);

    const label = document.createElement("div");
    label.className = "label";
    const name = document.createElement("span");
    name.id = `family-${family}-name`;
    name.textContent = strings.families[family] || family;
    const hint = document.createElement("small");
    hint.textContent = strings.examples[family] || "";
    if (hint.textContent) {
      hint.id = `family-${family}-examples`;
      row.setAttribute("aria-describedby", hint.id);
    }
    label.append(name, hint);

    /* Radios rather than a checkbox: a checkbox has one explicit state and
       one that means both "no" and "not yet", and this screen has to keep
       those apart. Grouped by family so the two answers are one question. */
    const choice = document.createElement("div");
    choice.className = "choice";
    for (const [value, text] of [[true, strings.heard], [false, strings.not_heard]]) {
      const option = document.createElement("label");
      const input = document.createElement("input");
      input.type = "radio";
      input.name = `family-${family}`;
      input.onchange = () => {
        answers.set(family, value);
        note("family.answered", { family, heard: value });
        count();
      };
      const caption = document.createElement("span");
      caption.textContent = text;
      option.append(input, caption);
      choice.append(option);
    }

    row.append(label, choice);
    container.append(row);
  }

  count();

  submit.onclick = async () => {
    submit.disabled = true;
    const heard = {};
    for (const family of detail.families) heard[family] = answers.get(family) === true;
    try {
      await flush();
      const result = await api("/api/auditory", { participant: state.participant, heard });
      if (result) render(result.step);
      else count();
    } catch (error) {
      submit.disabled = answers.size < detail.families.length;
      if (error.status) return fail(error.message, error.message);
      $("auditory-note").textContent = state.strings.network.retry;
    }
  };

  show("screen-auditory");
}

/* §6 — the wait. One POST, which is idempotent server-side, so a reload here
   returns the track already written rather than starting a second one. */
async function renderWaiting() {
  const copy = state.strings.waiting;
  head("waiting", copy.heading);
  $("waiting-eyebrow").textContent = copy.eyebrow;
  $("waiting-explains").textContent = copy.explains;
  $("waiting-stay").textContent = copy.stay;
  $("waiting-ahead-heading").textContent = copy.ahead_heading;
  $("waiting-ahead-body").textContent = copy.ahead_body;
  $("waiting-ceiling").textContent = fill(copy.ceiling, {
    seconds: Math.round(state.ceilingMs / 1000),
  });
  show("screen-waiting");

  /* The comment defending the indeterminate bar is right — nothing on this
     page can predict the model — but indeterminate had been implemented as
     silent. A number that moves is the proof that the page is alive, which is
     the question a waiting participant is actually asking; the sentence above
     it is what a screen reader gets, since role="progressbar" with no
     aria-valuenow announced once and then went quiet, and nothing said
     anything at all when the step finished. The bar's own reduced-motion
     treatment is the stylesheet's business, not this file's. */
  const startedAt = Date.now();
  $("waiting-status").textContent = copy.status;
  const tick = () => {
    $("waiting-elapsed").textContent = fill(copy.elapsed, {
      seconds: Math.floor((Date.now() - startedAt) / 1000),
    });
  };
  tick();
  const ticking = window.setInterval(tick, 1000);

  /* One cell per cue in the track, drawn from the slot count the study is
     calibrated to (§9.4) so the bar does not grow as it goes. The cell being
     written is the one that moves. */
  const bar = clear($("waiting-bar"));
  const cells = [];
  for (let slot = 0; slot < state.cueSlots; slot += 1) {
    const cell = document.createElement("i");
    bar.append(cell);
    cells.push(cell);
  }

  let seen = -1;
  const paintSlots = (done, total) => {
    if (done === seen) return;
    seen = done;
    cells.forEach((cell, index) => {
      cell.classList.toggle("done", index < done);
      cell.classList.toggle("at", index === done && done < total);
    });
    if (done > 0) $("waiting-status").textContent = fill(copy.status_at, { done, total });
    note("regeneration.slot", { done, total });
  };
  paintSlots(0, state.cueSlots);

  /* Polled rather than pushed. /api/regenerate is one blocking call — it is
     the model — and it is a sync route, so it runs in the threadpool and this
     GET is answered while it is still in flight. Best-effort like the event
     stream: a poll that fails leaves the bar where it was, because a progress
     display may not be the thing that ends a session. */
  /* §6 is the one wait the session already has, and the clip that follows it
     is 5–7 MB the participant would otherwise wait for a second time, from a
     standing start, on the next screen. So it is fetched here, behind a bar
     they are already watching. Nothing about the viewing changes: it is the
     same whole-file-then-play that §3 does, only started earlier, and if this
     has not finished by the time the screen opens the viewing awaits it rather
     than asking again. Once only — the poll runs every 700 ms. */
  const warm = (segment) => {
    if (state.warming || !segment) return;
    state.warming = { segment, clip: prefetchClip(`/media/video/${segment}`, "view_regenerated") };
  };

  const poll = async () => {
    try {
      const response = await fetch(`/api/regenerate/progress?participant=${state.participant}`);
      if (!response.ok) return;
      const at = await response.json();
      warm(at.next_segment);
      if (at.writing) paintSlots(at.done, at.total);
    } catch {
      /* the next poll will do */
    }
  };
  const polling = window.setInterval(poll, 700);
  poll();

  try {
    const result = await api("/api/regenerate", { participant: state.participant });
    if (!result) return;
    note("regeneration.finished", { fallback: result.fallback, cached: result.cached });
    paintSlots(state.cueSlots, state.cueSlots);
    $("waiting-status").textContent = copy.status_done;
    const next = await api(`/api/state?participant=${state.participant}`);
    if (next) render(next.step);
  } finally {
    window.clearInterval(ticking);
    window.clearInterval(polling);
  }
}

function renderDone() {
  const copy = state.strings.done;
  head("done", copy.heading);
  $("done-body").textContent = copy.body;
  // After a screen that warned them leaving would end the session, the
  // participant's last impression was an assurance that never came.
  $("done-receipt").textContent = fill(copy.receipt, {
    participant: state.participant,
    at: new Date().toISOString().replace("T", " ").slice(0, 19),
  });
  $("done-for").textContent = copy.for_researcher;
  // Off the study machine the server keeps the log to itself, and a button
  // that led to that refusal would replace this screen with an error body.
  $("done-for").parentElement.hidden = !state.download;
  const button = $("done-download");
  button.textContent = state.strings.actions.download;
  // Navigate rather than open a tab: the server sends the log as an
  // attachment, so the browser saves it and leaves the participant on this
  // screen. A new tab in a fullscreen kiosk is one the participant cannot
  // close.
  button.onclick = () => {
    window.location.href = `/api/log?participant=${encodeURIComponent(state.participant)}`;
  };
  show("screen-done");
  if (state.viewingEnabled) {
    const continuation = state.strings.continuation;
    head("done", continuation.heading);
    $("done-body").textContent = continuation.body;
    const next = $("done-continue"), status = $("done-continuation-status");
    next.hidden = false; status.hidden = false;
    next.textContent = continuation.next;
    if (!state.viewingToken) {
      next.hidden = true; status.textContent = continuation.recovery;
      return;
    }
    const proceed = async () => {
      next.disabled = true; status.textContent = continuation.opening;
      try {
        const response = await fetch("/api/continuation", {
          method: "POST", headers: {"content-type": "application/json", "x-study-request": "1"},
          body: JSON.stringify({participant: state.participant, token: state.viewingToken}),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error);
        window.location.assign(result.url);
      } catch {
        status.textContent = continuation.retry; next.disabled = false;
      }
    };
    next.onclick = proceed;
    proceed();
  }
}

async function render(step) {
  state.step = step;
  syncChrome();
  drawRail();
  drawLanguages();
  note("step.entered", { step });
  try {
    if (step === "view_prepared" || step === "view_regenerated") return await renderViewing(step);
    if (step === "art" || step === "survey") return await renderSurvey(step);
    if (step === "visual") return await renderVisual();
    if (step === "auditory") return await renderAuditory();
    if (step === "regenerating") return await renderWaiting();
    if (step === "done") return renderDone();
    fail(`unknown step ${step}`, `unknown step ${step}`);
  } catch (error) {
    fail(error.message, error.message);
  }
}

async function boot() {
  try {
    /* The copy and the enrolment are asked for at once. Neither needs anything
       from the other — the enrolment carries the identifier and the code from
       the URL, the copy is the same for everyone — and asking in turn spent two
       round trips on the landing screen where one does. On the published link
       a round trip is not free, and this is the one place the participant is
       looking at nothing at all. */
    const recovery = new URLSearchParams(window.location.hash.slice(1));
    const recoveredParticipant = recovery.get("participant"), recoveredToken = recovery.get("viewing_token");
    const stored = recoveredParticipant && recoveredToken ? recoveredParticipant : window.sessionStorage.getItem("regen.participant");
    const code = new URLSearchParams(window.location.search).get("code");
    const enrolment = stored ? { participant: stored, viewing_token: recoveredToken || window.sessionStorage.getItem("regen.viewing-token") } : {};
    if (code) enrolment.code = code;
    const asked = fetch("/api/session", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(enrolment),
    })
      .then((response) => response.json())
      // A request in flight before anything awaits it would otherwise reject
      // into nobody's hands. The failure becomes the shape the screen below
      // already reads, and is reported there rather than in the console.
      .catch((error) => ({ error: error.message }));
    const meta = await fetch("/api/strings").then((response) => response.json());
    state.chrome = meta.strings;
    // English until the enrolment says which language this session reads. The
    // screens reachable before that answer — the closed screen for a stale
    // link, the error screen for an enrolment that failed — are read by
    // someone who has no language on record, and both are drawn from
    // `state.strings`.
    state.strings = state.chrome.en || Object.values(state.chrome)[0];
    state.scale = meta.scale;
    state.minimumPoints = meta.minimum_points;
    state.ceilingMs = meta.latency_ceiling_ms || state.ceilingMs;
    state.cueSlots = meta.cue_slots || state.cueSlots;
    state.steps = meta.steps;
    state.languages = meta.languages || [];
    syncChrome();
    document.documentElement.style.setProperty("--points", String(meta.scale.points));
    // The identifier survives a reload; a fresh tab with none enrols anew (§1).
    // The study link may carry an access code; a published instrument refuses
    // new enrolments without it. It rides only the enrolment, sent above.
    const session = await asked;
    if (session.closed) return closed();
    if (session.error) return fail(session.error, session.error);
    state.participant = session.participant;
    state.viewingEnabled = Boolean(session.viewing_enabled);
    state.viewingToken = session.viewing_token || (stored ? window.sessionStorage.getItem("regen.viewing-token") : null);
    if (session.viewing_token) window.sessionStorage.setItem("regen.viewing-token", session.viewing_token);
    state.download = session.download !== false;
    state.language = session.language || state.languages[0] || "en";
    state.strings = state.chrome[state.language] || state.chrome.en;
    state.languageLocked = Boolean(session.language_locked);
    document.documentElement.lang = state.language;
    syncChrome();
    window.sessionStorage.setItem("regen.participant", session.participant);
    if (recoveredParticipant && recoveredToken) history.replaceState(null, "", window.location.pathname + window.location.search);
    await render(session.step);
  } catch (error) {
    fail(error.message, error.message);
  }
}

boot();
