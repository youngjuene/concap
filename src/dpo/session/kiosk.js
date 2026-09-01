/* kiosk.js — the caption session's participant screens.

   Why one file, no build step: the kiosk runs on a tablet from a local
   server; a script tag that fails to load is the only failure mode we can
   afford. The order below is the order the spec argues in: the copy (Table 6)
   is fixed first, then the resumable state and the log, then the skeleton
   math the browser owns (selection carried by balance, freshness by
   comparison), then a renderer per screen (Table 4), then the gestures
   (Table 5), then boot. Nothing acts on the caption except a gesture on the
   skeleton, Show caption, Keep this caption, and Revise this caption
   (spec Table 10, "one surface"). */
"use strict";

/* ---- STRINGS (spec Table 6, verbatim, defined once) ---------------------- */

const STRINGS = {
  appTitle: "Caption session",
  // Patterns: Table 6 gives "Clip 2 of 6"; the shot label uses the same
  // pattern (build contract 9).
  clipCount: "Clip {n} of {m}",
  shotLabel: "Shot {n} of {m}",
  lanes: ["Automatic", "Yours"],
  firstViewing: "First viewing",
  secondViewing: "Second viewing · your captions · revise any shot as it plays",
  attention: "Where was your attention?",
  attentionPoles: ["Mostly the image", "Mostly the sound"],
  listingHeading: "What sounds did you notice?",
  listingField: "Type a sound and press enter",
  listingConfirm: "Nothing else you noticed?",
  nothingElse: "Nothing else",
  addMore: "Add more",
  levels: { itemized: "Itemized", grouped: "Grouped", scene: "Scene", atmospheric: "Atmospheric" },
  roleHeads: ["Underneath", "Stands out", "Only here"],
  poles: { audio: ["Eye", "Ear"], visual: ["Near", "Far"] },
  stripEmpty: "Adjust, then show the caption",
  stripStale: "Written before your last change",
  stripAuditioning: "If only this were mentioned",
  criterion: "Is this the caption you'd want for this moment?",
  openItem: "What makes this place sound like itself?",
  begin: "Begin",
  start: "Start",
  doneListing: "Done listing",
  showCaption: "Show caption",
  keepCaption: "Keep this caption",
  reviseCaption: "Revise this caption",
  finishViewing: "Finish viewing",
  continue: "Continue",
  submit: "Submit",
  downloadLog: "Download session log",
  replayShot: "Replay shot",
  pauseLoop: "Pause loop",
  writing: "Writing…",
  probe: "Across the clips, what, if anything, would you still change about the captions that the controls didn't let you change?",
  error: "Caption request failed. Check the connection and try again.",
  // Not in Table 6 — build contract 9 fixes this one line for a missing id.
  participantMissing: "Open this page with a participant identifier.",
  // Not in Table 6 — instruction card copy and the summary labels are
  // placeholders in the interface's voice (spec 5: all copy is a placeholder).
  intro: {
    heading: "Caption session",
    body: [
      "You will watch six short street clips. After each one you list the sounds you noticed, shape the clip's captions shot by shot, and watch it again with your captions on it.",
      "There are no right answers. Work at your own pace.",
    ],
  },
  authorIntro: {
    heading: "Shaping the caption",
    body: [
      "Each shot carries a list of the sounds its caption could mention. Tap a row to leave a sound out or bring it back. Tap a column to lean the list toward what is seen or what is heard. Drag the handle to fold the list into fewer, broader words.",
      "Show the caption to read what the list writes, and keep it when it is the one you want for that moment.",
    ],
  },
  controlIntro: {
    heading: "Describing the shot",
    body: [
      "Each shot carries a list of the things in view its description could mention. Tap a row to leave one out or bring it back. Tap a column to lean the list toward what is near or what is far. Drag the handle to fold the list into fewer, broader words.",
      "Show the caption to read what the list writes, and keep it when it is the one you want for that moment.",
    ],
  },
  summary: { clips: "Clips shaped", kept: "Captions kept", revisions: "Revisions" },
};

const LEVELS = ["itemized", "grouped", "scene", "atmospheric"];
const HOLD_MS = 400; // spec Table 5: hold >= 400 ms auditions
const FOLD_STEP_PX = 40; // each 40 px of drag or pinch = one level
const MOTION_MS = 160; // spec-identity.md Motion: short linear motion
const AUTOSAVE_MS = 250; // build contract 9: debounce
const INVENTORY_RETRY_MS = 2000; // a resume whose inventory fetch failed asks again
const SNAPSHOT_SCHEMA = "dpo.caption-session-snapshot/v1";

function fill(pattern, n, m) {
  return pattern.replace("{n}", String(n)).replace("{m}", String(m));
}

/* ---- State and snapshot (build contract 6) ------------------------------- */

const state = {
  participant: null,
  session: null,
  snapshot: null,
  inventories: {}, // clip_id -> inventory; memory only, refetched on resume
  shaping: null, // the skeleton's working state for the open shot
  viewing: null, // {which, ended, listing} for the two viewings
  listing: null, // {entries, confirming, error} for the listing screen
  revising: false, // Revise is between its press and the open skeleton (a latch)
  reducedMotion: false,
  raf: 0,
};

function newSnapshot(participant, sessionId) {
  return {
    schema: SNAPSHOT_SCHEMA,
    participant,
    session_id: sessionId,
    began_at: new Date().toISOString(),
    screen: "intro",
    clip_index: 0,
    shot_index: 0,
    seen_author_intro: false,
    seen_control_intro: false,
    listings: {},
    attention: {},
    kept: {},
    revisions: {},
    measures: {},
    probe: "",
    done: false,
    updated_at: new Date().toISOString(),
  };
}

function currentClip() {
  return state.session.clips[state.snapshot.clip_index] || null;
}
function currentShotMeta() {
  const clip = currentClip();
  return clip ? clip.shots[state.snapshot.shot_index] || null : null;
}
function keptFor(clipId, shotId) {
  const clip = state.snapshot.kept[clipId];
  return clip ? clip[shotId] || null : null;
}
function elapsedMs() {
  return Math.max(0, Math.round(Date.now() - Date.parse(state.snapshot.began_at)));
}

/* ---- API: fetch helpers and the event queue (build contract 7, 9) -------- */

const api = {
  queue: [],
  timer: 0,
  inflight: Promise.resolve(true),

  // Every emitted event is queued with the current snapshot and POSTed after a
  // 250 ms quiet, so a crash loses at most a quarter second of gestures.
  emit(type, payload) {
    const event = Object.assign({ t: elapsedMs(), type }, payload || {});
    api.queue.push(event);
    api.save();
  },

  // A snapshot change with no event of its own (text being typed) rides the
  // same debounced POST, so a crash mid-typing keeps the text (Table 4
  // "filled" is a state to resume into, not only to submit from).
  save() {
    state.snapshot.updated_at = new Date().toISOString();
    clearTimeout(api.timer);
    api.timer = setTimeout(() => api.flush(false), AUTOSAVE_MS);
  },

  // Serialised: one POST at a time, in order, so the log stays in gesture
  // order. Resolves true when the server has appended the events.
  flush(keepalive) {
    clearTimeout(api.timer);
    api.inflight = api.inflight.then(() => api.send(keepalive));
    return api.inflight;
  },

  async send(keepalive) {
    const events = api.queue.splice(0, api.queue.length);
    const body = JSON.stringify({ participant: state.participant, events, snapshot: state.snapshot });
    try {
      const response = await fetch("/api/events", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
        keepalive: !!keepalive,
      });
      if (!response.ok) throw new Error(String(response.status));
      return true;
    } catch (_error) {
      // Put them back at the front: the next flush retries in order.
      api.queue.unshift(...events);
      return false;
    }
  },

  async getJSON(url) {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) throw new Error(String(response.status));
    return response.json();
  },

  async postJSON(url, payload) {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(String(response.status));
    return response.json();
  },

  session() {
    return api.getJSON("/api/session");
  },
  state() {
    return api.getJSON("/api/state?participant=" + encodeURIComponent(state.participant));
  },
  // Gated by the server (spec 2 last paragraph): 403 until listing.submit is
  // in the log, which is why callers flush the log before asking.
  inventory(clipId) {
    return api.getJSON(
      "/api/inventory/" + encodeURIComponent(clipId) + "?participant=" + encodeURIComponent(state.participant),
    );
  },
  caption(clipId, shotId, settings) {
    return api.postJSON("/api/caption", {
      participant: state.participant,
      clip_id: clipId,
      shot_id: shotId,
      settings,
    });
  },
};

/* ---- Pure helpers: settings key, freshness, moved rows, orderings -------- */

// Byte for byte the server's key (build contract 3), so freshness is the same
// comparison on both sides.
function settingsKey(level, admitted, order) {
  if (level === "itemized") return "itemized|" + order.join(",");
  if (level === "grouped") return "grouped|" + order.join(",") + "|" + admitted.slice().sort().join(",");
  return level;
}

function subsetKey(admitted) {
  return admitted.slice().sort().join("+");
}

// spec 4.2: "carry the selection to the nearest ordering rather than
// resetting it" — the ordering whose span contains the kept balance.
function resolveOrdering(orderings, balance) {
  for (let i = 0; i < orderings.length; i += 1) {
    const span = orderings[i].span;
    if (balance >= span[0] && balance <= span[1]) return i;
  }
  return Math.max(0, orderings.length - 1);
}

function orderingsFor(s) {
  const table = s.shot.orderings || {};
  if (s.level === "itemized") return (table.itemized || {})[subsetKey(s.admitted)] || [];
  if (s.level === "grouped") return (table.grouped || {})[subsetKey(s.admitted)] || [];
  return [];
}

// At grouped the members follow the itemized ranking at the midpoint of the
// selected head ordering's span — the same balance the server writes the
// caption from (skeleton.balance_of) — so the rows and the sentence agree.
function effectiveBalance(s) {
  if (s.level !== "grouped") return s.balance;
  const list = orderingsFor(s);
  if (!list.length) return s.balance;
  const span = list[resolveOrdering(list, s.balance)].span;
  return (span[0] + span[1]) / 2;
}

function itemizedOrder(s) {
  const list = ((s.shot.orderings || {}).itemized || {})[subsetKey(s.admitted)] || [];
  return list.length ? list[resolveOrdering(list, effectiveBalance(s))].order : [];
}

function currentOrder(s) {
  const list = orderingsFor(s);
  return list.length ? list[resolveOrdering(list, s.balance)].order : [];
}

function sourceById(s, id) {
  return s.shot.sources.find((source) => source.id === id) || null;
}

function membersOf(s, roleId) {
  // Members sit under their head in the head's itemized order (contract 3):
  // the current itemized ranking restricted to this role.
  return itemizedOrder(s).filter((id) => sourceById(s, id).role === roleId);
}

function roleIds(s) {
  return Object.keys(s.roles);
}

function settingsOf(s) {
  return { level: s.level, admitted: s.admitted.slice().sort(), order: currentOrder(s) };
}

function currentKey(s) {
  const st = settingsOf(s);
  return settingsKey(st.level, st.admitted, st.order);
}

// The visible admitted rows, top to bottom, as ids: what a caption was
// "written for" and what the moved marks compare against (spec 4.4).
function rowsOf(s) {
  if (s.level === "itemized") return currentOrder(s).slice();
  if (s.level === "grouped") {
    const rows = [];
    currentOrder(s).forEach((roleId) => {
      rows.push("role:" + roleId);
      membersOf(s, roleId).forEach((id) => rows.push(id));
    });
    return rows;
  }
  return [s.level];
}

// spec 4.4: freshness is a comparison, not a flag. The audition never
// enables Keep (spec 4.1).
function isFresh() {
  const s = state.shaping;
  return !!s && !!s.strip && !s.audition && s.strip.key === currentKey(s);
}

// Rows whose index differs between two row lists — the stale marking and the
// reduced-motion marking use the same comparison (spec-identity.md Motion).
function movedRows(before, after) {
  const moved = new Set();
  after.forEach((id, index) => {
    if (before.indexOf(id) !== index) moved.add(id);
  });
  before.forEach((id) => {
    if (after.indexOf(id) < 0) moved.add(id);
  });
  return moved;
}

function shotAt(clip, ms) {
  const shots = clip.shots;
  for (let i = 0; i < shots.length; i += 1) {
    if (ms >= shots[i].start_ms && ms < shots[i].end_ms) return i;
  }
  return ms >= shots[shots.length - 1].end_ms ? shots.length - 1 : 0;
}

/* ---- DOM helpers --------------------------------------------------------- */

function $(id) {
  return document.getElementById(id);
}

function el(tag, attrs, children) {
  const node = document.createElement(tag);
  Object.entries(attrs || {}).forEach(([name, value]) => {
    if (value === null || value === undefined || value === false) return;
    if (name === "text") node.textContent = value;
    else if (name === "class") node.className = value;
    else if (name.startsWith("on")) node.addEventListener(name.slice(2), value);
    else if (value === true) node.setAttribute(name, "");
    else node.setAttribute(name, String(value));
  });
  (children || []).forEach((child) => {
    if (child) node.appendChild(child);
  });
  return node;
}

function button(label, attrs) {
  return el("button", Object.assign({ type: "button", class: "button", text: label }, attrs || {}));
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
  return node;
}

/* ---- Screens: entering, the frame regions, the timeline ------------------ */

const STAGE_SCREENS = ["watch1", "shape", "watch2"];
const CLIP_SCREENS = ["watch1", "list", "author_intro", "control_intro", "shape", "watch2", "measures"];

function enter(screen) {
  state.snapshot.screen = screen;
  const clip = currentClip();
  const payload = { screen };
  if (clip && CLIP_SCREENS.indexOf(screen) >= 0) payload.clip_id = clip.clip_id;
  if (screen === "shape" && currentShotMeta()) payload.shot_id = currentShotMeta().shot_id;
  api.emit("screen.enter", payload);
  render();
}

function render() {
  const screen = state.snapshot.screen;
  stopLoop();
  // Each screen starts from its own beginning (build contract 9 resume): the
  // per-screen working state is not carried across an enter.
  state.shaping = null;
  state.viewing = null;
  state.revising = false;
  const stage = STAGE_SCREENS.indexOf(screen) >= 0;
  // A single-column screen carries nothing of the clip (spec 4.5): the stage
  // is hidden, so its footage must not keep sounding underneath.
  if (!stage) $("video").pause();
  $("stage-column").hidden = !stage;
  $("main").classList.toggle("single", !stage);
  $("timeline").hidden = !(currentClip() && CLIP_SCREENS.indexOf(screen) >= 0);
  $("actions").hidden = true;
  clear($("actions"));
  clear($("working"));
  if (currentClip()) renderTimeline();
  const renderers = {
    intro: renderIntro,
    watch1: renderWatchFirst,
    list: renderList,
    author_intro: renderAuthorIntro,
    control_intro: renderControlIntro,
    shape: renderShape,
    watch2: renderWatchSecond,
    measures: renderMeasures,
    probe: renderProbe,
    done: renderDone,
  };
  renderers[screen]();
}

// spec 3.1: the timeline belongs to the current clip; shots as segments
// proportional to duration; a display, never a navigation.
function renderTimeline() {
  const clip = currentClip();
  const total = clip.shots[clip.shots.length - 1].end_ms;
  $("clip-count").textContent = fill(STRINGS.clipCount, state.snapshot.clip_index + 1, state.session.clips.length);
  $("lane-upper-label").textContent = STRINGS.lanes[0];
  $("lane-lower-label").textContent = STRINGS.lanes[1];
  const upper = clear($("lane-upper"));
  const lower = clear($("lane-lower"));
  clip.shots.forEach((shot, index) => {
    const grow = String((shot.end_ms - shot.start_ms) / total);
    upper.appendChild(el("span", { class: "segment", "data-index": index, style: "flex-grow:" + grow }));
    lower.appendChild(el("span", { class: "segment", "data-index": index, style: "flex-grow:" + grow }));
  });
  paintLanes(0);
}

function paintLanes(playedMs) {
  const clip = currentClip();
  if (!clip) return;
  const screen = state.snapshot.screen;
  const upper = $("lane-upper").children;
  const lower = $("lane-lower").children;
  clip.shots.forEach((shot, index) => {
    // The upper lane fills with the automatic captions during the first
    // viewing when the clip carries them (spec 3.1), and stays filled after.
    const firstDone = screen !== "watch1" || (state.viewing && state.viewing.ended);
    const automatic = clip.first_viewing_captions && (firstDone || playedMs >= shot.end_ms);
    upper[index].classList.toggle("filled", !!automatic);
    const kept = !!keptFor(clip.clip_id, shot.shot_id);
    const shaping = state.shaping && state.shaping.shot.shot_id === shot.shot_id;
    // mark only on the segment being shaped or revised (spec-identity.md Palette).
    lower[index].classList.toggle("current", !!shaping);
    lower[index].classList.toggle("filled", kept && !shaping);
  });
}

function loadClipMedia(clip) {
  const video = $("video");
  const src = "/media/" + encodeURIComponent(clip.clip_id);
  if (video.getAttribute("data-clip") !== clip.clip_id) {
    video.setAttribute("data-clip", clip.clip_id);
    video.src = src;
    video.load();
  }
}

function setShotLabel(index) {
  const clip = currentClip();
  $("shot-label").textContent = fill(STRINGS.shotLabel, index + 1, clip.shots.length);
}

function card(heading, body, action) {
  const node = el("div", { class: "card" }, [el("h1", { class: "heading", text: heading })]);
  body.forEach((paragraph) => node.appendChild(el("p", { class: "body", text: paragraph })));
  node.appendChild(action);
  return node;
}

/* ---- intro / author intro / control intro (spec Table 4) ----------------- */

function renderIntro() {
  const begin = button(STRINGS.begin, { class: "button primary", onclick: onBegin });
  $("working").appendChild(card(STRINGS.intro.heading, STRINGS.intro.body, begin));
}

function onBegin() {
  // Kiosk (spec 7): fullscreen from the click handler, rejection ignored.
  const root = document.documentElement;
  if (root.requestFullscreen) root.requestFullscreen().catch(() => undefined);
  api.emit("session.begin", {
    participant: state.participant,
    session_id: state.session.session_id,
    reduced_motion: state.reducedMotion,
    viewport: [window.innerWidth, window.innerHeight],
  });
  startClip();
}

function startClip() {
  state.snapshot.shot_index = 0;
  if (!currentClip()) {
    enter("probe");
    return;
  }
  enter("watch1");
}

function renderAuthorIntro() {
  const start = button(STRINGS.start, {
    class: "button primary",
    onclick: () => {
      state.snapshot.seen_author_intro = true;
      openShaping(0);
    },
  });
  $("working").appendChild(card(STRINGS.authorIntro.heading, STRINGS.authorIntro.body, start));
}

function renderControlIntro() {
  const start = button(STRINGS.start, {
    class: "button primary",
    onclick: () => {
      state.snapshot.seen_control_intro = true;
      openShaping(0);
    },
  });
  $("working").appendChild(card(STRINGS.controlIntro.heading, STRINGS.controlIntro.body, start));
}

/* ---- The viewings (spec 3.1, 4.4 placed states, Table 4) ----------------- */

function renderWatchFirst() {
  renderViewing("first");
}
function renderWatchSecond() {
  renderViewing("second");
}

// The frame is a target only during the viewings (tap toggles playback), so
// it is focusable only then; while a shot is being shaped it is footage.
function stageFocusable(on) {
  if (on) $("frame").setAttribute("tabindex", "0");
  else $("frame").removeAttribute("tabindex");
}

function renderViewing(which) {
  const clip = currentClip();
  state.viewing = { which, ended: false, playthrough: false };
  stageFocusable(true);
  loadClipMedia(clip);
  const capobj = $("caption");
  // Placed for the viewing: the object sits in the frame and shows whatever
  // the current shot carries, or nothing (spec 4.4 placed states). The strip
  // beneath stays reserved and empty.
  $("frame").appendChild(capobj);
  capobj.className = "caption-object placed";
  capobj.hidden = true;
  $("caption-eyebrow").textContent = "";
  $("caption-prose").textContent = "";
  renderViewingColumn();
  const video = $("video");
  video.currentTime = 0;
  startLoop(viewingTick);
  // Without a user activation (a resume) the browser may refuse to play
  // unmuted; the frame keeps focus so the first tap or Enter starts the
  // viewing through togglePlayback, which logs it (docs/v1-session/runbook.md
  // names the kiosk flag that lifts the policy).
  video.play().catch(() => $("frame").focus({ preventScroll: true }));
}

function renderViewingColumn() {
  const clip = currentClip();
  const v = state.viewing;
  const working = clear($("working"));
  const stack = el("div", { class: "stack" });
  stack.appendChild(
    el("span", { class: "eyebrow instruction", text: v.which === "first" ? STRINGS.firstViewing : STRINGS.secondViewing }),
  );
  if (v.ended) stack.appendChild(attentionItem(clip, v.which));
  if (v.error) stack.appendChild(el("p", { class: "body", text: v.error }));
  if (v.which === "first") {
    const answered = attentionAnswered(clip, "first");
    stack.appendChild(
      button(STRINGS.continue, {
        class: "button primary",
        disabled: !(v.ended && answered),
        onclick: () => enter("list"),
      }),
    );
  } else {
    // Revise pauses playback, drops the caption into the strip, and opens the
    // skeleton for the current shot (spec 3.3, Table 5).
    stack.appendChild(button(STRINGS.reviseCaption, { onclick: onRevise }));
    const answered = attentionAnswered(clip, "second");
    stack.appendChild(
      button(STRINGS.finishViewing, {
        class: "button primary",
        disabled: !(v.playthrough && answered),
        onclick: () => enter("measures"),
      }),
    );
  }
  working.appendChild(stack);
}

function attentionAnswered(clip, which) {
  const record = state.snapshot.attention[clip.clip_id];
  return !!(record && record[which]);
}

// The attention item alone after each viewing: seven boxes from Mostly the
// image to Mostly the sound (spec 4.7).
function attentionItem(clip, which) {
  const item = state.session.measures.items.find((entry) => entry.id === "attention");
  const boxes = item ? item.boxes : 7;
  const record = state.snapshot.attention[clip.clip_id] || {};
  const row = measureRow(STRINGS.attention, boxes, STRINGS.attentionPoles, record[which] || 0, (value) => {
    state.snapshot.attention[clip.clip_id] = Object.assign({}, record, { [which]: value });
    api.emit("attention.answer", { viewing: which, clip_id: clip.clip_id, value });
    renderViewingColumn();
  });
  row.classList.add("stacked"); // beside the stage the scale sits beneath the text
  return row;
}

function viewingTick() {
  const clip = currentClip();
  const video = $("video");
  const ms = Math.round(video.currentTime * 1000);
  const index = shotAt(clip, ms);
  setShotLabel(index);
  const shot = clip.shots[index];
  let text = null;
  if (state.viewing.which === "first" && clip.first_viewing_captions) text = shot.automatic_caption;
  if (state.viewing.which === "second") {
    const kept = keptFor(clip.clip_id, shot.shot_id);
    text = kept ? kept.caption : null;
  }
  showPlaced(text);
  paintLanes(ms);
}

function showPlaced(text) {
  const capobj = $("caption");
  if (capobj.parentNode !== $("frame")) return;
  capobj.hidden = !text;
  if (text && $("caption-prose").textContent !== text) $("caption-prose").textContent = text;
}

function onVideoEnded() {
  const clip = currentClip();
  if (!clip) return;
  if (state.shaping) {
    // Shaping loops the shot; the clip's own end is just another wrap. This
    // is keyed on the shaping state, not the screen: while revising (spec
    // 3.3) the screen is still watch2, and a loop on the clip's last shot
    // must wrap like any other rather than end the viewing.
    replayShot(false);
    return;
  }
  if (!state.viewing) return;
  api.emit("viewing.ended", { viewing: state.viewing.which, clip_id: clip.clip_id, at_ms: Math.round($("video").currentTime * 1000) });
  state.viewing.ended = true;
  state.viewing.playthrough = true;
  paintLanes(clip.shots[clip.shots.length - 1].end_ms);
  renderViewingColumn();
}

function togglePlayback() {
  const video = $("video");
  const clip = currentClip();
  if (!clip || !state.viewing || state.revising) return;
  const at = Math.round(video.currentTime * 1000);
  if (video.paused) {
    video.play().catch(() => undefined);
    api.emit("viewing.play", { viewing: state.viewing.which, clip_id: clip.clip_id, at_ms: at });
  } else {
    video.pause();
    api.emit("viewing.pause", { viewing: state.viewing.which, clip_id: clip.clip_id, at_ms: at });
  }
}

/* ---- The rAF loop shared by viewing and shaping -------------------------- */

function startLoop(tick) {
  stopLoop();
  const step = () => {
    tick();
    state.raf = requestAnimationFrame(step);
  };
  state.raf = requestAnimationFrame(step);
}

function stopLoop() {
  if (state.raf) cancelAnimationFrame(state.raf);
  state.raf = 0;
}

/* ---- list (spec 4.5) ----------------------------------------------------- */

function renderList() {
  const clip = currentClip();
  state.shaping = null;
  state.viewing = null;
  if (!state.listing || state.listing.clipId !== clip.clip_id) {
    state.listing = { clipId: clip.clip_id, confirming: false, error: null, busy: false, submitted: false };
  }
  if (!state.snapshot.listings[clip.clip_id]) state.snapshot.listings[clip.clip_id] = [];
  renderListColumn();
}

function renderListColumn() {
  const clip = currentClip();
  const entries = state.snapshot.listings[clip.clip_id];
  const working = clear($("working"));
  const stack = el("div", { class: "stack" });
  const listing = state.listing;
  if (listing.confirming) {
    // Done listing asks once (spec 4.5).
    stack.appendChild(el("h1", { class: "heading", text: STRINGS.listingConfirm }));
    stack.appendChild(listedTokens(entries));
    if (listing.error) stack.appendChild(el("p", { class: "body", text: listing.error }));
    const row = el("div");
    row.appendChild(button(STRINGS.nothingElse, { class: "button primary", disabled: listing.busy, onclick: submitListing }));
    row.appendChild(
      button(STRINGS.addMore, {
        onclick: () => {
          listing.confirming = false;
          listing.error = null;
          api.emit("listing.more", { clip_id: clip.clip_id });
          renderListColumn();
        },
      }),
    );
    stack.appendChild(row);
  } else {
    stack.appendChild(el("h1", { class: "heading", text: STRINGS.listingHeading }));
    const field = el("input", {
      class: "field",
      type: "text",
      placeholder: STRINGS.listingField,
      autocomplete: "off",
      autocapitalize: "off",
      spellcheck: "false",
      onkeydown: (event) => {
        if (event.key !== "Enter") return;
        event.preventDefault();
        const text = field.value.trim();
        field.value = "";
        if (!text) return; // empty entries are ignored (contract 10)
        entries.push(text);
        api.emit("listing.add", { clip_id: clip.clip_id, text });
        renderListColumn();
        $("listing-field").focus();
      },
    });
    field.id = "listing-field";
    stack.appendChild(field);
    stack.appendChild(listedTokens(entries));
    stack.appendChild(
      button(STRINGS.doneListing, {
        class: "button primary",
        onclick: () => {
          listing.confirming = true;
          api.emit("listing.confirm", { clip_id: clip.clip_id });
          renderListColumn();
        },
      }),
    );
  }
  working.appendChild(stack);
}

// Each entry is a bracketed token in the participant's own words, the same
// register the skeleton will use (spec 4.5).
function listedTokens(entries) {
  const list = el("div", { class: "listed" });
  entries.forEach((text) => list.appendChild(el("span", { class: "token", text: "[" + text + "]" })));
  return list;
}

// Gating (spec 2, 7; contract 10): the inventory is requested only after the
// listing.submit POST has resolved OK, so the skeleton cannot appear early by
// any path. A 403 or a failed POST shows the error line and leaves Nothing
// else to be pressed again.
async function submitListing() {
  const clip = currentClip();
  const listing = state.listing;
  const sounds = state.snapshot.listings[clip.clip_id].slice();
  listing.busy = true;
  listing.error = null;
  renderListColumn();
  // Emitted once: a retry after a failed POST or fetch must not log a second
  // gating event (the first is still queued when the POST failed).
  if (!listing.submitted) api.emit("listing.submit", { clip_id: clip.clip_id, sounds });
  listing.submitted = true;
  const appended = await api.flush(false);
  let inventory = null;
  if (appended) {
    try {
      inventory = await api.inventory(clip.clip_id);
    } catch (_error) {
      inventory = null;
    }
  }
  listing.busy = false;
  if (!inventory) {
    listing.error = STRINGS.error;
    renderListColumn();
    return;
  }
  state.inventories[clip.clip_id] = inventory;
  afterListing();
}

function afterListing() {
  const clip = currentClip();
  state.snapshot.shot_index = 0;
  if (clip.task === "shaped" && !state.snapshot.seen_author_intro) enter("author_intro");
  else if (clip.task === "control" && !state.snapshot.seen_control_intro) enter("control_intro");
  else openShaping(0);
}

/* ---- shape / control (spec 4.1–4.4, 4.6, 4.8) ---------------------------- */

function openShaping(shotIndex) {
  state.snapshot.shot_index = shotIndex;
  enter("shape");
}

function inventoryShot(clip, shotId) {
  const inventory = state.inventories[clip.clip_id];
  return inventory ? inventory.shots.find((shot) => shot.shot_id === shotId) || null : null;
}

function newShaping(clip, shot, revising, kept) {
  const inventory = state.inventories[clip.clip_id];
  const roles =
    inventory.role_heads ||
    Object.fromEntries(["underneath", "stands_out", "only_here"].map((id, i) => [id, STRINGS.roleHeads[i]]));
  const poles = inventory.poles || STRINGS.poles[clip.task === "control" ? "visual" : "audio"];
  const s = {
    clip,
    shot,
    roles,
    poles,
    level: shot.opening ? shot.opening.level : clip.opening.level,
    balance: shot.opening ? shot.opening.balance : clip.opening.balance,
    admitted: shot.sources.map((source) => source.id),
    strip: null,
    writing: false,
    error: null,
    audition: null,
    revising,
    locked: false, // Keep locks the list (spec 4.8) for the caption's move
    paused: false,
    lastRows: [],
    movedNow: new Set(),
  };
  if (kept) {
    // Revise reopens the shot with its kept settings, fresh (contract 10).
    s.level = kept.settings.level;
    s.admitted = shot.sources.map((source) => source.id).filter((id) => kept.settings.admitted.indexOf(id) >= 0);
    const list = orderingsFor(s);
    const index = list.findIndex((ordering) => ordering.order.join(",") === kept.settings.order.join(","));
    if (index >= 0) s.balance = (list[index].span[0] + list[index].span[1]) / 2;
    s.strip = { caption: kept.caption, key: kept.key, rows: [] };
    s.strip.rows = rowsOf(s);
  }
  s.lastRows = rowsOf(s);
  return s;
}

function renderShape() {
  const clip = currentClip();
  const meta = currentShotMeta();
  const shot = inventoryShot(clip, meta.shot_id);
  if (!shot) {
    // Resume before the inventory is in memory: fetch it, then re-enter. The
    // server still gates this on the logged listing.submit (contract 6).
    const pending = api.inventory(clip.clip_id);
    pending
      .then((inventory) => {
        state.inventories[clip.clip_id] = inventory;
        render();
      })
      .catch(() => {
        // The error line (Table 6) where the skeleton would be, and another
        // request after a pause: nothing else on this screen is interactive.
        clear($("working")).appendChild(el("p", { class: "body", text: STRINGS.error }));
        setTimeout(() => {
          if (state.snapshot.screen === "shape" && !state.shaping) render();
        }, INVENTORY_RETRY_MS);
      });
    return;
  }
  state.viewing = null;
  stageFocusable(false);
  loadClipMedia(clip);
  state.shaping = newShaping(clip, shot, false, null);
  setShotLabel(state.snapshot.shot_index);
  parkCut(null);
  api.emit("shape.open", {
    clip_id: clip.clip_id,
    shot_id: shot.shot_id,
    settings: settingsOf(state.shaping),
    key: currentKey(state.shaping),
  });
  buildSkeleton();
  renderActions();
  renderStrip();
  paintLanes(0);
  startShotLoop();
}

function startShotLoop() {
  const video = $("video");
  const s = state.shaping;
  video.currentTime = s.shot.start_ms / 1000;
  startLoop(shapingTick);
  if (!s.paused) video.play().catch(() => undefined);
}

// Shaping loops the current shot so the moment stays present (spec 3.3): the
// loop clamps currentTime to [start_ms, end_ms).
function shapingTick() {
  const s = state.shaping;
  if (!s) return;
  const video = $("video");
  const ms = video.currentTime * 1000;
  if (ms >= s.shot.end_ms || ms < s.shot.start_ms) video.currentTime = s.shot.start_ms / 1000;
}

function replayShot(emit) {
  const s = state.shaping;
  const video = $("video");
  video.currentTime = s.shot.start_ms / 1000;
  if (s.paused) {
    // Replaying a paused loop runs it again; the pause is over, and logged so.
    s.paused = false;
    reflectPause();
    if (emit) api.emit("loop.resume", { clip_id: s.clip.clip_id, shot_id: s.shot.shot_id });
  }
  video.play().catch(() => undefined);
  if (emit) api.emit("loop.replay", { clip_id: s.clip.clip_id, shot_id: s.shot.shot_id });
}

function togglePauseLoop() {
  const s = state.shaping;
  const video = $("video");
  s.paused = !s.paused;
  if (s.paused) video.pause();
  else video.play().catch(() => undefined);
  reflectPause();
  api.emit(s.paused ? "loop.pause" : "loop.resume", { clip_id: s.clip.clip_id, shot_id: s.shot.shot_id });
}

// Pause loop keeps its one name (Table 6); its state reaches assistive tech
// through aria-pressed, updated in place so the button keeps focus.
function reflectPause() {
  const s = state.shaping;
  const node = $("pause-loop");
  if (s && node) node.setAttribute("aria-pressed", s.paused ? "true" : "false");
}

// Show caption has something to write only while the list has a row to act
// on (spec 2: "only the gestures with something to act on remain"): at
// itemized and grouped an empty admitted set is nothing to write.
function canShow(s) {
  return !!s && !s.locked && !s.writing && rowsOf(s).length > 0;
}

// Actions (spec 4.8): helpers, Show caption (secondary, writing state on the
// button), Keep this caption (the one filled action), criterion beneath.
function renderActions() {
  const s = state.shaping;
  const actions = clear($("actions"));
  actions.hidden = false;
  actions.appendChild(button(STRINGS.replayShot, { onclick: () => replayShot(true) }));
  actions.appendChild(
    button(STRINGS.pauseLoop, { id: "pause-loop", "aria-pressed": s.paused ? "true" : "false", onclick: togglePauseLoop }),
  );
  actions.appendChild(el("span", { class: "spacer" }));
  const show = button(s.writing ? STRINGS.writing : STRINGS.showCaption, {
    id: "show-caption",
    disabled: !canShow(s),
    onclick: showCaption,
  });
  actions.appendChild(show);
  const keep = button(STRINGS.keepCaption, { id: "keep-caption", class: "button primary", onclick: keepCaption });
  keep.disabled = !isFresh();
  actions.appendChild(keep);
  actions.appendChild(el("p", { class: "body criterion", text: STRINGS.criterion }));
}

function refreshKeep() {
  const keep = $("keep-caption");
  if (keep) keep.disabled = !isFresh();
}

// The parked strip's states (spec 4.4): empty, fresh, stale, auditioning, and
// the error line in the eyebrow position (contract 10).
function renderStrip() {
  const s = state.shaping;
  const capobj = $("caption");
  const eyebrow = $("caption-eyebrow");
  const prose = $("caption-prose");
  if (!s || capobj.parentNode !== $("strip")) return;
  capobj.hidden = false;
  let classes = "caption-object parked";
  if (s.audition) {
    const key = s.audition.id;
    prose.textContent = (s.shot.auditions || {})[key] || "";
    eyebrow.textContent = STRINGS.stripAuditioning;
    classes += " auditioning noted";
  } else if (!s.strip) {
    prose.textContent = "";
    eyebrow.textContent = s.error || STRINGS.stripEmpty;
    classes += " empty";
  } else {
    prose.textContent = s.strip.caption;
    const stale = s.strip.key !== currentKey(s);
    if (s.error) {
      eyebrow.textContent = s.error;
      classes += " noted" + (stale ? " stale" : "");
    } else if (stale) {
      eyebrow.textContent = STRINGS.stripStale;
      classes += " stale noted";
    } else {
      eyebrow.textContent = "";
    }
  }
  capobj.className = classes;
  refreshKeep();
}

// Cut the object into the strip (no motion): used on every shot load and when
// a viewing starts, where nothing is moving yet.
function parkCut(text) {
  const capobj = $("caption");
  $("strip").appendChild(capobj);
  capobj.className = "caption-object parked" + (text ? "" : " empty");
  capobj.hidden = false;
  $("caption-prose").textContent = text || "";
  $("caption-eyebrow").textContent = text ? "" : STRINGS.stripEmpty;
}

async function showCaption() {
  const s = state.shaping;
  if (!canShow(s)) return;
  const settings = settingsOf(s);
  const key = settingsKey(settings.level, settings.admitted, settings.order);
  const rows = rowsOf(s);
  s.writing = true;
  s.error = null;
  renderActions();
  api.emit("caption.request", { clip_id: s.clip.clip_id, shot_id: s.shot.shot_id, settings, key });
  const started = performance.now();
  try {
    const written = await api.caption(s.clip.clip_id, s.shot.shot_id, settings);
    if (state.shaping !== s) return;
    s.strip = { caption: written.caption, key: written.key, rows };
    // The rows are the ones the caption was written for: under reduced motion
    // the last change's marks would otherwise read as rows it no longer
    // describes (spec 4.4 reserves the mark for that).
    s.movedNow = new Set();
    api.emit("caption.written", {
      clip_id: s.clip.clip_id,
      shot_id: s.shot.shot_id,
      key: written.key,
      caption: written.caption,
      cached: !!written.cached,
      latency_ms: Math.round(performance.now() - started),
    });
  } catch (error) {
    if (state.shaping !== s) return;
    s.error = STRINGS.error;
    api.emit("caption.failed", { clip_id: s.clip.clip_id, shot_id: s.shot.shot_id, key, error: String(error && error.message) });
  }
  s.writing = false;
  renderActions();
  renderStrip();
  markRows();
}

// Keep (spec 4.8, Table 5): locks the list, places the caption, fills the
// segment, advances — or, when revising, re-places and resumes playback.
async function keepCaption() {
  const s = state.shaping;
  if (!s || s.locked) return;
  if (!isFresh()) return;
  const settings = settingsOf(s);
  const key = currentKey(s);
  const caption = s.strip.caption;
  const clipId = s.clip.clip_id;
  const shotId = s.shot.shot_id;
  // Keeping locks the list (spec 4.8): no gesture lands while the caption
  // moves, so nothing can be logged for this shot after caption.keep.
  s.locked = true;
  api.emit("caption.keep", { clip_id: clipId, shot_id: shotId, settings, key, caption, revision: s.revising });
  if (!state.snapshot.kept[clipId]) state.snapshot.kept[clipId] = {};
  state.snapshot.kept[clipId][shotId] = { caption, settings, key, revised: s.revising };
  if (s.revising) state.snapshot.revisions[clipId] = (state.snapshot.revisions[clipId] || 0) + 1;
  const revising = s.revising;
  const next = state.snapshot.shot_index + 1;
  if (!revising) {
    // The snapshot advances before the motion: a flush or a crash during the
    // slide must resume on the next shot (or the second viewing), never
    // reopen the shot just kept with an empty strip.
    if (next < s.clip.shots.length) state.snapshot.shot_index = next;
    else state.snapshot.screen = "watch2";
  }
  $("actions").hidden = true;
  await placeCaption(caption);
  state.shaping = null;
  paintLanes(0);
  if (revising) {
    // Back to the second viewing where it was paused: the playthrough and
    // the attention answer, if given, stand (state.viewing survives Revise).
    stageFocusable(true);
    renderViewingColumn();
    startLoop(viewingTick);
    $("video").play().catch(() => undefined);
    return;
  }
  if (next < s.clip.shots.length) openShaping(next);
  else enter("watch2");
}

/* ---- The caption's move between its two positions (spec 4.4) ------------- */

function motionCut() {
  return state.reducedMotion;
}

// FLIP on one element: measure where it is, re-parent to where it goes,
// measure again, and play the difference as a 160 ms linear translate.
function moveCaption(target, className) {
  const capobj = $("caption");
  capobj.hidden = false;
  const before = capobj.getBoundingClientRect();
  target.appendChild(capobj);
  capobj.className = className;
  if (motionCut()) return Promise.resolve();
  const after = capobj.getBoundingClientRect();
  const dy = before.top - after.top;
  if (!dy) return Promise.resolve();
  capobj.style.transform = "translateY(" + dy + "px)";
  void capobj.offsetHeight; // commit the start position before animating
  capobj.classList.add("moving"); // identity.css: transform, 160 ms, linear
  capobj.style.transform = "";
  return new Promise((resolve) => {
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      capobj.classList.remove("moving");
      capobj.style.transform = "";
      capobj.removeEventListener("transitionend", finish);
      resolve();
    };
    capobj.addEventListener("transitionend", finish);
    setTimeout(finish, MOTION_MS + 40);
  });
}

function placeCaption(text) {
  $("caption-prose").textContent = text;
  $("caption-eyebrow").textContent = "";
  return moveCaption($("frame"), "caption-object placed");
}

function dropCaption() {
  return moveCaption($("strip"), "caption-object parked");
}

/* ---- Revise (spec 3.3, 4.4, Table 5) ------------------------------------- */

async function onRevise() {
  const clip = currentClip();
  const video = $("video");
  if (!state.viewing || state.shaping || state.revising) return;
  // The latch: a second press, a frame tap, or Space during the drop and the
  // inventory fetch must not run this chain again or log a play mid-revise.
  state.revising = true;
  const viewing = state.viewing;
  const wasPaused = video.paused;
  video.pause();
  const ms = Math.round(video.currentTime * 1000);
  const index = shotAt(clip, ms);
  const meta = clip.shots[index];
  const kept = keptFor(clip.clip_id, meta.shot_id);
  let shot = inventoryShot(clip, meta.shot_id);
  if (!shot) {
    try {
      state.inventories[clip.clip_id] = await api.inventory(clip.clip_id);
    } catch (_error) {
      state.revising = false;
      if (state.viewing !== viewing) return;
      // The viewing is where it was, with the error line (Table 6); no pause
      // was logged, so the footage resumes if it was running.
      viewing.error = STRINGS.error;
      renderViewingColumn();
      if (!wasPaused) video.play().catch(() => undefined);
      return;
    }
    shot = inventoryShot(clip, meta.shot_id);
  }
  if (state.viewing !== viewing) {
    state.revising = false;
    return;
  }
  if (!shot || !kept) {
    state.revising = false;
    if (!wasPaused) video.play().catch(() => undefined);
    return;
  }
  viewing.error = null;
  api.emit("revise.open", { clip_id: clip.clip_id, shot_id: meta.shot_id, at_ms: ms });
  state.snapshot.shot_index = index;
  stopLoop();
  stageFocusable(false);
  await dropCaption();
  state.shaping = newShaping(clip, shot, true, kept);
  state.revising = false;
  setShotLabel(index);
  buildSkeleton();
  renderActions();
  renderStrip();
  paintLanes(ms);
  startShotLoop();
}

/* ---- Skeleton rendering (spec 4.1–4.3, 4.6) ------------------------------ */

function buildSkeleton() {
  const s = state.shaping;
  const working = clear($("working"));
  const skeleton = el("div", { class: "skeleton", id: "skeleton" });
  const labels = el("div", { class: "orderings-labels", id: "orderings-labels" }, [
    el("span", { class: "eyebrow", text: s.poles[0] }),
    el("span", { class: "eyebrow", text: s.poles[1] }),
  ]);
  const rows = el("div", { class: "rows", id: "rows" });
  const orderings = el("div", { class: "orderings", id: "orderings" });
  const body = el("div", { class: "skeleton-body" }, [rows, orderings]);
  const fold = el("div", { class: "fold", id: "fold" }, [
    el("button", { type: "button", class: "fold-handle", id: "fold-handle" }, [el("span", { class: "bar" })]),
    el("span", { class: "eyebrow", id: "fold-level" }),
  ]);
  const struck = el("div", { class: "struck-rows hairline", id: "struck-rows" });
  // At atmospheric the handle sits at its top position above the one row
  // (spec 4.3); at every other level it is at the foot of the admitted rows.
  if (s.level === "atmospheric") skeleton.append(labels, fold, body, struck);
  else skeleton.append(labels, body, fold, struck);
  working.appendChild(skeleton);
  wireFold();
  wirePinch(skeleton);
  renderRows(false);
}

function rowNode(id, source, extraClass) {
  const node = el("button", { type: "button", class: "row " + (extraClass || ""), "data-id": id }, [
    el("span", { class: "token", text: "[" + source.token + "]" }),
    el("span", { class: "phrases eyebrow", text: source.phrases.join(" · ") }),
  ]);
  wireRow(node);
  return node;
}

function headNode(s, roleId, struckAll) {
  const node = el("button", { type: "button", class: "row head" + (struckAll ? " struck" : ""), "data-id": "role:" + roleId }, [
    // spec-identity.md Table 5: a role head is set as a token — bracketed, uppercase.
    el("span", { class: "token", text: "[" + s.roles[roleId] + "]" }),
  ]);
  wireRow(node);
  return node;
}

function struckIds(s) {
  return s.shot.sources.map((source) => source.id).filter((id) => s.admitted.indexOf(id) < 0);
}

// Rows are rebuilt on every change; the FLIP (or the reduced-motion mark)
// makes the reorder legible as movement (spec 7).
function renderRows(animate) {
  const s = state.shaping;
  const rowsBox = $("rows");
  const struckBox = $("struck-rows");
  const before = new Map();
  if (animate) {
    $("skeleton")
      .querySelectorAll(".row[data-id]")
      .forEach((node) => before.set(node.getAttribute("data-id"), node.getBoundingClientRect().top));
  }
  clear(rowsBox);
  clear(struckBox);
  if (s.level === "itemized") {
    currentOrder(s).forEach((id) => rowsBox.appendChild(rowNode(id, sourceById(s, id))));
    struckIds(s).forEach((id) => struckBox.appendChild(rowNode(id, sourceById(s, id), "struck")));
  } else if (s.level === "grouped") {
    currentOrder(s).forEach((roleId) => {
      rowsBox.appendChild(headNode(s, roleId, false));
      membersOf(s, roleId).forEach((id) => rowsBox.appendChild(rowNode(id, sourceById(s, id), "member")));
    });
    // Struck: heads whose members are all struck, with their members beneath;
    // then struck members whose head is still admitted (spec 4.3).
    const struck = struckIds(s);
    roleIds(s).forEach((roleId) => {
      const members = s.shot.sources.filter((source) => source.role === roleId).map((source) => source.id);
      if (!members.length) return;
      const allStruck = members.every((id) => struck.indexOf(id) >= 0);
      if (allStruck) {
        struckBox.appendChild(headNode(s, roleId, true));
        members.forEach((id) => struckBox.appendChild(rowNode(id, sourceById(s, id), "member struck")));
      } else {
        members.filter((id) => struck.indexOf(id) >= 0).forEach((id) => struckBox.appendChild(rowNode(id, sourceById(s, id), "member struck")));
      }
    });
  } else if (s.level === "scene") {
    // One row naming the scene; nothing to strike (spec 4.3). Struck rows are
    // hidden with the admitted rows and return on unfolding.
    rowsBox.appendChild(el("div", { class: "row", "data-id": "scene" }, [el("span", { class: "token", text: "[" + s.shot.scene.token + "]" })]));
  } else {
    rowsBox.appendChild(
      el("div", { class: "row atmospheric", "data-id": "atmospheric" }, [el("span", { class: "phrases eyebrow", text: s.shot.atmosphere.phrase })]),
    );
  }
  $("fold-level").textContent = STRINGS.levels[s.level];
  // The columns' width can reflow the rows beside them, so they are sized
  // before the FLIP measures where each row ended up.
  drawOrderings();
  if (animate) flipRows(before);
  markRows();
  renderStrip();
}

function flipRows(before) {
  const s = state.shaping;
  const after = rowsOf(s);
  s.movedNow = movedRows(s.lastRows, after);
  s.lastRows = after;
  if (motionCut()) return;
  $("skeleton")
    .querySelectorAll(".row[data-id]")
    .forEach((node) => {
      const id = node.getAttribute("data-id");
      if (!before.has(id)) return;
      const dy = before.get(id) - node.getBoundingClientRect().top;
      if (!dy) return;
      node.style.transform = "translateY(" + dy + "px)";
      void node.offsetHeight;
      node.classList.add("moving");
      node.style.transform = "";
      setTimeout(() => {
        node.classList.remove("moving");
      }, MOTION_MS + 40);
    });
}

// The marking channel (identity.css .row.moved): rows that moved since the
// caption was written while it is stale (spec 4.4), and under reduced motion
// the rows that moved in the last change (spec-identity.md Motion).
function markRows() {
  const s = state.shaping;
  if (!s) return;
  const stale = s.strip && s.strip.key !== currentKey(s);
  const sinceWritten = stale ? movedRows(s.strip.rows, rowsOf(s)) : new Set();
  const recent = motionCut() ? s.movedNow : new Set();
  $("skeleton")
    .querySelectorAll(".row[data-id]")
    .forEach((node) => {
      const id = node.getAttribute("data-id");
      node.classList.toggle("moved", sinceWritten.has(id) || recent.has(id));
    });
}

/* ---- Orderings drawing (spec 4.2): rect, line, polyline only ------------- */

const SVG_NS = "http://www.w3.org/2000/svg";
const COLUMN_MIN_W = 40; // spec 7: ordering columns at least 40 wide
const COLUMN_MAX_W = 48;
const COLUMN_INSET = 4;
const ROWS_MIN_W = 148; // kiosk.css .rows min-width, the room the rows keep

// Columns share the width the rows leave, between 40 and 48 each. With more
// orderings than fit at 40 the drawing scrolls inside its own box rather than
// covering the rows.
function columnWidth(count) {
  const available = $("working").clientWidth - ROWS_MIN_W - 8;
  return Math.max(COLUMN_MIN_W, Math.min(COLUMN_MAX_W, Math.floor(available / Math.max(1, count))));
}

function svgNode(name, attrs) {
  const node = document.createElementNS(SVG_NS, name);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
  return node;
}

function drawOrderings() {
  const s = state.shaping;
  const box = clear($("orderings"));
  const rowsBox = $("rows");
  const list = orderingsFor(s);
  const twoLabels = s.level === "scene" || s.level === "atmospheric";
  const columns = twoLabels ? 0 : list.length;
  const colW = columnWidth(columns);
  const width = Math.max(columns * colW, 2 * colW);
  $("orderings-labels").style.width = width + "px";
  box.style.width = width + "px";
  if (!columns) return; // spec 4.3: collapse to the two labels
  const height = Math.max(rowsBox.offsetHeight, 44);
  const svg = svgNode("svg", { width, height, viewBox: "0 0 " + width + " " + height, "aria-hidden": "true" });
  const selected = resolveOrdering(list, s.balance);
  const crossfader = s.clip.balance_control === "crossfader";
  const rowNodes = Array.from(rowsBox.children);
  const centerOf = (id) => {
    const node = rowNodes.find((candidate) => candidate.getAttribute("data-id") === (s.level === "grouped" ? "role:" + id : id));
    return node ? node.offsetTop + node.offsetHeight / 2 : height / 2;
  };
  const colors = { ink: "var(--ink)", rule: "var(--rule)", bench: "var(--bench)" };
  list.forEach((ordering, i) => {
    const x = i * colW + COLUMN_INSET;
    const w = colW - 2 * COLUMN_INSET;
    if (i === selected) svg.appendChild(svgNode("rect", { x, y: 0, width: w, height, fill: colors.ink }));
    else svg.appendChild(svgNode("rect", { x: x + 0.5, y: 0.5, width: w - 1, height: height - 1, fill: "none", stroke: colors.rule, "stroke-width": 1 }));
  });
  if (!crossfader && list.length > 1) {
    // One line per source through its position in each ordering. The segment
    // over the selected (filled) column is drawn in bench so it stays legible.
    const ids = list[0].order;
    ids.forEach((id) => {
      const ys = list.map((ordering) => {
        const position = ordering.order.indexOf(id);
        const shownId = list[selected].order[position];
        return centerOf(shownId);
      });
      for (let i = 0; i + 1 < list.length; i += 1) {
        const x0 = i * colW + colW / 2;
        const x1 = (i + 1) * colW + colW / 2;
        const xb = (i + 1) * colW;
        const yb = (ys[i] + ys[i + 1]) / 2;
        svg.appendChild(svgNode("line", { x1: x0, y1: ys[i], x2: xb, y2: yb, stroke: i === selected ? colors.bench : colors.ink, "stroke-width": 1 }));
        svg.appendChild(svgNode("line", { x1: xb, y1: yb, x2: x1, y2: ys[i + 1], stroke: i + 1 === selected ? colors.bench : colors.ink, "stroke-width": 1 }));
      }
    });
  }
  box.appendChild(svg);
  // Columns are buttons: a hit area at least 44 tall over each drawn column.
  list.forEach((ordering, i) => {
    const hit = el("button", {
      type: "button",
      class: "column",
      "aria-pressed": i === selected ? "true" : "false",
      style: "left:" + i * colW + "px;width:" + colW + "px",
      onclick: () => selectOrdering(i),
    });
    box.appendChild(hit);
  });
}

/* ---- Gestures on the skeleton (spec Table 5) ----------------------------- */

// Every change rebuilds the rows and the columns, so the focused node is
// found again by what it was (a row's id, a column's index): a keyboard user
// stays where they were instead of tabbing in from the top.
function focusedTarget() {
  const active = document.activeElement;
  if (!active || !$("skeleton") || !$("skeleton").contains(active)) return null;
  if (active.hasAttribute("data-id")) return { row: active.getAttribute("data-id") };
  if (active.classList.contains("column")) {
    return { column: Array.from($("orderings").querySelectorAll(".column")).indexOf(active) };
  }
  return null;
}

function refocus(target) {
  if (!target) return;
  let node = null;
  if (target.row !== undefined) node = $("skeleton").querySelector('.row[data-id="' + CSS.escape(target.row) + '"]');
  else node = $("orderings").querySelectorAll(".column")[target.column] || null;
  if (node) node.focus({ preventScroll: true });
}

function afterChange() {
  const s = state.shaping;
  const target = focusedTarget();
  // Any change to the list is a new state; the last request's error line no
  // longer describes it (the strip shows the fresh/stale comparison again).
  s.error = null;
  renderRows(true);
  renderActions();
  refocus(target);
}

function strikeOrRestore(id) {
  const s = state.shaping;
  if (!s || s.locked) return;
  if (id.startsWith("role:")) {
    const roleId = id.slice(5);
    const members = s.shot.sources.filter((source) => source.role === roleId).map((source) => source.id);
    const allStruck = members.every((memberId) => s.admitted.indexOf(memberId) < 0);
    if (allStruck) {
      s.admitted = s.shot.sources.map((source) => source.id).filter((sid) => s.admitted.indexOf(sid) >= 0 || members.indexOf(sid) >= 0);
      api.emit("skeleton.restore", { clip_id: s.clip.clip_id, shot_id: s.shot.shot_id, id });
    } else {
      s.admitted = s.admitted.filter((sid) => members.indexOf(sid) < 0);
      api.emit("skeleton.strike", { clip_id: s.clip.clip_id, shot_id: s.shot.shot_id, id });
    }
  } else if (s.admitted.indexOf(id) >= 0) {
    s.admitted = s.admitted.filter((sid) => sid !== id);
    api.emit("skeleton.strike", { clip_id: s.clip.clip_id, shot_id: s.shot.shot_id, id });
  } else {
    // Restore keeps document order so the subset key is order-free.
    s.admitted = s.shot.sources.map((source) => source.id).filter((sid) => sid === id || s.admitted.indexOf(sid) >= 0);
    api.emit("skeleton.restore", { clip_id: s.clip.clip_id, shot_id: s.shot.shot_id, id });
  }
  afterChange();
}

function selectOrdering(index) {
  const s = state.shaping;
  if (!s || s.locked) return;
  const list = orderingsFor(s);
  if (!list[index]) return;
  // Balance is kept as the midpoint of the selected span, so a later change
  // of admission carries the selection to the ordering that still holds it.
  s.balance = (list[index].span[0] + list[index].span[1]) / 2;
  api.emit("skeleton.balance", { clip_id: s.clip.clip_id, shot_id: s.shot.shot_id, index, order: list[index].order });
  afterChange();
}

function setLevel(level, via) {
  const s = state.shaping;
  if (!s || s.locked || level === s.level || !LEVELS.includes(level)) return;
  s.level = level;
  s.error = null;
  api.emit("skeleton.fold", { clip_id: s.clip.clip_id, shot_id: s.shot.shot_id, level, via });
  // The handle's place depends on the level (spec 4.3), so rebuild the frame.
  const before = new Map();
  $("skeleton")
    .querySelectorAll(".row[data-id]")
    .forEach((node) => before.set(node.getAttribute("data-id"), node.getBoundingClientRect().top));
  buildSkeleton();
  flipRows(before);
  markRows();
  renderActions();
  $("fold-handle").focus({ preventScroll: true });
}

function foldBy(steps, via) {
  const s = state.shaping;
  const index = Math.min(LEVELS.length - 1, Math.max(0, LEVELS.indexOf(s.level) + steps));
  setLevel(LEVELS[index], via);
}

// Audition (spec 4.1): a hold >= 400 ms marks the row alone and ghosts the
// one-source sentence in the strip; release restores whatever the strip held.
function beginAudition(node, id, pressedAt) {
  const s = state.shaping;
  if (!s || s.locked || s.audition || pointers.size > 1) return; // a pinch is not a hold
  // held_ms counts from the press: the threshold has already elapsed here.
  s.audition = { id, node, since: pressedAt };
  node.classList.add("held");
  renderStrip();
}

function endAudition() {
  const s = state.shaping;
  if (!s || !s.audition) return;
  const { id, node, since } = s.audition;
  node.classList.remove("held");
  s.audition = null;
  api.emit("skeleton.hold", { clip_id: s.clip.clip_id, shot_id: s.shot.shot_id, id, held_ms: Math.round(performance.now() - since) });
  renderStrip();
}

const pointers = new Map(); // active pointers on the skeleton, for the pinch
let pinched = false; // two pointers were down together; cleared when all lift

function liftPointer(pointerId) {
  pointers.delete(pointerId);
  if (!pointers.size) pinched = false;
}

function wireRow(node) {
  const id = node.getAttribute("data-id");
  let timer = 0;
  let pressed = false;
  const cancelHold = () => {
    clearTimeout(timer);
    timer = 0;
  };
  node.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || pointers.size > 1) return;
    pressed = true;
    const pressedAt = performance.now();
    node.setPointerCapture(event.pointerId);
    timer = setTimeout(() => {
      timer = 0;
      beginAudition(node, id, pressedAt);
    }, HOLD_MS);
  });
  const release = (event) => {
    if (!pressed) return;
    pressed = false;
    const wasHold = timer === 0;
    cancelHold();
    if (state.shaping && state.shaping.audition && state.shaping.audition.node === node) endAudition();
    // A second finger that was down at any point makes this a pinch attempt,
    // not a tap, even after the first finger has already lifted.
    else if (!wasHold && event.type === "pointerup" && pointers.size < 2 && !pinched) strikeOrRestore(id);
  };
  node.addEventListener("pointerup", release);
  node.addEventListener("pointercancel", release);
  // Keyboard: Enter strikes or restores; Space held auditions (Table 5).
  node.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      strikeOrRestore(id);
    } else if (event.key === " ") {
      event.preventDefault();
      if (event.repeat || timer) return;
      pressed = true;
      const pressedAt = performance.now();
      timer = setTimeout(() => {
        timer = 0;
        beginAudition(node, id, pressedAt);
      }, HOLD_MS);
    }
  });
  node.addEventListener("keyup", (event) => {
    if (event.key !== " " || !pressed) return;
    event.preventDefault();
    pressed = false;
    const wasHold = timer === 0;
    cancelHold();
    if (state.shaping && state.shaping.audition && state.shaping.audition.node === node) endAudition();
    else if (!wasHold) strikeOrRestore(id);
  });
  node.addEventListener("click", (event) => event.preventDefault());
}

// The fold handle (spec 4.3): drag upward folds, downward unfolds, one level
// per 40 px; ArrowUp / ArrowDown when focused. The drag is tracked on the
// document, because every level change rebuilds the skeleton (the handle's
// place depends on the level) and the finger must not lose it.
function wireFold() {
  const handle = $("fold-handle");
  handle.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || pointers.size > 1) return;
    const pointerId = event.pointerId;
    const startY = event.clientY;
    const startIndex = LEVELS.indexOf(state.shaping.level);
    const move = (moveEvent) => {
      if (moveEvent.pointerId !== pointerId || !state.shaping) return;
      const steps = Math.trunc((startY - moveEvent.clientY) / FOLD_STEP_PX);
      const index = Math.min(LEVELS.length - 1, Math.max(0, startIndex + steps));
      if (LEVELS[index] !== state.shaping.level) setLevel(LEVELS[index], "drag");
    };
    const stop = (stopEvent) => {
      if (stopEvent.pointerId !== pointerId) return;
      document.removeEventListener("pointermove", move);
      document.removeEventListener("pointerup", stop);
      document.removeEventListener("pointercancel", stop);
    };
    document.addEventListener("pointermove", move);
    document.addEventListener("pointerup", stop);
    document.addEventListener("pointercancel", stop);
  });
  handle.addEventListener("keydown", (event) => {
    if (event.key === "ArrowUp") {
      event.preventDefault();
      foldBy(1, "key");
    } else if (event.key === "ArrowDown") {
      event.preventDefault();
      foldBy(-1, "key");
    }
  });
}

// Pinch on the skeleton (spec 4.3): two pointers, distance shrinking by 40 px
// folds one level, growing by 40 px unfolds.
function wirePinch(skeleton) {
  let baseline = 0;
  const distance = () => {
    const [a, b] = Array.from(pointers.values());
    return Math.hypot(a.x - b.x, a.y - b.y);
  };
  skeleton.addEventListener("pointerdown", (event) => {
    pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    if (pointers.size === 2) {
      pinched = true;
      baseline = distance();
      endAudition();
    }
  });
  skeleton.addEventListener("pointermove", (event) => {
    if (!pointers.has(event.pointerId)) return;
    pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    if (pointers.size !== 2) return;
    const now = distance();
    if (baseline - now >= FOLD_STEP_PX) {
      baseline = now;
      foldBy(1, "pinch");
    } else if (now - baseline >= FOLD_STEP_PX) {
      baseline = now;
      foldBy(-1, "pinch");
    }
  });
  const lift = (event) => liftPointer(event.pointerId);
  skeleton.addEventListener("pointerup", lift);
  skeleton.addEventListener("pointercancel", lift);
  skeleton.addEventListener("lostpointercapture", lift);
}

/* ---- measures (spec 4.7) ------------------------------------------------- */

function measureRow(text, boxes, poles, value, onAnswer) {
  const scale = el("div", { class: "scale", role: "radiogroup" });
  for (let i = 1; i <= boxes; i += 1) {
    const box = el("button", {
      type: "button",
      class: "box",
      role: "radio",
      "aria-checked": value === i ? "true" : "false",
      onclick: () => onAnswer(i),
    });
    if (i === 1 || i === boxes) {
      scale.appendChild(el("span", { class: "pole" }, [box, el("span", { class: "eyebrow", text: poles[i === 1 ? 0 : 1] })]));
    } else {
      scale.appendChild(box);
    }
  }
  return el("div", { class: "item measure" }, [el("p", { class: "body text", text }), scale]);
}

function renderMeasures() {
  const clip = currentClip();
  state.shaping = null;
  state.viewing = null;
  const record = state.snapshot.measures[clip.clip_id] || { answers: {}, open: "" };
  state.snapshot.measures[clip.clip_id] = record;
  const working = clear($("working"));
  const stack = el("div", { class: "stack" });
  const items = state.session.measures.items;
  items.forEach((item) => {
    stack.appendChild(
      measureRow(item.text, item.boxes, item.poles, record.answers[item.id] || 0, (value) => {
        record.answers[item.id] = value;
        api.emit("measures.answer", { clip_id: clip.clip_id, item_id: item.id, value });
        renderMeasures();
      }),
    );
  });
  const open = el("div", { class: "item" }, [el("p", { class: "body text", text: state.session.measures.open_item || STRINGS.openItem })]);
  const area = el("textarea", { class: "field", rows: 3 });
  area.value = record.open || "";
  // The text is snapshotted as it is typed (the "filled" state resumes); the
  // event that records it is emitted when the field is left.
  area.addEventListener("input", () => {
    record.open = area.value;
    api.save();
  });
  area.addEventListener("change", () => {
    record.open = area.value;
    api.emit("measures.open", { clip_id: clip.clip_id, text: area.value });
  });
  open.appendChild(area);
  stack.appendChild(open);
  const complete = items.every((item) => record.answers[item.id]);
  stack.appendChild(
    button(STRINGS.continue, {
      class: "button primary",
      disabled: !complete,
      onclick: () => {
        record.open = area.value;
        api.emit("measures.submit", { clip_id: clip.clip_id, answers: Object.assign({}, record.answers), open: record.open });
        state.snapshot.clip_index += 1;
        state.snapshot.shot_index = 0;
        state.listing = null;
        startClip();
      },
    }),
  );
  working.appendChild(stack);
}

/* ---- probe and done (spec Table 4) --------------------------------------- */

function summaryNode() {
  const kept = state.snapshot.kept;
  const clips = Object.keys(kept).filter((clipId) => Object.keys(kept[clipId]).length > 0).length;
  const captions = Object.values(kept).reduce((n, shots) => n + Object.keys(shots).length, 0);
  const revisions = Object.values(state.snapshot.revisions).reduce((n, count) => n + count, 0);
  const cell = (label, count) => el("div", {}, [el("span", { class: "eyebrow", text: label }), el("span", { class: "count", text: String(count) })]);
  return el("div", { class: "summary" }, [
    cell(STRINGS.summary.clips, clips),
    cell(STRINGS.summary.kept, captions),
    cell(STRINGS.summary.revisions, revisions),
  ]);
}

// The one download action (spec 7). Secondary beside Submit on the probe
// screen; on the done screen it is the screen's single action, so it is the
// filled primary there (spec-identity.md Actions: one filled primary per screen).
function downloadLink(primary) {
  const href = "/api/log?participant=" + encodeURIComponent(state.participant);
  return el("a", {
    class: primary ? "button primary" : "button",
    href,
    download: "",
    text: STRINGS.downloadLog,
    onclick: () => api.emit("log.download", {}),
  });
}

function renderProbe() {
  state.shaping = null;
  state.viewing = null;
  const working = clear($("working"));
  const stack = el("div", { class: "stack" });
  stack.appendChild(el("p", { class: "body", text: STRINGS.probe }));
  const area = el("textarea", { class: "field", rows: 5 });
  area.value = state.snapshot.probe || "";
  area.addEventListener("input", () => {
    state.snapshot.probe = area.value;
    api.save();
  });
  stack.appendChild(area);
  stack.appendChild(summaryNode());
  const row = el("div");
  row.appendChild(
    button(STRINGS.submit, {
      class: "button primary",
      onclick: () => {
        state.snapshot.probe = area.value;
        api.emit("probe.submit", { text: area.value });
        state.snapshot.done = true;
        api.emit("session.done", {});
        enter("done");
      },
    }),
  );
  row.appendChild(downloadLink());
  stack.appendChild(row);
  working.appendChild(stack);
}

function renderDone() {
  const working = clear($("working"));
  const stack = el("div", { class: "stack" });
  stack.appendChild(summaryNode());
  stack.appendChild(downloadLink(true));
  working.appendChild(stack);
}

/* ---- Boot: participant, session, resume (build contract 9) --------------- */

function participantFromUrl() {
  const id = new URLSearchParams(window.location.search).get("participant") || "";
  return /^[A-Za-z0-9_-]{1,64}$/.test(id) ? id : null;
}

function wireGlobal() {
  document.addEventListener("contextmenu", (event) => event.preventDefault());
  const video = $("video");
  video.addEventListener("ended", onVideoEnded);
  $("frame").addEventListener("click", () => {
    if (state.viewing && !state.shaping && !state.revising) togglePlayback();
  });
  // The focused frame answers Enter as it answers a tap; Space reaches it
  // through the document handler below, as it reaches the whole viewing.
  $("frame").addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || !state.viewing || state.shaping) return;
    if (state.revising) return;
    event.preventDefault();
    togglePlayback();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key !== " " || !state.viewing || state.shaping || state.revising) return;
    const tag = (event.target.tagName || "").toLowerCase();
    if (tag === "button" || tag === "input" || tag === "textarea" || tag === "a") return;
    event.preventDefault();
    togglePlayback();
  });
  // A pointer that lifts anywhere leaves the pinch map, so a finger that left
  // the skeleton mid-gesture cannot leave the next hold reading as a pinch.
  window.addEventListener("pointerup", (event) => liftPointer(event.pointerId));
  window.addEventListener("pointercancel", (event) => liftPointer(event.pointerId));
  const flushNow = () => api.flush(true);
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") flushNow();
  });
  window.addEventListener("pagehide", flushNow);
}

async function boot() {
  state.reducedMotion = !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  state.participant = participantFromUrl();
  if (!state.participant) {
    clear($("working")).appendChild(el("p", { class: "body", text: STRINGS.participantMissing }));
    $("main").classList.add("single");
    return;
  }
  wireGlobal();
  document.title = STRINGS.appTitle;
  try {
    state.session = await api.session();
  } catch (_error) {
    clear($("working")).appendChild(el("p", { class: "body", text: STRINGS.error }));
    $("main").classList.add("single");
    return;
  }
  let stored = null;
  try {
    stored = await api.state();
  } catch (_error) {
    stored = null;
  }
  const snapshot = stored && stored.snapshot;
  if (snapshot && snapshot.schema === SNAPSHOT_SCHEMA) {
    state.snapshot = snapshot;
    // Resume re-enters the interrupted step from its start (contract 9): the
    // snapshot carries the indices, the inventory is fetched again on demand.
    enter(snapshot.screen);
    return;
  }
  state.snapshot = newSnapshot(state.participant, state.session.session_id);
  enter("intro");
}

boot();
